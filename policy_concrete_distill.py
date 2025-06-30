
import os, time, json, re, pathlib, random
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import openai             
from tqdm import tqdm
from scipy.stats import spearmanr

openai.api_key = ""
_PAT = re.compile(r"(?:\*\*)?FINAL_SCORE(?:\*\*)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.I)

GT_PATH   = "/home/qic69/projects/domain_metric/data/GT/Y_real_scaled_ECfluxnet_Combined_0.npy"
TEST_PATH = "/home/qic69/projects/domain_metric/data/sampled_combined_CO2_0_5.npy"

SAVE_DIR  = pathlib.Path("distil_outputs")
SAVE_DIR.mkdir(exist_ok=True)


BEST_POLICY = r"""
METRICS:
1. Peak Period Alignment
2. Derivative Consistency
3. Amplitude Stability
4. Tolerance Level
5. Correlation with Target

FORMULA:
1. Peak Period Alignment Score = exp(-|observed_peak_period - target_peak_period| / target_peak_period)
2. Derivative Consistency Score = exp(-|mean_derivative(TS) - mean_derivative(target)| / mean_derivative(target))
3. Amplitude Stability Score = exp(-|std_dev_amplitude(TS) - std_dev_amplitude(target)| / std_dev_amplitude(target))
4. Tolerance Level Score = 1 - |tolerance_ratio(TS, target) - 1|
5. Correlation with Target Score = Pearson_correlation(TS, target)

SCORING:
1. Peak Period Alignment Score: 1.5 points
2. Derivative Consistency Score: 1.5 points
3. Amplitude Stability Score: 1.5 points
4. Tolerance Level Score: 1.5 points
5. Correlation with Target Score: 4 points

DECISION:
- Total Score = Peak Period Alignment Score * 1.5 + Derivative Consistency Score * 1.5 + Amplitude Stability Score * 1.5 + Tolerance Level Score * 1.5 + Correlation with Target Score * 4
- Choose the series with the higher total score.
- In case of a tie, select the series with the higher Correlation with Target Score. If still tied, apply user-defined secondary criteria.
"""

def load_ground_truth() -> np.ndarray:
    data = np.load(GT_PATH)
    return data[365*5 : 365*6, 0, 1].astype(np.float32)

def build_synthetic(gt: np.ndarray, n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.roll(gt + rng.normal(0, 0.1*gt.std(), gt.shape),
                    rng.integers(-10, 10))
        x *= rng.uniform(0.8, 1.2)
        out.append(x)
    return np.stack(out).astype(np.float32)

def load_holdout() -> np.ndarray:
    arr = np.load(TEST_PATH)          # (20, 365, 1)
    return arr[..., 0].astype(np.float32)

def _avg10(x: np.ndarray) -> List[float]:
    n = len(x) // 10
    vals = x[:n*10].reshape(n, 10).mean(1).tolist()
    if len(x) % 10:
        vals.append(float(x[n*10:].mean()))
    return vals

def llm_score(ts: np.ndarray, gt: np.ndarray, retries: int = 1) -> float:
    prompt = f"""
            You are an expert evaluator. Using the policy below, compute the similarity score between a candidate time series and the ground truth.

            {BEST_POLICY}

            Now evaluate:

            Ground Truth (as Python list of floats):
            {gt.tolist()}

            Candidate Series (as Python list of floats):
            {ts.tolist()}

            Your task:
            - Compute **all metric scores** numerically using the formulas in the policy.
            - Use NumPy-style calculations for any derivative, stddev, tolerance, or correlation.
            - Plug all values into the weighted sum.
            - Return **only** the final result in this exact format:

            FINAL_SCORE: <float>

            Do not explain anything. Do not output markdown. Just return:
            FINAL_SCORE: <float>  with actual number.
            """
    
    # print("LLM prompt:")
    # print(prompt)

    for _ in range(retries):

        txt = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role":"system","content":"You are a meticulous expert."},
                {"role":"user",  "content":prompt}
            ]
        ).choices[0].message.content
        m = _PAT.search(txt)
        # print(txt)
        if m:
            return float(m.group(1))
        
    return -1.0


class TSDataset(Dataset):
    def __init__(self, series: np.ndarray, scores: np.ndarray):
        self.x = torch.tensor(series, dtype=torch.float32)   
        self.y = torch.tensor(scores, dtype=torch.float32)   
    def __len__(self):              return len(self.x)
    def __getitem__(self, i):       return self.x[i], self.y[i]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(365, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64 , 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_student(ds: Dataset, epochs=50, lr=3e-4, bs=32) -> nn.Module:
    dl  = DataLoader(ds, bs, shuffle=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = MLP().to(dev)                         
    opt = optim.Adam(net.parameters(), lr=lr)   
    lossf = nn.MSELoss()

    for ep in range(1, epochs + 1):
        net.train(); running = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()

            pred  = net(x)
            loss  = lossf(pred, y)
            loss.backward()
            opt.step()

            running += loss.item() * len(x)

        if ep % 10 == 0:
            print(f"Epoch {ep:3d}/{epochs}  train MSE={running/len(ds):.4f}")

    return net


def spearman(a, b): return float(spearmanr(a, b)[0])


def main():
    print("Loading ground truth")
    gt = load_ground_truth()

    print("Building synthetic training set")
    train_x = build_synthetic(gt, n=5)

    print("Querying LLM for training scores")
    train_y = [llm_score(s, gt) for s in tqdm(train_x)]
    print("LLM scores for training set:", train_y)

    print("Training student model")
    student = train_student(TSDataset(train_x, np.array(train_y, dtype=np.float32)))
    print("Loading 20 test series")
    test_x = load_holdout()

    print("LLM scoring of test set")
    test_y_llm = [llm_score(s, gt) for s in tqdm(test_x)]

    print("Student predictions on test set")
    student.eval()
    with torch.no_grad():
        dev = next(student.parameters()).device
        test_pred = student(torch.tensor(test_x, dtype=torch.float32).to(dev)).cpu().numpy()

    rho = spearman(test_y_llm, test_pred)
    print(f"Spearman score between LLM and student on 20 samples: {rho:.3f}")

    np.save(SAVE_DIR / "train_series.npy", train_x.astype(np.float32))
    np.save(SAVE_DIR / "test_series.npy",  test_x.astype(np.float32))

    for fname, data in [
        ("train_scores.json",         train_y),
        ("test_scores_llm.json",      test_y_llm),
        ("test_scores_student.json",  test_pred.tolist())
    ]:
        with open(SAVE_DIR / fname, "w") as f:
            json.dump(data, f, indent=2)

    with open(SAVE_DIR / "best_policy.txt", "w") as f:
        f.write(BEST_POLICY.strip() + "\n")

    meta = {
        "spearman": rho,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_n": len(train_x),
        "test_n":  len(test_x)
    }
    with open(SAVE_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All files written to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
