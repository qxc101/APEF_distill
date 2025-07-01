
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
1. Peak Period Alignment - How well do the peak periods match?
2. Derivative Consistency - How similar are the rates of change?
3. Amplitude Stability - How similar are the value ranges?
4. Tolerance Level - How similar are the variations?
5. Correlation with Target - How well do values correlate?

SCORING FORMULAS:
1. Peak Period Alignment Score = max(0, 1 - |peak_diff| / max_peak) * 1.5
2. Derivative Consistency Score = max(0, 1 - |deriv_diff| / max_deriv) * 1.5  
3. Amplitude Stability Score = max(0, 1 - |range_diff| / max_range) * 1.5
4. Tolerance Level Score = max(0, 1 - |std_diff| / max_std) * 1.5
5. Correlation Score = max(0, correlation_coefficient) * 4.0

CALCULATION STEPS:
- Find peaks in both series (local maxima)
- Calculate derivatives using adjacent differences
- Compute ranges (max - min) and standard deviations
- Calculate Pearson correlation coefficient
- Apply formulas above and sum all scores

TOTAL SCORE = Sum of all 5 component scores (Range: 0-10)
"""

def load_ground_truth() -> np.ndarray:
    data = np.load(GT_PATH)
    return data[365*5 : 365*6, 0, 1].astype(np.float32)

def build_synthetic(gt: np.ndarray, n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        # Create more diverse synthetic data with different types of variations
        base = gt.copy()
        
        # Add various types of noise and transformations to create diversity
        noise_level = rng.uniform(0.05, 0.3)  # More varied noise levels
        base += rng.normal(0, noise_level * gt.std(), gt.shape)
        
        # Time shifts
        shift = rng.integers(-20, 21)
        base = np.roll(base, shift)
        
        # Amplitude scaling with more variation
        scale = rng.uniform(0.6, 1.4)
        base *= scale
        
        # Add seasonal variations
        seasonal_amp = rng.uniform(0.0, 0.2)
        seasonal_freq = rng.uniform(0.5, 2.0)
        seasonal = seasonal_amp * gt.std() * np.sin(2 * np.pi * seasonal_freq * np.arange(len(gt)) / 365)
        base += seasonal
        
        # Add trend variations
        if rng.random() < 0.3:  # 30% chance of trend
            trend_strength = rng.uniform(-0.1, 0.1)
            trend = trend_strength * gt.std() * np.linspace(-1, 1, len(gt))
            base += trend
        
        # Add outliers occasionally
        if rng.random() < 0.2:  # 20% chance of outliers
            n_outliers = rng.integers(1, 5)
            outlier_indices = rng.choice(len(base), n_outliers, replace=False)
            outlier_magnitude = rng.uniform(2, 4) * gt.std()
            outlier_sign = rng.choice([-1, 1], n_outliers)
            base[outlier_indices] += outlier_magnitude * outlier_sign
        
        out.append(base)
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

def llm_score(ts: np.ndarray, gt: np.ndarray, retries: int = 3) -> float:
    # Simplify the input to make LLM processing more reliable
    ts_simplified = _avg10(ts)
    gt_simplified = _avg10(gt)
    
    prompt = f"""
You are an expert evaluator. Using the policy below, compute the similarity score between a candidate time series and the ground truth.

{BEST_POLICY}

Now evaluate:

Ground Truth (simplified to 10-point averages):
{gt_simplified}

Candidate Series (simplified to 10-point averages):
{ts_simplified}

Your task:
- Compute **all metric scores** numerically using the formulas in the policy.
- For derivatives, use simple differences between adjacent points.
- For correlation, use the standard Pearson correlation formula.
- For peak period alignment, identify the dominant period using simple peak detection.
- For amplitude stability, compute standard deviation of the values.
- For tolerance level, use the ratio of standard deviations.
- Plug all values into the weighted sum formula.
- Return **only** the final result in this exact format:

FINAL_SCORE: <number>

Example: FINAL_SCORE: 7.85

Do not explain your calculations. Just return the final score.
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
    def __init__(self, input_size=365):
        super().__init__()
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Statistical feature computation
        self.stat_features = nn.Linear(6, 32)  # mean, std, min, max, skew, kurt
        
        # Combined processing
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x): 
        # Extract sequential features
        seq_features = self.feature_extractor(x)
        
        # Compute statistical features
        batch_size = x.shape[0]
        stat_feats = torch.zeros(batch_size, 6, device=x.device)
        for i in range(batch_size):
            ts = x[i]
            stat_feats[i, 0] = ts.mean()
            stat_feats[i, 1] = ts.std()
            stat_feats[i, 2] = ts.min()
            stat_feats[i, 3] = ts.max()
            # Approximate skewness and kurtosis
            centered = ts - ts.mean()
            stat_feats[i, 4] = (centered**3).mean() / (ts.std()**3 + 1e-8)
            stat_feats[i, 5] = (centered**4).mean() / (ts.std()**4 + 1e-8)
        
        stat_features = self.stat_features(stat_feats)
        
        # Combine features
        combined = torch.cat([seq_features, stat_features], dim=1)
        
        # Predict score
        raw_output = self.classifier(combined).squeeze(-1)
        
        # Scale output to typical LLM score range (6-10 based on observed data)
        return raw_output * 4.0 + 6.0  # Scale from [0,1] to [6,10]

def train_student(ds: Dataset, epochs=100, lr=5e-4, bs=16) -> nn.Module:
    # Split dataset for validation
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    train_dl = DataLoader(train_ds, bs, shuffle=True)
    val_dl = DataLoader(val_ds, bs, shuffle=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = MLP().to(dev)                         
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.7)
    lossf = nn.MSELoss()

    best_val_corr = -1.0
    best_model_state = None
    patience_counter = 0
    
    for ep in range(1, epochs + 1):
        # Training
        net.train(); train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()

            pred  = net(x)
            loss  = lossf(pred, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

            train_loss += loss.item() * len(x)

        avg_train_loss = train_loss / len(train_ds)
        
        # Validation
        net.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                pred = net(x)
                loss = lossf(pred, y)
                val_loss += loss.item() * len(x)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_ds)
        val_corr = spearman(val_targets, val_preds) if len(val_preds) > 1 else 0
        
        scheduler.step(avg_val_loss)
        
        # Save best model based on correlation
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_model_state = net.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 25:
            print(f"Early stopping at epoch {ep}")
            break

        if ep % 10 == 0:
            print(f"Epoch {ep:3d}/{epochs}  train MSE={avg_train_loss:.4f}  val MSE={avg_val_loss:.4f}  val corr={val_corr:.3f}")

    # Load best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        print(f"Loaded best model with validation correlation: {best_val_corr:.3f}")
    
    return net


def spearman(a, b): return float(spearmanr(a, b)[0])


def main():
    print("Loading ground truth")
    gt = load_ground_truth()
    print(f"Ground truth shape: {gt.shape}, range: [{gt.min():.3f}, {gt.max():.3f}]")

    print("Building synthetic training set")
    train_x = build_synthetic(gt, n=200)
    print(f"Training set shape: {train_x.shape}")

    print("Querying LLM for training scores")
    train_y = []
    failed_queries = 0
    for i, s in enumerate(tqdm(train_x)):
        score = llm_score(s, gt)
        if score == -1.0:
            failed_queries += 1
            # Use a fallback score based on simple correlation
            score = max(0, np.corrcoef(s, gt)[0, 1] * 10)
        train_y.append(score)
    
    print(f"LLM scores for training set (failed queries: {failed_queries}):")
    print(f"Mean: {np.mean(train_y):.3f}, Std: {np.std(train_y):.3f}")
    print(f"Range: [{min(train_y):.3f}, {max(train_y):.3f}]")
    print("Sample scores:", train_y[:20])

    # Filter out invalid scores
    valid_indices = [i for i, score in enumerate(train_y) if score > 0]
    if len(valid_indices) < len(train_y):
        print(f"Filtering out {len(train_y) - len(valid_indices)} invalid scores")
        train_x = train_x[valid_indices]
        train_y = [train_y[i] for i in valid_indices]

    print("Training student model")
    student = train_student(TSDataset(train_x, np.array(train_y, dtype=np.float32)))
    
    print("Loading 20 test series")
    test_x = load_holdout()
    print(f"Test set shape: {test_x.shape}")

    print("LLM scoring of test set")
    test_y_llm = []
    for i, s in enumerate(tqdm(test_x)):
        score = llm_score(s, gt)
        if score == -1.0:
            # Use fallback score
            score = max(0, np.corrcoef(s, gt)[0, 1] * 10)
        test_y_llm.append(score)

    print("Student predictions on test set")
    student.eval()
    with torch.no_grad():
        dev = next(student.parameters()).device
        test_pred = student(torch.tensor(test_x, dtype=torch.float32).to(dev)).cpu().numpy()

    print(f"Test LLM scores - Mean: {np.mean(test_y_llm):.3f}, Std: {np.std(test_y_llm):.3f}")
    print(f"Student predictions - Mean: {np.mean(test_pred):.3f}, Std: {np.std(test_pred):.3f}")
    
    rho = spearman(test_y_llm, test_pred)
    print(f"Spearman score between LLM and student on {len(test_x)} samples: {rho:.3f}")

    # Save results
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
        "test_n":  len(test_x),
        "failed_llm_queries": failed_queries,
        "train_score_stats": {
            "mean": float(np.mean(train_y)),
            "std": float(np.std(train_y)),
            "min": float(min(train_y)),
            "max": float(max(train_y))
        },
        "test_score_stats": {
            "llm_mean": float(np.mean(test_y_llm)),
            "llm_std": float(np.std(test_y_llm)),
            "student_mean": float(np.mean(test_pred)),
            "student_std": float(np.std(test_pred))
        }
    }
    with open(SAVE_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All files written to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
