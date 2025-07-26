# ==============================================================
#
#。 1. Load "best_policy.txt"
#   2. Verify if the variance of the LLM score is stable (mean <= THRESHOLD)
#   3. If unstable, automatically refine and update "best_policy.txt"
#   4. Proceed with generating synthetic data -> LLM scoring -> score normalization -> Student training -> test evaluation
# 
#   python normalized.py
#   python normalized.py --skip-refine
#   python normalized.py --refine-only
#
# ==============================================================
import os, time, json, re, pathlib, random, argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm
import openai

# ==============================================================
# ----------- Adjustable parameters / constants ---------------
# ==============================================================
openai.api_key = ""


GT_PATH         = "./Y_real_scaled_ECfluxnet_Combined_0.npy"
TEST_PATH       = "./sampled_combined_CO2_0_5.npy"
SAVE_DIR        = pathlib.Path("distil_outputs"); SAVE_DIR.mkdir(exist_ok=True)
BEST_POLICY_PATH = "./best_policy.txt"

VAL_SIZE    = 30        # Sample size of the validation set (to verify the stability of the policy)
REPEATS     = 2         # The number of times each sequence is scored repeatedly
THRESHOLD   = 0.1       
VAL_CACHE   = SAVE_DIR / "val_series.npy"

# Training parameters
SYNTH_TRAIN_N = 200     
TRAIN_EPOCHS  = 100
LR            = 5e-4
BATCH_SIZE    = 16

RNG_SEED      = 42
_PAT = re.compile(r"(?:\*\*)?FINAL_SCORE(?:\*\*)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.I)

# ==============================================================
# -------------------- Policy Refine ---------------------------
# ==============================================================
def load_best_policy() -> str:
    with open(BEST_POLICY_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_best_policy(new_policy: str):
    bak = pathlib.Path(BEST_POLICY_PATH + ".bak")
    if not bak.exists():
        bak.write_text(pathlib.Path(BEST_POLICY_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    pathlib.Path(BEST_POLICY_PATH).write_text(new_policy, encoding="utf-8")

def refine_policy(current_policy: str, mean_std: float) -> str:
    """
    Have LLM provide a more stable version of the policy (with the same format), and only return the new policy in the code block.
    """
    summary_json = json.dumps({"avg_sigma": round(mean_std, 4)}, indent=2)
    user_prompt = f"""
CURRENT_POLICY:
{current_policy}

It yields avg σ = {mean_std:.3f}. Please rewrite it (same format, keep concise) to reduce variance.
Return **only** the new policy inside the first fenced code block.
Diagnostics:
{summary_json}
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a senior data scientist optimising evaluation metrics."},
            {"role": "user", "content": user_prompt},
        ],
    ).choices[0].message.content.strip()

    m = re.search(r"```(?:\w+)?\n(.*?)```", response, re.S)
    if not m:
        raise RuntimeError("Failed to extract refined policy from LLM response:\n" + response)
    return m.group(1).strip()

# ==============================================================
# -------------------- Data-------------------------------------
# ==============================================================
def load_ground_truth() -> np.ndarray:
    data = np.load(GT_PATH)

    return data[365*5 : 365*6, 0, 1].astype(np.float32)

def build_synthetic(gt: np.ndarray, n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = gt.copy()
        noise_level = rng.uniform(0.05, 0.3)
        base += rng.normal(0, noise_level * gt.std(), gt.shape)
        # shift
        shift = rng.integers(-20, 21)
        base = np.roll(base, shift)
        # scale
        scale = rng.uniform(0.6, 1.4)
        base *= scale
        # seasonal
        seasonal_amp  = rng.uniform(0.0, 0.2)
        seasonal_freq = rng.uniform(0.5, 2.0)
        seasonal = seasonal_amp * gt.std() * np.sin(2 * np.pi * seasonal_freq * np.arange(len(gt)) / 365)
        base += seasonal
        # trend
        if rng.random() < 0.3:
            trend_strength = rng.uniform(-0.1, 0.1)
            base += trend_strength * gt.std() * np.linspace(-1, 1, len(gt))
        # outliers
        if rng.random() < 0.2:
            n_outliers = rng.integers(1, 5)
            idx = rng.choice(len(base), n_outliers, replace=False)
            magnitude = rng.uniform(2, 4) * gt.std()
            signs = rng.choice([-1, 1], n_outliers)
            base[idx] += magnitude * signs
        out.append(base)
    return np.stack(out).astype(np.float32)

def load_holdout() -> np.ndarray:
    arr = np.load(TEST_PATH)          # shape (20, 365, 1) per your note
    return arr[..., 0].astype(np.float32)

def _avg10(x: np.ndarray) -> List[float]:
    n = len(x) // 10
    vals = x[:n*10].reshape(n, 10).mean(1).tolist()
    if len(x) % 10:
        vals.append(float(x[n*10:].mean()))
    return vals

# ==============================================================
# --------- LLM scoring function (with policy parameter) -------
# ==============================================================
def llm_score(ts: np.ndarray, gt: np.ndarray, policy: str, retries: int = 3) -> float:
    ts_simplified = _avg10(ts)
    gt_simplified = _avg10(gt)
    prompt = f"""
You are an ecological-time-series scoring bot.
Follow **every** rule below *exactly*.  
Any deviation (extra words, wrong decimals, wrong key, markdown) must be treated as an error.

────────────────────────────────────────
INPUTS
Ground Truth (10-point averages): {gt_simplified}
Candidate    (10-point averages): {ts_simplified}

────────────────────────────────────────
POLICY TEXT
{policy}
───────────────────────────────────────
IMPLEMENTATION RULES 
R1. Use **only** the numeric vectors above; no external data.  
R2. Peak detection (both series):                              
    • Index i with x[i] > x[i-1] and x[i] > x[i+1].            
    • If multiple peaks ➔ choose the *largest value's index*.  
    • If no peak ➔ use the global-max index.                   
R3. Derivative vector d = x[i+1] - x[i]  for i = 0…N-2.        
R4. range(x) = max(x) - min(x) ; std(x) = population σ.        
R5. All “diff” terms use absolute value |·|.                   
R6. Keep **6 decimal places** for every intermediate number.   
R7. After summing the component scores, **round TOTAL to 4     |
    decimal places (half-up, e.g. 7.85306 → 7.8531)**.         |

────────────────────────────────────────
OUTPUT SPECIFICATION
Return **exactly one line**, no Markdown, no explanation, no units:  
FINAL_SCORE: <TOTAL_SCORE with 4 decimals>

Valid example (do **not** repeat):  
FINAL_SCORE: 7.8531
────────────────────────────────────────
"""
    for _ in range(retries):
        txt = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role":"system","content":"You are a meticulous expert."},
                {"role":"user","content":prompt}
            ]
        ).choices[0].message.content
        m = _PAT.search(txt)
        if m:
            return float(m.group(1))
    return -1.0

# ==============================================================
# ----------- Policy stability assessment ----------------------
# ==============================================================
def load_or_build_val_series(gt: np.ndarray) -> np.ndarray:
    if VAL_CACHE.exists():
        return np.load(VAL_CACHE)
    val = build_synthetic(gt, n=VAL_SIZE, seed=123)
    np.save(VAL_CACHE, val.astype(np.float32))
    return val

def collect_val_scores(val_series: np.ndarray, gt: np.ndarray, policy: str, repeats: int) -> List[List[float]]:
    nest = []
    for ts in tqdm(val_series, desc=f"LLM scoring {len(val_series)} val series"):
        scores = []
        for _ in range(repeats):
            s = llm_score(ts, gt, policy)
            scores.append(s)
        nest.append(scores)
    return nest

def compute_variability(score_nest: List[List[float]]):
    stds = np.array([np.std(s) for s in score_nest], dtype=np.float32)
    return stds, float(stds.mean())

def ensure_policy_stable(gt: np.ndarray, skip_refine: bool = False) -> str:
    current_policy = load_best_policy()
    if skip_refine:
        print("Skip the policy stability check/optimization.")
        return current_policy

    print("Checking the stability of the current policy...")
    val_series = load_or_build_val_series(gt)
    score_nest = collect_val_scores(val_series, gt, current_policy, REPEATS)
    stds, mean_std = compute_variability(score_nest)
    SAVE_DIR.joinpath("val_score_stds.json").write_text(json.dumps(stds.tolist(), indent=2))

    print(f"Mean σ = {mean_std:.3f} (threshold {THRESHOLD})")
    if mean_std <= THRESHOLD:
        print("Policy stable - no refinement needed.")
        return current_policy

    print("✱ Policy unstable - refining via GPT-4o …")
    new_policy = refine_policy(current_policy, mean_std)
    SAVE_DIR.joinpath("new_policy.txt").write_text(new_policy, encoding="utf-8")

    if new_policy == current_policy:
        print("The returned content of LLM was the same as the existing policy, so we decided not to update.")
        return current_policy

    save_best_policy(new_policy)
    print("The file best_policy.txt has been updated (with a backup file named .bak). ")
    return new_policy

# ==============================================================
# -------------------- Normalize -------------------------------
# ==============================================================
def normalize_scores(scores, s_min, s_max):
    denom = max(s_max - s_min, 1e-8)
    return [(s - s_min) / denom for s in scores]

def denormalize_scores(norm_scores, s_min, s_max):
    return [s * (s_max - s_min) + s_min for s in norm_scores]

# ==============================================================
# -------------------- Dataset / Model -------------------------
# ==============================================================
class TSDataset(Dataset):
    def __init__(self, series: np.ndarray, scores: np.ndarray):
        self.x = torch.tensor(series, dtype=torch.float32)
        self.y = torch.tensor(scores,  dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, input_size=365):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.1),
        )
        self.stat_features = nn.Linear(6, 32)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(),
            nn.Linear(64, 32),       nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        seq_features = self.feature_extractor(x)
        batch, _ = x.shape
        stat = torch.zeros(batch, 6, device=x.device)
        for i in range(batch):
            ts = x[i]
            mu, sigma = ts.mean(), ts.std()
            stat[i, 0:4] = torch.tensor([mu, sigma, ts.min(), ts.max()])
            centered = ts - mu
            stat[i, 4] = (centered**3).mean() / (sigma**3 + 1e-8)
            stat[i, 5] = (centered**4).mean() / (sigma**4 + 1e-8)
        stat = self.stat_features(stat)
        combined = torch.cat([seq_features, stat], 1)
        return self.classifier(combined).squeeze(-1)

def spearman(a, b): return float(spearmanr(a, b)[0])

def train_student(ds: Dataset, epochs=100, lr=5e-4, bs=16):
    train_size = int(0.8 * len(ds))
    val_size   = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_dl = DataLoader(train_ds, bs, shuffle=True)
    val_dl   = DataLoader(val_ds,   bs, shuffle=False)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MLP().to(dev)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.7)
    lossf = nn.MSELoss()

    best_rho, best_state, wait = -1, None, 0
    for ep in range(1, epochs+1):
        net.train(); tot = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pred = net(x)
            loss = lossf(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(x)
        train_loss = tot / train_size
        net.eval(); vp, vt = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                p = net(x); vp.extend(p.cpu()); vt.extend(y.cpu())
        vp, vt = torch.stack(vp).numpy(), torch.stack(vt).numpy()
        val_loss = lossf(torch.tensor(vp), torch.tensor(vt)).item()
        rho = spearman(vp, vt)
        scheduler.step(val_loss)

        if rho > best_rho:
            best_rho, best_state, wait = rho, net.state_dict().copy(), 0
        else:
            wait += 1
        if wait >= 25:
            print(f"Early stop @ {ep}")
            break
        if ep % 10 == 0:
            print(f"Epoch {ep:3d} | train MSE={train_loss:.4f} | val MSE={val_loss:.4f} | rho={rho:.3f}")
    net.load_state_dict(best_state)
    print(f"Best validation ρ = {best_rho:.3f}")
    return net

# ==============================================================
# -------------------- Main ----------------------------------
# ==============================================================
def run(args):
    random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

    print("Loading ground truth ...")
    gt = load_ground_truth()
    print(f"GT shape={gt.shape}  range=[{gt.min():.3f}, {gt.max():.3f}]")

    # 1. Policy stability assessment / refinement
    policy = ensure_policy_stable(gt, skip_refine=args.skip_refine)
    if args.refine_only:
        print("已完成 refine-only 流程，程序结束。")
        return

    # 2. Construct a synthetic training set
    print("Building synthetic training set ...")
    train_x = build_synthetic(gt, n=SYNTH_TRAIN_N, seed=RNG_SEED)
    print(f"Train-X shape={train_x.shape}")

    # 3. LLM scoring
    print("Querying LLM for training scores ...")
    train_y = []
    failed = 0
    for s in tqdm(train_x, desc="LLM-train"):
        sc = llm_score(s, gt, policy)
        if sc == -1:
            failed += 1
            sc = max(0, np.corrcoef(s, gt)[0, 1] * 10)
        train_y.append(sc)
    print(f"LLM train scores  mean={np.mean(train_y):.3f}  std={np.std(train_y):.3f}  "
          f"range=[{min(train_y):.2f}, {max(train_y):.2f}]  (failed={failed})")

    # 4. Normalize score
    s_min, s_max = min(train_y), max(train_y)
    train_y_norm = normalize_scores(train_y, s_min, s_max)
    ds = TSDataset(train_x, np.array(train_y_norm, np.float32))

    # 5. Training Student
    print("Training student model ...")
    student = train_student(ds, epochs=TRAIN_EPOCHS, lr=LR, bs=BATCH_SIZE)

    # 6. Hold-out Training
    print("Loading hold-out test series ...")
    test_x = load_holdout()
    print(f"Test-X shape={test_x.shape}")

    print("LLM scoring test set ...")
    test_y = []
    for s in tqdm(test_x, desc="LLM-test"):
        sc = llm_score(s, gt, policy)
        if sc == -1:
            sc = max(0, np.corrcoef(s, gt)[0, 1] * 10)
        test_y.append(sc)

    # 7. Student Prediction & Denormalization
    student.eval()
    with torch.no_grad():
        dev = next(student.parameters()).device
        preds_norm = student(torch.tensor(test_x, dtype=torch.float32).to(dev)).cpu().numpy()
    preds = denormalize_scores(preds_norm, s_min, s_max)

    # 8. Evaluation
    rho = spearman(test_y, preds)
    print(f"Spearman (LLM vs Student) on {len(test_x)} samples: {rho:.3f}")

    # 9. Save
    print("Writing outputs ...")
    np.save(SAVE_DIR / "train_series.npy", train_x.astype(np.float32))
    np.save(SAVE_DIR / "test_series.npy",  test_x.astype(np.float32))

    def _json_dump(name, data):
        with open(SAVE_DIR / name, "w") as f:
            json.dump([float(x) for x in data], f, indent=2)

    _json_dump("train_scores_raw.json",  train_y)
    _json_dump("train_scores_norm.json", train_y_norm)
    _json_dump("test_scores_llm.json",   test_y)
    _json_dump("test_scores_student.json", preds)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_x),
        "test_samples":  len(test_x),
        "failed_llm_queries": failed,
        "score_range": {"min": s_min, "max": s_max},
        "spearman": rho,
        "policy_mean_sigma_threshold": THRESHOLD,
        "policy_refinement_skipped": args.skip_refine,
    }
    with open(SAVE_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All artifacts saved to", SAVE_DIR.resolve())
    print("Done")

# ==============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refine", action="store_true",
                    help="Skip policy stability check/optimization (direct training)")
    ap.add_argument("--refine-only", action="store_true",
                    help="Only conduct stability testing + possible refinement, without subsequent training")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
