import os, time, json, re, pathlib, random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm
import openai

openai.api_key = ""
_PAT = re.compile(r"(?:\*\*)?FINAL_SCORE(?:\*\*)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.I)

GT_PATH   = "./Y_real_scaled_ECfluxnet_Combined_0.npy"
TEST_PATH = "./sampled_combined_CO2_0_5.npy"
SAVE_DIR  = pathlib.Path("distil_outputs")
SAVE_DIR.mkdir(exist_ok=True)

# ==========================================
def normalize_scores(scores, s_min, s_max):                          
    denom = max(s_max - s_min, 1e-8)                                 
    return [(s - s_min) / denom for s in scores]                     
def denormalize_scores(norm_scores, s_min, s_max):                   
    return [s * (s_max - s_min) + s_min for s in norm_scores]        

# ==========================================
BEST_POLICY = r"""
NEW_POLICY:
METRICS:
1. Peak Period Alignment - How well do the peak periods match?
2. Derivative Consistency - How similar are the rates of change?
3. Amplitude Stability - How similar are the value ranges?
4. Tolerance Level - How similar are the variations?
5. Correlation with Target - How well do values correlate?

SCORING FORMULAS:
1. Peak Period Alignment Score = max(0, 1 - |peak_diff| / max_peak) * 1.2
2. Derivative Consistency Score = max(0, 1 - |deriv_diff| / max_deriv) * 1.2  
3. Amplitude Stability Score = max(0, 1 - |range_diff| / max_range) * 1.2
4. Tolerance Level Score = max(0, 1 - |std_diff| / max_std) * 1.2
5. Correlation Score = max(0, correlation_coefficient) * 5.2

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
You are an ecological-time-series scoring bot.
Follow **every** rule below *exactly*.  
Any deviation (extra words, wrong decimals, wrong key, markdown) must be treated as an error.

────────────────────────────────────────
INPUTS
Ground Truth (10-point averages): {gt_simplified}
Candidate    (10-point averages): {ts_simplified}

────────────────────────────────────────
POLICY TEXT
{BEST_POLICY}
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
# ==========================================
class TSDataset(Dataset):
    def __init__(self, series: np.ndarray, scores: np.ndarray):
        self.x = torch.tensor(series, dtype=torch.float32)
        self.y = torch.tensor(scores,  dtype=torch.float32)
    def __len__(self):              return len(self.x)
    def __getitem__(self, i):       return self.x[i], self.y[i]

# ==========================================
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
            nn.Linear(128 + 32, 64), 
            nn.ReLU(),
            nn.Linear(64,  32),      
            nn.ReLU(),
            nn.Linear(32,   1),
            nn.Sigmoid()                          # 输出 ∈ (0,1)
        )
    def forward(self, x):
        seq_features = self.feature_extractor(x)
        # 统计特征
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
        return self.classifier(combined).squeeze(-1)  # 直接输出 0-1 归一化分数

# ==========================================
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
        # -- Train
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
        # -- Val
        net.eval(); vp, vt = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                p = net(x); vp.extend(p.cpu()); vt.extend(y.cpu())
        vp, vt = torch.stack(vp).numpy(), torch.stack(vt).numpy()
        val_loss = lossf(torch.tensor(vp), torch.tensor(vt)).item()
        rho = spearman(vp, vt)        # 已在同一归一化空间
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

# ==========================================
def main():
    #载入 Ground-Truth
    print("Loading ground truth ...")
    gt = load_ground_truth()
    print(f"GT shape={gt.shape}  range=[{gt.min():.3f}, {gt.max():.3f}]")

    #构造合成训练集
    print("Building synthetic training set ...")
    train_x = build_synthetic(gt, n=200)
    print(f"Train-X shape={train_x.shape}")

    #调用 LLM 评分（原始 0-10 或其他尺度）
    print("Querying LLM for training scores ...")
    train_y = []
    failed = 0
    for s in tqdm(train_x, desc="LLM-train"):
        sc = llm_score(s, gt)
        if sc == -1:
            failed += 1
            sc = max(0, np.corrcoef(s, gt)[0, 1] * 10)
        train_y.append(sc)
    print(f"LLM train scores  mean={np.mean(train_y):.3f}  std={np.std(train_y):.3f}  "
          f"range=[{min(train_y):.2f}, {max(train_y):.2f}]  (failed={failed})")

    #归一化分数 → Student 训练标签
    s_min, s_max = min(train_y), max(train_y)                  # 记录范围
    train_y_norm = normalize_scores(train_y, s_min, s_max)     # 0-1
    ds = TSDataset(train_x, np.array(train_y_norm, np.float32))

    #训练 Student
    print("Training student model ...")
    student = train_student(ds)

    #加载 20 条测试集 + LLM 打分
    print("Loading hold-out test series ...")
    test_x = load_holdout()
    print(f"Test-X shape={test_x.shape}")

    print("LLM scoring test set ...")
    test_y = []
    for s in tqdm(test_x, desc="LLM-test"):
        sc = llm_score(s, gt)
        if sc == -1:
            sc = max(0, np.corrcoef(s, gt)[0, 1] * 10)
        test_y.append(sc)

    #Student 预测 → 反归一化
    student.eval()
    with torch.no_grad():
        dev = next(student.parameters()).device
        preds_norm = student(torch.tensor(test_x, dtype=torch.float32).to(dev)).cpu().numpy()
    preds = denormalize_scores(preds_norm, s_min, s_max)       # 回到原尺度

    #评估相关性
    rho = spearman(test_y, preds)
    print(f"Spearman (LLM vs Student) on {len(test_x)} samples: {rho:.3f}")

    #保存结果
    print("Writing outputs ...")
    train_y_norm = [float(x) for x in train_y_norm]
    preds = [float(x) for x in preds]
    train_y = [float(x) for x in train_y]
    test_y = [float(x) for x in test_y]
    np.save(SAVE_DIR / "train_series.npy", train_x.astype(np.float32))
    np.save(SAVE_DIR / "test_series.npy",  test_x.astype(np.float32))
    for fname, data in [
        ("train_scores_raw.json",  train_y),
        ("train_scores_norm.json", train_y_norm),
        ("test_scores_llm.json",   test_y),
        ("test_scores_student.json", preds),
    ]:
        with open(SAVE_DIR / fname, "w") as f:
            json.dump(data, f, indent=2)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_x),
        "test_samples":  len(test_x),
        "failed_llm_queries": failed,
        "score_range": {"min": s_min, "max": s_max},
        "spearman": rho,
    }
    with open(SAVE_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All artifacts saved to", SAVE_DIR.resolve())


if __name__ == "__main__":
    main()
