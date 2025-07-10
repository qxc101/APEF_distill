
"""
--------------------------------------------------
目标：
1. 生成 100 条合成 Time Series (365*1)
2. 使用同一个 base_policy 连续打分两次
3. 计算两次分数的差异：
   - 平均绝对差 (mean_abs_diff)
   - Spearman ρ
4. 若 mean_abs_diff < 0.01 ⇒ 判定为“稳定”
   否则提示需要细化 / 修正 policy prompt
结果同时保存到 distil_outputs/verification_*.json
--------------------------------------------------
"""
import os, json, time, pathlib, random, re
from typing import List

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import openai

# ----------- 配置 -------------
N_SERIES          = 100          # 合成序列条数
REPEATS           = 2            # 打分轮数
DIFF_THRESHOLD    = 0.01         # 稳定性阈值
RNG_SEED          = 42           # 复现实验
SAVE_DIR          = pathlib.Path("distil_outputs")
GT_PATH           = "./Y_real_scaled_ECfluxnet_Combined_0.npy"
# ------------ OpenAI ----------
openai.api_key = ""
_PAT = re.compile(r"(?:\*\*)?FINAL_SCORE(?:\*\*)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.I)

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

TOTAL SCORE = Sum of all 5 component scores (Range: 0-10)
"""

# ---------- 数据相关 ----------
def load_ground_truth() -> np.ndarray:
    data = np.load(GT_PATH)                    # (≈5 年, 365,  ?, ?)
    return data[365*5 : 365*6, 0, 1].astype(np.float32)

def build_synthetic(gt: np.ndarray, n:int=N_SERIES, seed:int=RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = gt.copy()
        # 多样化扰动
        base += rng.normal(0, rng.uniform(0.05,0.3)*gt.std(), gt.shape)           # 噪声
        base = np.roll(base, rng.integers(-20,21))                                # 时间平移
        base *= rng.uniform(0.6,1.4)                                              # 幅值缩放
        seasonal = rng.uniform(0,0.2)*gt.std()*np.sin(2*np.pi*rng.uniform(0.5,2)*np.arange(len(gt))/365)
        base += seasonal                                                          # 季节项
        if rng.random() < 0.3:                                                    # 线性趋势
            base += rng.uniform(-0.1,0.1)*gt.std()*np.linspace(-1,1,len(gt))
        if rng.random() < 0.2:                                                    # 离群点
            idx = rng.choice(len(base), rng.integers(1,5), replace=False)
            base[idx] += rng.choice([-1,1],len(idx))*rng.uniform(2,4)*gt.std()
        out.append(base)
    return np.stack(out).astype(np.float32)

def _avg10(x: np.ndarray) -> List[float]:
    n = len(x)//10
    vals = x[:n*10].reshape(n,10).mean(1).tolist()
    if len(x)%10: vals.append(float(x[n*10:].mean()))
    return vals

# ---------- LLM 打分 ----------
def llm_score(ts: np.ndarray, gt: np.ndarray, retries:int=3) -> float:
    ts_s, gt_s = _avg10(ts), _avg10(gt)
    prompt = f"""
You are an ecological-time-series scoring bot.
Follow **every** rule below *exactly*.  
Any deviation (extra words, wrong decimals, wrong key, markdown) must be treated as an error.

────────────────────────────────────────
INPUTS
Ground Truth (10-point averages): {gt_s}
Candidate    (10-point averages): {ts_s}

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
    for _ in range(retries):
        txt = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[{"role":"system","content":"You are a meticulous expert."},
                      {"role":"user","content":prompt}]
        ).choices[0].message.content
        m = _PAT.search(txt)
        if m:
            return float(m.group(1))
    return -1.0   # 标记失败

def score_batch(series: np.ndarray, gt: np.ndarray) -> List[float]:
    scores, fails = [], 0
    for s in tqdm(series, desc="LLM scoring"):
        sc = llm_score(s, gt)
        if sc < 0:
            fails += 1
            sc = max(0, np.corrcoef(s, gt)[0,1]*10)  # 简单兜底
        scores.append(sc)
    if fails:
        print(f"[WARN] {fails} queries fell back to correlation-based score")
    return scores

# ---------- 稳定性评估 ----------
def evaluate_stability(scores_matrix: np.ndarray, threshold:float=DIFF_THRESHOLD):
    """scores_matrix shape:(repeats,n_series)"""
    diff = np.abs(scores_matrix[0] - scores_matrix[1])
    mean_abs_diff = float(diff.mean())
    spearman_r   = float(spearmanr(scores_matrix[0], scores_matrix[1])[0])
    stable = mean_abs_diff < threshold
    return stable, mean_abs_diff, spearman_r

# ---------- 主流程 ----------
def main():
    SAVE_DIR.mkdir(exist_ok=True)
    ts_start = time.strftime("%Y-%m-%d %H:%M:%S")
    gt   = load_ground_truth()
    synth= build_synthetic(gt, N_SERIES)
    print(f"[INFO] Built {N_SERIES} synthetic series")

    # 两轮打分
    all_scores = []
    for rep in range(REPEATS):
        print(f"\n=== Scoring round {rep+1}/{REPEATS} ===")
        random.seed(RNG_SEED+rep)   # 保证请求顺序一致
        all_scores.append(score_batch(synth, gt))
    scores_np = np.array(all_scores)  # shape (REPEATS,N_SERIES)

    stable, mean_diff, rho = evaluate_stability(scores_np)
    verdict = "稳定" if stable else "不稳定，需要改进 policy"

    print(f"\n--- Stability report ---")
    print(f"平均绝对差 (|Δ|) : {mean_diff:.4f}")
    print(f"Spearman ρ       : {rho:.3f}")
    print(f"阈值             : {DIFF_THRESHOLD}")
    print(f"结论             : {verdict}")

    # 保存 JSON 记录
    out_json = {
        "timestamp": ts_start,
        "n_series": N_SERIES,
        "repeats": REPEATS,
        "diff_threshold": DIFF_THRESHOLD,
        "mean_abs_diff": mean_diff,
        "spearman_r": rho,
        "stable": stable,
        "scores_round1": scores_np[0].tolist(),
        "scores_round2": scores_np[1].tolist(),
        "policy": BEST_POLICY.strip()
    }
    fname = SAVE_DIR / f"verification_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[INFO] Verification record written to {fname.resolve()}")

if __name__ == "__main__":
    main()
