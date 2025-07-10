from __future__ import annotations

import json
import os
import pathlib
import re
from typing import List, Tuple

import numpy as np
import openai
from tqdm import tqdm
# ============================================================================
# Fixed parameters (edit here if needed) --------------------------------------
# ============================================================================
DISTILL_PATH = "distill_normalized.py"
VAL_SIZE     = 30
REPEATS      = 2
THRESHOLD    = 0.1

GT_PATH = "./Y_real_scaled_ECfluxnet_Combined_0.npy"  # one‑year slice
SAVE_DIR = pathlib.Path("distil_outputs"); SAVE_DIR.mkdir(exist_ok=True)
VAL_PATH  = SAVE_DIR / "val_series.npy"               # cached validation set

openai.api_key = ""

_PAT = re.compile(r"(?:\*\*)?FINAL_SCORE(?:\*\*)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.I)

# ----------------------------------------------------------------------------
# Helper functions (mostly identical to distill_normalized.py) ----------------
# ----------------------------------------------------------------------------

def load_ground_truth() -> np.ndarray:
    data = np.load(GT_PATH)
    return data[365 * 5 : 365 * 6, 0, 1].astype(np.float32)


def build_synthetic(gt: np.ndarray, n: int = 30, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = gt.copy()
        base += rng.normal(0, rng.uniform(0.05, 0.3) * gt.std(), gt.shape)
        base = np.roll(base, rng.integers(-20, 21))
        base *= rng.uniform(0.6, 1.4)
        seasonal = rng.uniform(0.0, 0.2) * gt.std() * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * np.arange(len(gt)) / 365)
        base += seasonal
        if rng.random() < 0.3:
            base += rng.uniform(-0.1, 0.1) * gt.std() * np.linspace(-1, 1, len(gt))
        if rng.random() < 0.2:
            idx = rng.choice(len(base), rng.integers(1, 5), replace=False)
            base[idx] += rng.uniform(2, 4) * gt.std() * rng.choice([-1, 1], len(idx))
        out.append(base)
    return np.stack(out).astype(np.float32)


def _avg10(x: np.ndarray) -> List[float]:
    n = len(x) // 10
    vals = x[: n * 10].reshape(n, 10).mean(1).tolist()
    if len(x) % 10:
        vals.append(float(x[n * 10 :].mean()))
    return vals


def llm_score(ts: np.ndarray, gt: np.ndarray, policy: str, retries: int = 3) -> float:
    prompt = f"""
You are an ecological-time-series scoring bot.
Follow **every** rule below *exactly*.  
Any deviation (extra words, wrong decimals, wrong key, markdown) must be treated as an error.

────────────────────────────────────────
INPUTS
Ground Truth (10-point averages): {_avg10(gt)}
Candidate    (10-point averages): {_avg10(ts)}

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
                {"role": "system", "content": "You are a meticulous expert."},
                {"role": "user", "content": prompt},
            ],
        ).choices[0].message.content
        m = _PAT.search(txt)
        if m:
            return float(m.group(1))
    return -1.0

# ----------------------------------------------------------------------------
# Validation + variability computations --------------------------------------
# ----------------------------------------------------------------------------

def load_validation(gt: np.ndarray, n: int) -> np.ndarray:
    if VAL_PATH.exists():
        return np.load(VAL_PATH)
    val = build_synthetic(gt, n=n)
    np.save(VAL_PATH, val)
    return val


def collect_scores(val_series: np.ndarray, gt: np.ndarray, policy: str, repeats: int) -> List[List[float]]:
    score_nest = []
    for i, ts in enumerate(tqdm(val_series, desc=f"LLM scoring {len(val_series)} val series")):
        scores = []
        for _ in range(repeats):
            scores.append(llm_score(ts, gt, policy))
        score_nest.append(scores)
    return score_nest


def compute_variability(score_nest: List[List[float]]):
    stds = np.array([np.std(s) for s in score_nest], dtype=np.float32)
    return stds, float(stds.mean())

# ----------------------------------------------------------------------------
# Policy extraction / refinement / patching ----------------------------------
# ----------------------------------------------------------------------------

def extract_best_policy(code_text: str) -> str:
    m = re.search(r"BEST_POLICY\s*=\s*r?\"\"\"(.*?)\"\"\"", code_text, re.S)
    if not m:
        raise ValueError("BEST_POLICY block not found in distill script.")
    return m.group(1).strip()


def refine_policy(current_policy: str, score_nest, mean_std):
    summary_json = json.dumps({"avg_sigma": round(mean_std, 4)}, indent=2)
    user_prompt = f"""
CURRENT_POLICY:
{current_policy}

It yields avg σ = {mean_std:.3f}. Please rewrite it (same format) to reduce variance.
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
        raise RuntimeError("Failed to extract policy from LLM response:\n" + response)
    return m.group(1).strip()


def patch_distill(path: pathlib.Path, new_policy: str):
    original = path.read_text()
    patched = re.sub(r"BEST_POLICY\s*=\s*r?\"\"\".*?\"\"\"", f'BEST_POLICY = r"""\n{new_policy}\n"""', original, flags=re.S)
    path.with_suffix(".bak").write_text(original)
    path.write_text(patched)

# ----------------------------------------------------------------------------
# Driver ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

def run():
    gt = load_ground_truth()
    distill_path = pathlib.Path(DISTILL_PATH)
    current_policy = extract_best_policy(distill_path.read_text())

    val_series = load_validation(gt, VAL_SIZE)
    score_nest = collect_scores(val_series, gt, current_policy, REPEATS)
    stds, mean_std = compute_variability(score_nest)
    print(f"Mean σ = {mean_std:.3f} (threshold {THRESHOLD})")
    SAVE_DIR.joinpath("val_score_stds.json").write_text(json.dumps(stds.tolist(), indent=2))

    if mean_std <= THRESHOLD:
        print("Policy stable – no refinement needed.")
        return

    print("✱ Policy unstable – refining via GPT‑4o …")
    new_policy = refine_policy(current_policy, score_nest, mean_std)
    SAVE_DIR.joinpath("new_policy.txt").write_text(new_policy)

    if new_policy == current_policy:
        print("LLM returned identical policy ⇒ abort.")
        return

    patch_distill(distill_path, new_policy)
    print("Patched", distill_path, "(backup saved with .bak)")


if __name__ == "__main__":
    run()
