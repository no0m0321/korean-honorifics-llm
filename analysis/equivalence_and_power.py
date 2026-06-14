"""T1-2 + T1-3: 고빈도 단계쌍 TOST 동등성검정 + 검정력/사전등록 모순 정리.

TOST: 현대 고빈도 단계(해체·해요체·하십시오체) 간 정확도 '차이 없음'을 적극 입증.
대응 이진자료이므로 아이템별 차이의 부트스트랩 90% CI가 ±Δ(=2%p) 안에 들면 동등.
(TOST at alpha=0.05 ⇔ 90% CI 포함 검사)

Power-fix: power_sim의 P0=0.85 플레이스홀더를 태스크·모델별 실측 P0로 재계산.

실행: .venv/bin/python analysis/equivalence_and_power.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import load_results  # noqa: E402

OUT = ROOT / "analysis" / "output"
MODELS = ["exaone3.5", "qwen3", "llama3.1"]
LEVEL_NAMES = {1: "해라체", 2: "해체", 3: "하게체", 4: "하오체", 5: "해요체", 6: "하십시오체"}
HIGH_FREQ = [2, 5, 6]  # 해체·해요체·하십시오체 (현대 고빈도)
DELTA = 0.02           # 실질 동등성 한계 ±2%p (사전 고정)


def paired_matrix(g):
    """행=item_id, 열=level, 값=정확도(템플릿 평균)."""
    return g.pivot_table(index="item_id", columns="level", values="correct", aggfunc="mean")


def tost_pair(a, b, delta=DELTA, reps=10000, seed=42):
    """대응 차이의 90% 부트스트랩 CI가 [-delta, delta]에 포함되면 동등."""
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(reps, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [5, 95])
    return {"mean_diff": float(d.mean()), "ci90": [float(lo), float(hi)],
            "equivalent": bool(lo > -delta and hi < delta)}


def run_tost():
    out = {}
    print(f"== TOST 동등성검정 (고빈도쌍, Δ=±{DELTA*100:.0f}%p, 90% 부트스트랩 CI) ==")
    for task in ("nsmc", "copa"):
        df = load_results(task)
        out[task] = {}
        for m in MODELS:
            mat = paired_matrix(df[df["model"] == m])
            out[task][m] = {}
            print(f"\n{task.upper()} / {m}:")
            for i, j in combinations(HIGH_FREQ, 2):
                r = tost_pair(mat[i], mat[j])
                pair = f"{LEVEL_NAMES[i]}-{LEVEL_NAMES[j]}"
                out[task][m][pair] = r
                flag = "✓ 동등" if r["equivalent"] else "✗ 불확정"
                print(f"  {pair}: Δ={r['mean_diff']*100:+.1f}%p "
                      f"CI90[{r['ci90'][0]*100:+.1f},{r['ci90'][1]*100:+.1f}] {flag}")
    (OUT / "equivalence_tost.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


def run_power_fix():
    """실측 P0로 검정력 맥락 재서술용 베이스라인 표."""
    print("\n== 실측 베이스라인 정확도 P0 (검정력 시뮬 P0=0.85 플레이스홀더 대체용) ==")
    rows = []
    for task in ("nsmc", "copa"):
        df = load_results(task)
        for m in MODELS:
            g = df[df["model"] == m]
            p0 = g["correct"].mean()
            rows.append({"task": task, "model": m, "P0": round(float(p0), 3)})
            print(f"  {task.upper():5s} {m:12s} P0={p0:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "baseline_p0.csv", index=False)
    print("주: power_sim의 단일 P0=0.85 대신 위 셀별 P0로 검정력 해석. "
          "통합분석 검정력은 코드 주석대로 '낙관적 상한'으로 본문 명시.")


if __name__ == "__main__":
    run_tost()
    run_power_fix()
    print(f"\n저장: equivalence_tost.json, baseline_p0.csv")
