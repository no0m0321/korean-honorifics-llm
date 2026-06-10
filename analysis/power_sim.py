"""시뮬레이션 기반 검정력 분석 (사전 등록 문서·논문 6장용).

모형: 아이템 난이도 u_i ~ N(logit(p0), sigma^2) 를 공유하는 대응 이항 자료.
조건 B는 로짓에 beta를 더해 주변 정확도 차이가 delta가 되도록 보정.
검정: 대응 McNemar (불일치쌍 정확 이항검정), alpha = 0.05/15 (15쌍 Bonferroni —
Holm의 최악 경우에 해당하는 보수적 기준).

실행: python analysis/power_sim.py
산출: analysis/output/power_sim.csv, analysis/output/power_curve.png(선택)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "output"

P0 = 0.85          # 베이스라인 정확도 (파일럿 실측 후 갱신)
SIGMA = 1.0        # 아이템 난이도 산포 (로짓 스케일)
ALPHA = 0.05 / 15  # 6단계 -> 15 쌍별 비교
REPS = 2000
NS = (200, 1000)
DELTAS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
TEMPLATES = 3          # 통합 분석: 아이템 x 템플릿 관측
TAU = 0.5              # 템플릿 효과 산포 (로짓)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_beta(delta: float, rng: np.random.Generator) -> float:
    """주변 정확도 차이가 delta가 되는 로짓 이동량 beta를 이분탐색으로 보정."""
    u = rng.normal(np.log(P0 / (1 - P0)), SIGMA, size=200_000)
    base = _sigmoid(u).mean()
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(u + mid).mean() - base < delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """대응 이진 결과의 McNemar 정확검정 p값."""
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    return stats.binomtest(min(n01, n10), n, 0.5).pvalue * 1  # two-sided

def power(n: int, delta: float, rng: np.random.Generator) -> float:
    beta = calibrate_beta(delta, rng)
    mu = np.log(P0 / (1 - P0))
    hits = 0
    for _ in range(REPS):
        u = rng.normal(mu, SIGMA, size=n)
        a = rng.random(n) < _sigmoid(u)          # 조건 A (낮은 단계)
        b = rng.random(n) < _sigmoid(u + beta)   # 조건 B (delta만큼 높음)
        if mcnemar_p(a, b) < ALPHA:
            hits += 1
    return hits / REPS


def power_pooled(n_items: int, delta: float, rng: np.random.Generator) -> float:
    """아이템 x 템플릿(3) 통합 관측에 대한 McNemar 검정력.

    주의: 같은 아이템의 템플릿 관측은 u_i를 공유해 독립이 아니므로 이 추정치는
    낙관적(anticonservative) 상한이다. 본 분석의 주 검정은 아이템 무선효과를
    명시적으로 다루는 혼합효과모형이며, 이 수치는 그 검정력의 근사로 해석한다.
    """
    beta = calibrate_beta(delta, rng)
    mu = np.log(P0 / (1 - P0))
    hits = 0
    for _ in range(REPS):
        u = np.repeat(rng.normal(mu, SIGMA, size=n_items), TEMPLATES)
        tau = rng.normal(0, TAU, size=n_items * TEMPLATES)  # 조건 간 공유되는 템플릿 효과
        a = rng.random(u.size) < _sigmoid(u + tau)
        b = rng.random(u.size) < _sigmoid(u + tau + beta)
        if mcnemar_p(a, b) < ALPHA:
            hits += 1
    return hits / REPS


def main() -> None:
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"p0={P0}, sigma={SIGMA}, alpha={ALPHA:.4f} (Bonferroni 15쌍), reps={REPS}")
    print(f"{'n':>6} {'delta':>7} {'power':>7}")
    for n in NS:
        for d in DELTAS:
            pw = power(n, d, rng)
            rows.append({"n": n, "delta": d, "power": round(pw, 3)})
            print(f"{n:>6} {d:>7.2f} {pw:>7.3f}")
    print(f"-- 통합 분석 (아이템 1000 x 템플릿 {TEMPLATES} = 3000 관측, 낙관적 상한) --")
    for d in DELTAS:
        pw = power_pooled(1000, d, rng)
        rows.append({"n": f"1000x{TEMPLATES}", "delta": d, "power": round(pw, 3)})
        print(f"{'1000x3':>6} {d:>7.2f} {pw:>7.3f}")
    with open(OUT / "power_sim.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "delta", "power"])
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        series = [(200, "n = 200 (v2.0)"), (1000, "n = 1000 (v2.1)"),
                  (f"1000x{TEMPLATES}", "n = 1000 x 3 templates (pooled)")]
        for key, label in series:
            xs = [r["delta"] for r in rows if r["n"] == key]
            ys = [r["power"] for r in rows if r["n"] == key]
            ax.plot(xs, ys, marker="o", label=label)
        ax.axhline(0.8, ls="--", c="gray", lw=1)
        ax.set_xlabel("Accuracy difference between levels (delta)")
        ax.set_ylabel("Power (McNemar, Bonferroni-corrected)")
        ax.set_title(f"Simulated power (p0={P0}, {REPS} reps)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "power_curve.png", dpi=150)
        print(f"그림 저장: {OUT / 'power_curve.png'}")
    except ImportError:
        print("matplotlib 미설치 — 그림 생략")


if __name__ == "__main__":
    main()
