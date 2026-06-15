"""T1-5 보강: 하게체 COPA 붕괴의 패러프레이즈(템플릿) 분해.

붕괴가 가장 잘게 분절되는 명령형 '골라 보게나'(보게나=3토큰)에 집중되는지 확인.
토크나이저 fertility 기전을 템플릿 수준에서 직접 검증한다.

실행: .venv/bin/python analysis/template_breakdown.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import parse_copa  # noqa: E402

OUT = ROOT / "analysis" / "output"
FIG = ROOT / "analysis" / "figures"
MODELS = ["exaone3.5", "qwen3", "llama3.1"]
# 템플릿별 하게체 명령형과 토큰 수(세 토크나이저 공통: 보게나=3, 보게=2)
TPL = {1: ("골라 보게", 2), 2: ("판단해 주게", 2), 3: ("골라 보게나", 3)}


def boot_ci(x, reps=10000, seed=42):
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(reps, len(x)), replace=True).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    gold = {json.loads(l)["item_id"]: json.loads(l)["label"]
            for l in (ROOT / "data/samples/copa.jsonl").read_text(encoding="utf-8").splitlines() if l}
    db = sqlite3.connect(ROOT / "data" / "responses.db")
    rows = []
    for m in MODELS:
        for tn, (form, ntok) in TPL.items():
            for lv, lvn in [(1, "해라체"), (3, "하게체")]:
                rs = db.execute("SELECT item_id,response FROM responses "
                                "WHERE model=? AND task='copa' AND level=? AND template=?",
                                (m, lv, tn)).fetchall()
                corr = np.array([parse_copa(r) == gold[i] for i, r in rs], float)
                lo, hi = boot_ci(corr)
                rows.append({"model": m, "template": tn, "form": form, "n_tokens": ntok,
                             "level": lvn, "acc": corr.mean(), "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows)
    # Δ(하게체-해라체) per model×template
    piv = df.pivot_table(index=["model", "template", "form", "n_tokens"], columns="level", values="acc").reset_index()
    piv["delta"] = piv["하게체"] - piv["해라체"]
    piv.to_csv(OUT / "template_breakdown_copa.csv", index=False)
    print(piv.to_string(index=False))

    # 그림: 모델별 Δ(하게체 효과)를 템플릿(토큰수)별로
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    w = 0.25
    for k, m in enumerate(MODELS):
        sub = piv[piv["model"] == m].sort_values("template")
        ax.bar(x + (k - 1) * w, sub["delta"] * 100, w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TPL[t][0]}\n({TPL[t][1]}토큰)" for t in (1, 2, 3)], fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("하게체 효과 Δ (vs 해라체, %p)")
    ax.set_title("하게체 COPA 붕괴는 가장 잘게 분절되는 '보게나'에 집중")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "template_breakdown.png", dpi=150)
    print(f"\n저장: template_breakdown_copa.csv, template_breakdown.png")


if __name__ == "__main__":
    main()
