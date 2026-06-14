"""업그레이드 결과 그림: (1) 토크나이저 fertility, (2) 디코딩 강건성 오버레이."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import parse_copa  # noqa: E402

OUT = ROOT / "analysis" / "output"
FIG = ROOT / "analysis" / "figures"
plt.rcParams["font.family"] = ["AppleGothic", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
LEVELS = ["해라체", "해체", "하게체", "하오체", "해요체", "하십시오체"]


def fig_fertility():
    rep = json.loads((OUT / "tokenizer_fertility.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, d in rep["per_model"].items():
        ys = [d["by_level"][str(lv)]["ending_fertility"] for lv in range(1, 7)]
        ax.plot(LEVELS, ys, marker="o", label=label)
    ax.axvspan(1.5, 3.5, color="orange", alpha=0.12)  # 하게체·하오체(사어) 음영
    ax.annotate("사어체\n(하게·하오)", xy=(2.5, ax.get_ylim()[1]), ha="center", va="top", fontsize=8, color="darkorange")
    ax.set_ylabel("종결어미 fertility (토큰/음절)")
    ax.set_xlabel("경어 단계")
    ax.set_title("종결어미 토크나이저 분절도 — 하게체가 하오체보다 더 잘게 깨짐")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "tokenizer_fertility.png", dpi=150)
    print("저장: tokenizer_fertility.png")


def fig_decoding():
    import json as _j
    gold = {_j.loads(l)["item_id"]: _j.loads(l)["label"]
            for l in (ROOT / "data/samples/copa.jsonl").read_text(encoding="utf-8").splitlines() if l}
    db = sqlite3.connect(ROOT / "data" / "responses.db")
    LV = {1: "해라체", 3: "하게체", 5: "해요체"}
    rows = db.execute("SELECT level,item_id,sample_idx,response FROM copa_t07 "
                      "WHERE model='exaone3.5' AND response IS NOT NULL").fetchall()
    ids = {iid for _, iid, _, _ in rows}
    from collections import defaultdict
    acc07 = defaultdict(list)
    for lv, iid, s, r in rows:
        acc07[(lv, s)].append(parse_copa(r) == gold[iid])
    g = db.execute("SELECT level,item_id,response FROM responses "
                   "WHERE model='exaone3.5' AND task='copa' AND template=1").fetchall()
    accG = defaultdict(list)
    for lv, iid, r in g:
        if iid in ids and lv in LV:
            accG[lv].append(parse_copa(r) == gold[iid])
    xs = [LV[lv] for lv in (1, 3, 5)]
    gy = [sum(accG[lv]) / len(accG[lv]) * 100 for lv in (1, 3, 5)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, gy, marker="s", lw=2, label="greedy (temp 0)")
    for s in range(3):
        sy = [sum(acc07[(lv, s)]) / len(acc07[(lv, s)]) * 100 for lv in (1, 3, 5)]
        ax.plot(xs, sy, marker="o", alpha=0.5, lw=1,
                label="temp 0.7 표본" if s == 0 else None, color="gray")
    ax.set_ylabel("COPA 정확도 (%)")
    ax.set_title("하게체 저하의 디코딩 강건성 (EXAONE, 200아이템)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "decoding_robustness.png", dpi=150)
    print("저장: decoding_robustness.png")


if __name__ == "__main__":
    fig_fertility()
    fig_decoding()
