"""T2-1: EXAONE 크기 사다리 (2.4B vs 7.8B) — 패밀리·한국어비중 고정, 크기만 변화.

교란 없는 비교로 묻는다: 사어 경어(특히 '보게나') 취약성이 모델 크기에 따라 변하는가?
3 민감성 프로파일이 패밀리 차이인지 크기 차이인지 분리하는 단서.

실행: .venv/bin/python analysis/size_ladder.py
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
LADDER = [("exaone-2.4b", "2.4B"), ("exaone3.5", "7.8B")]
LEVELS = {1: "해라체", 2: "해체", 3: "하게체", 4: "하오체", 5: "해요체", 6: "하십시오체"}


def boot_ci(x, reps=10000, seed=42):
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(reps, len(x)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    gold = {json.loads(l)["item_id"]: json.loads(l)["label"]
            for l in (ROOT / "data/samples/copa.jsonl").read_text(encoding="utf-8").splitlines() if l}
    db = sqlite3.connect(ROOT / "data" / "responses.db")

    # 1) 단계별 정확도 (크기 × 단계)
    print("== COPA 정확도: EXAONE 크기 × 경어 단계 ==")
    rows = []
    for key, size in LADDER:
        n_have = db.execute("SELECT COUNT(*) FROM responses WHERE model=? AND task='copa'", (key,)).fetchone()[0]
        if n_have == 0:
            print(f"  {size}: 데이터 없음 — 실행 필요")
            continue
        line = [size]
        for lv in range(1, 7):
            rs = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='copa' AND level=?",
                            (key, lv)).fetchall()
            acc = np.mean([parse_copa(r) == gold[i] for i, r in rs])
            rows.append({"size": size, "level": lv, "어체": LEVELS[lv], "acc": acc})
            line.append(f"{acc*100:.1f}")
        print(f"  {size:5s}: " + " ".join(f"{LEVELS[l]} {v}" for l, v in zip(range(1, 7), line[1:])))
    df = pd.DataFrame(rows)
    if df.empty or df["size"].nunique() < 2:
        print("\n사다리 비교에 두 크기 모두 필요. 2.4B 실행 후 재실행.")
        return
    df.to_csv(OUT / "size_ladder_copa.csv", index=False)

    # 2) 보게나(L3,T3) 취약성이 크기에 따라 변하나
    print("\n== '보게나'(L3,T3) 취약성 vs 크기 (해라체 T1 대비 Δ) ==")
    for key, size in LADDER:
        if db.execute("SELECT COUNT(*) FROM responses WHERE model=? AND task='copa'", (key,)).fetchone()[0] == 0:
            continue
        bg = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='copa' AND level=3 AND template=3", (key,)).fetchall()
        hr = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='copa' AND level=1 AND template=1", (key,)).fetchall()
        a_bg = np.mean([parse_copa(r) == gold[i] for i, r in bg])
        a_hr = np.mean([parse_copa(r) == gold[i] for i, r in hr])
        print(f"  {size:5s}: 보게나 {a_bg*100:.1f}% vs 해라체 {a_hr*100:.1f}% → Δ={ (a_bg-a_hr)*100:+.1f}%p")

    # 3) 그림
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for size in df["size"].unique():
        s = df[df["size"] == size].sort_values("level")
        ax.plot([LEVELS[l] for l in s["level"]], s["acc"] * 100, marker="o", label=f"EXAONE {size}")
    ax.axvspan(1.5, 3.5, color="orange", alpha=0.1)
    ax.set_ylabel("COPA 정확도 (%)")
    ax.set_title("EXAONE 크기 사다리 — 사어체 취약성의 규모 의존성")
    ax.legend(fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "size_ladder.png", dpi=150)
    print(f"\n저장: size_ladder_copa.csv, size_ladder.png")


if __name__ == "__main__":
    main()
