"""2축(주체·객체높임) 분석: 전파율·과잉존대·굴절vs보충법 해리 + 미러링.

H6(보충법 vs 굴절), H8(모델 비대칭), H9(과잉존대) 검정 + 그림.
채점은 honor_axes(셀 조건 일치). 상대높임=해요체 1종 고정(측정 직교).

실행: .venv/bin/python analysis/honor_2axis.py
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
from src.honor_axes import detect_object_hon, detect_subject_hon  # noqa: E402

OUT = ROOT / "analysis" / "output"
FIG = ROOT / "analysis" / "figures"
MODELS = ["exaone3.5", "qwen3", "llama3.1"]


def _first_line(s):
    return (s or "").strip().split("\n")[0]


def hsc_scores(db, model, items):
    rows = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='hsc' AND response IS NOT NULL", (model,)).fetchall()
    sp, sm, op, om, oc = [], [], [], [], []
    for iid, resp in rows:
        it = items[iid]; r = _first_line(resp)
        s, o = detect_subject_hon(r), detect_object_hon(r)
        if it["subj"] == 1:
            sp.append(s)
        elif it["subj"] == 0:
            sm.append(s)
        if it.get("obj") == 1 and not it.get("obj_ctrl"):
            op.append(o)
        elif it.get("obj") == 0:
            om.append(o)
        if it.get("obj_ctrl"):
            oc.append(o)
    rate = lambda x: float(np.mean(x)) if x else float("nan")
    return {
        "subj_prod_plus": rate(sp), "subj_overgen": rate(sm), "subj_prop": rate(sp) - rate(sm),
        "obj_prod_plus": rate(op), "obj_overgen": rate(om), "obj_prop": rate(op) - rate(om),
        "obj_ctrl_suppletive": rate(oc),
        "n_subj": len(sp), "n_obj": len(op),
    }


def memo_scores(db, model, memo):
    rows = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='memo' AND response IS NOT NULL", (model,)).fetchall()
    plus, minus = [], []
    for iid, resp in rows:
        r = _first_line(resp); s = detect_subject_hon(r)
        (plus if memo[iid]["subj"] == 1 else minus).append(s)
    rate = lambda x: float(np.mean(x)) if x else float("nan")
    return {"memo_si_plus": rate(plus), "memo_si_minus": rate(minus),
            "memo_mirror": rate(plus) - rate(minus)}


def main():
    items = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / "data/samples/hsc.jsonl").read_text(encoding="utf-8").splitlines() if l}
    memo = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / "data/samples/memo.jsonl").read_text(encoding="utf-8").splitlines() if l}
    db = sqlite3.connect(ROOT / "data" / "responses.db")
    rows = []
    for m in MODELS:
        if db.execute("SELECT COUNT(*) FROM responses WHERE model=? AND task='hsc'", (m,)).fetchone()[0] == 0:
            print(f"{m}: HSC 데이터 없음")
            continue
        r = {"model": m, **hsc_scores(db, m, items), **memo_scores(db, m, memo)}
        rows.append(r)
        print(f"\n== {m} ==")
        print(f"  주체높임 -시-:  생성 {r['subj_prod_plus']*100:.1f}% / 과잉 {r['subj_overgen']*100:.1f}% / 전파Δ {r['subj_prop']*100:+.1f}%p")
        print(f"  객체높임 보충법: 생성 {r['obj_prod_plus']*100:.1f}% / 과잉 {r['obj_overgen']*100:.1f}% / 전파Δ {r['obj_prop']*100:+.1f}%p")
        print(f"  → 굴절(−시−) vs 보충법 격차: {(r['subj_prop']-r['obj_prop'])*100:+.1f}%p (H6)")
        print(f"  memo 입력-시- 미러링Δ: {r['memo_mirror']*100:+.1f}%p")
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "honor_2axis.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df)); w = 0.26
    ax.bar(x - w, df["subj_prop"] * 100, w, label="주체높임 -시- 전파")
    ax.bar(x, df["obj_prop"] * 100, w, label="객체높임 보충법 전파")
    ax.bar(x + w, df["memo_mirror"] * 100, w, label="memo -시- 미러링")
    ax.set_xticks(x); ax.set_xticklabels(df["model"])
    ax.set_ylabel("전파 Δ (%p)")
    ax.set_title("2축 경어 전파: 굴절(-시-) ≫ 보충법, 모델별 비대칭")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "honor_2axis.png", dpi=150)
    print(f"\n저장: honor_2axis.csv, honor_2axis.png")


if __name__ == "__main__":
    main()
