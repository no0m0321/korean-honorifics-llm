"""2축 잔여 실험 분석: HAJ(판정↔생성 해리, H7) + NSMC채널2(직교 H5 + 전파).

실행: .venv/bin/python analysis/honor_residual.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.honor_axes import detect_object_hon, detect_subject_hon  # noqa: E402
from src.metrics import parse_nsmc  # noqa: E402

OUT = ROOT / "analysis" / "output"
MODELS = ["exaone3.5", "qwen3", "llama3.1"]


def _first(s):
    return (s or "").strip()


def haj_analysis(db):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / "data/samples/haj.jsonl").read_text(encoding="utf-8").splitlines() if l}
    print("== HAJ 경어 판정 (H7: 판정 vs 생성 해리) ==")
    out = {}
    for m in MODELS:
        rows = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='haj' AND response IS NOT NULL", (m,)).fetchall()
        if not rows:
            continue
        judge_ok = {"none": [], "subj": [], "obj": []}
        for iid, resp in rows:
            it = items[iid]; r = _first(resp)
            # 판정 추출: 응답 첫 부분의 O/X
            mtxt = re.search(r"\b([OXox])\b", r[:10])
            pred = mtxt.group(1).upper() if mtxt else ("O" if r.startswith("O") else ("X" if r.startswith("X") else "?"))
            judge_ok[it["vio"]].append(int(pred == it["label"]))
        acc = lambda k: np.mean(judge_ok[k]) if judge_ok[k] else float("nan")
        out[m] = {"judge_overall": np.mean([v for vs in judge_ok.values() for v in vs]),
                  "judge_subj_vio": acc("subj"), "judge_obj_vio": acc("obj"), "judge_ok_sent": acc("none")}
        print(f"  {m}: 전체판정 {out[m]['judge_overall']*100:.1f}% | 주체위반탐지 {out[m]['judge_subj_vio']*100:.1f}% | 객체위반탐지 {out[m]['judge_obj_vio']*100:.1f}% | 적절문장 {out[m]['judge_ok_sent']*100:.1f}%")
    (OUT / "haj.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def nsmcref_analysis(db):
    gold = {json.loads(l)["item_id"]: json.loads(l)["label"] for l in (ROOT / "data/samples/nsmcref.jsonl").read_text(encoding="utf-8").splitlines() if l}
    print("\n== NSMC 채널2 (H5 직교: 정확도 / 전파: 출력 경어) ==")
    out = {}
    for m in MODELS:
        if db.execute("SELECT COUNT(*) FROM responses WHERE model=? AND task='nsmcref'", (m,)).fetchone()[0] == 0:
            continue
        rec = {}
        for tn, ref in [(1, "평어"), (2, "존대")]:
            accs, props = [], []
            for lv in (1, 5, 6):
                rows = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task='nsmcref' AND level=? AND template=?", (m, lv, tn)).fetchall()
                if not rows:
                    continue
                a = np.mean([parse_nsmc(r) == gold[i] for i, r in rows])
                # 전파: 의뢰자(수신자)는 객체높임 대상 → '전해 드리다/말씀'
                p = np.mean([int("드리" in _first(r) or "말씀" in _first(r)) for _, r in rows])
                accs.append(a); props.append(p)
            rec[ref] = {"acc": float(np.mean(accs)), "obj_hon_rate": float(np.mean(props))}
        out[m] = rec
        a_p, a_h = rec["평어"]["acc"], rec["존대"]["acc"]
        pr_p, pr_h = rec["평어"]["obj_hon_rate"], rec["존대"]["obj_hon_rate"]
        print(f"  {m}: 정확도 평어 {a_p*100:.1f}% / 존대 {a_h*100:.1f}% (Δ {(a_h-a_p)*100:+.1f}%p, H5 직교)")
        print(f"        객체높임 전파(전해 드리다) 평어 {pr_p*100:.1f}% / 존대 {pr_h*100:.1f}% (Δ {(pr_h-pr_p)*100:+.1f}%p)")
    (OUT / "nsmcref.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    db = sqlite3.connect(ROOT / "data" / "responses.db")
    haj_analysis(db)
    nsmcref_analysis(db)
