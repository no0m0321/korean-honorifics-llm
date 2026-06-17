"""경어 강건성 벤치마크 하네스 (T2-6).

이미 data/responses.db에 응답이 있는 모델의 '강건성 프로파일'을 출력한다.
새 모델은 `python -m src.runner --model <name> --task copa` 등으로 응답을 채운 뒤 실행.
새 모델 추가법: src/adapters.py의 MODELS에 한 줄(+필요 시 API 키 env) 추가.

강건성 스코어 = COPA 단계 간 정확도 범위(%p). 낮을수록 경어에 강건.
참조값(파서 v1.1): EXAONE 8.0 · Llama 3.6 · Qwen3 2.2.

사용: python benchmark.py --model qwen3
      python benchmark.py --leaderboard
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.metrics import is_refusal, parse_copa, parse_nsmc  # noqa: E402
from src.style_classifier import classify  # noqa: E402

LEVELS = {1: "해라체", 2: "해체", 3: "하게체", 4: "하오체", 5: "해요체", 6: "하십시오체"}


def _gold(task):
    return {json.loads(l)["item_id"]: json.loads(l)["label"]
            for l in (ROOT / f"data/samples/{task}.jsonl").read_text(encoding="utf-8").splitlines() if l}


def profile(db, model):
    out = {"model": model}
    # COPA 단계 간 정확도 범위 = 강건성 스코어
    for task, parser in (("copa", parse_copa), ("nsmc", parse_nsmc)):
        gold = _gold(task)
        accs = {}
        pf = {}
        for lv in range(1, 7):
            rs = db.execute("SELECT item_id,response FROM responses WHERE model=? AND task=? AND level=?",
                            (model, task, lv)).fetchall()
            if not rs:
                break
            preds = [(parser(r), gold[i]) for i, r in rs]
            accs[lv] = np.mean([p == g for p, g in preds])
            pf[lv] = np.mean([parser(r) is None for _, r in rs])
        if accs:
            out[f"{task}_acc_range_pp"] = round((max(accs.values()) - min(accs.values())) * 100, 1)
            out[f"{task}_parse_fail_max_pct"] = round(max(pf.values()) * 100, 2)
    # 어체 미러링 강도 (요약): 격식체 출력 비율의 입력 단계 1→6 증가폭
    rs = db.execute("SELECT level,response FROM responses WHERE model=? AND task='summ' AND response IS NOT NULL", (model,)).fetchall()
    if rs:
        from collections import defaultdict
        formal = defaultdict(list)
        for lv, r in rs:
            formal[lv].append(classify(r) == "하십시오체")
        if 1 in formal and 6 in formal:
            out["mirroring_pp"] = round((np.mean(formal[6]) - np.mean(formal[1])) * 100, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--db", default=str(ROOT / "data" / "responses.db"))
    args = ap.parse_args()
    db = sqlite3.connect(args.db)
    models = [r[0] for r in db.execute("SELECT DISTINCT model FROM responses")] if args.leaderboard else [args.model]
    print(f"{'model':14s} {'COPA범위':>8s} {'NSMC범위':>8s} {'COPA파싱실패':>11s} {'미러링':>7s}")
    rows = sorted((profile(db, m) for m in models if m), key=lambda d: d.get("copa_acc_range_pp", 1e9))
    for p in rows:
        print(f"{p['model']:14s} {p.get('copa_acc_range_pp','—'):>7}p {p.get('nsmc_acc_range_pp','—'):>7}p "
              f"{p.get('copa_parse_fail_max_pct','—'):>10}% {p.get('mirroring_pp','—'):>6}p")
    print("\n강건성 스코어 = COPA 단계 간 정확도 범위(%p), 낮을수록 강건.")


if __name__ == "__main__":
    main()
