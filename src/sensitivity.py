"""민감도 분석 (사전 등록 부록 항목): temperature 0.7 × 3표집.

서브셋: NSMC 앞 100아이템 × 6단계 × 템플릿 1 × 3표집 = 모델당 1,800 호출.
greedy 본 실험의 단계 효과가 표집 변동성에 견고한지 확인하는 목적.

실행: python -m src.sensitivity --model exaone3.5
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import adapters
from .runner import build_prompt, load_samples, load_templates

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses_t07 (
    model      TEXT NOT NULL,
    task       TEXT NOT NULL,
    level      INTEGER NOT NULL,
    template   INTEGER NOT NULL,
    item_id    TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    response   TEXT,
    error      TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model, task, level, template, item_id, sample_idx)
);
"""

N_ITEMS = 100
SAMPLES = 3
TEMPERATURE = 0.7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(adapters.MODELS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tpl = load_templates()
    items = load_samples("nsmc")[:N_ITEMS]
    max_tokens = tpl["max_tokens"]["nsmc"]

    db = sqlite3.connect(ROOT / "data" / "responses.db")
    db.execute(SCHEMA)
    done = {
        (lv, iid, s)
        for lv, iid, s in db.execute(
            "SELECT level, item_id, sample_idx FROM responses_t07 "
            "WHERE model=? AND task='nsmc' AND response IS NOT NULL", (args.model,))
    }
    jobs = [
        (lv, item, s)
        for lv in range(1, 7)
        for item in items
        for s in range(SAMPLES)
        if (lv, str(item["item_id"]), s) not in done
    ]
    print(f"[sensitivity] {args.model}: 대상 {len(jobs)}건 (캐시 제외)", flush=True)

    def run_one(job):
        lv, item, s = job
        prompt = build_prompt(tpl, "nsmc", lv, 1, item)
        response, error = None, None
        t1 = time.time()
        for attempt in (1, 2):
            try:
                response = adapters.generate(args.model, prompt, max_tokens=max_tokens,
                                             temperature=TEMPERATURE, seed=42 + s)
                error = None
                break
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                if attempt == 1:
                    time.sleep(3)
        return lv, item, s, response, error, int((time.time() - t1) * 1000)

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in jobs]):
            lv, item, s, response, error, latency = fut.result()
            db.execute(
                "INSERT OR REPLACE INTO responses_t07 VALUES (?,?,?,?,?,?,?,?,?,?)",
                (args.model, "nsmc", lv, 1, str(item["item_id"]), s,
                 response, error, latency, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            n += 1
            if n % 100 == 0:
                db.commit()
                print(f"[sensitivity] {n}/{len(jobs)}", flush=True)
    db.commit()
    print(f"[sensitivity] {args.model} 완료: {n}건, {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
