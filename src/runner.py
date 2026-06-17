"""배치 실행기: SQLite 캐시 + 중단 재개 + 진행률/ETA 로그.

사용 예:
  python -m src.runner --model exaone3.5 --task nsmc                       # 전체 (6단계 x 3템플릿)
  python -m src.runner --model exaone3.5 --task nsmc --levels 1,6 --templates 1 --limit 6   # 파일럿

응답은 (model, task, level, template, item_id) 키로 캐싱되어
같은 명령을 다시 실행하면 미완료분만 이어서 돈다.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import adapters

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    model      TEXT NOT NULL,
    task       TEXT NOT NULL,
    level      INTEGER NOT NULL,
    template   INTEGER NOT NULL,
    item_id    TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    response   TEXT,
    error      TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model, task, level, template, item_id)
);
"""


def load_templates() -> dict:
    return yaml.safe_load((ROOT / "prompts" / "templates.yaml").read_text(encoding="utf-8"))


def load_samples(task: str) -> list[dict]:
    path = ROOT / "data" / "samples" / f"{task}.jsonl"
    if not path.exists():
        raise SystemExit(f"샘플 파일이 없습니다: {path} — 먼저 `python -m src.data_prep`을 실행하세요.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(tpl: dict, task: str, level: int, template_no: int, item: dict) -> str:
    instruction = tpl["instructions"][task][level][template_no - 1]
    block = tpl["input_blocks"][task].format(**item)
    return f"{instruction}\n\n{block}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(adapters.MODELS))
    ap.add_argument("--task", required=True, choices=["nsmc", "copa", "summ", "hsc", "memo"])
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    ap.add_argument("--templates", default="1,2,3")
    ap.add_argument("--limit", type=int, default=None, help="아이템 수 제한 (파일럿용)")
    ap.add_argument("--workers", type=int, default=1,
                    help="동시 요청 수 — Ollama 서버 배칭 활용 (결정성 검증 후 사용)")
    ap.add_argument("--db", default=str(ROOT / "data" / "responses.db"))
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    templates = [int(x) for x in args.templates.split(",")]
    tpl = load_templates()
    items = load_samples(args.task)
    if args.limit:
        items = items[: args.limit]
    max_tokens = tpl["max_tokens"][args.task]

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.execute(SCHEMA)
    done = {
        (lv, tn, iid)
        for lv, tn, iid in db.execute(
            "SELECT level, template, item_id FROM responses "
            "WHERE model=? AND task=? AND response IS NOT NULL",
            (args.model, args.task),
        )
    }

    jobs = [
        (lv, tn, item)
        for lv in levels
        for tn in templates
        for item in items
        if (lv, tn, str(item["item_id"])) not in done
    ]
    total = len(levels) * len(templates) * len(items)
    print(f"[runner] {args.model}/{args.task}: 전체 {total}건 중 캐시 {total - len(jobs)}건, 실행 대상 {len(jobs)}건")

    def run_one(job):
        lv, tn, item = job
        prompt = build_prompt(tpl, args.task, lv, tn, item)
        response, error = None, None
        t1 = time.time()
        for attempt in (1, 2):  # 일시 오류 1회 재시도
            try:
                response = adapters.generate(args.model, prompt, max_tokens=max_tokens)
                error = None
                break
            except Exception as e:  # noqa: BLE001 — 오류도 데이터로 기록
                error = f"{type(e).__name__}: {e}"
                if attempt == 1:
                    time.sleep(3)
        return lv, tn, item, prompt, response, error, int((time.time() - t1) * 1000)

    t0 = time.time()
    n_done = 0
    n_err = 0
    # 생성은 워커 스레드, DB 기록은 메인 스레드에서만 수행
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in jobs]):
            lv, tn, item, prompt, response, error, latency = fut.result()
            db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?)",
                (args.model, args.task, lv, tn, str(item["item_id"]),
                 prompt, response, error, latency,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            n_done += 1
            n_err += error is not None
            if n_done % 20 == 0:
                db.commit()
            if n_done % 25 == 0 or n_done == len(jobs):
                rate = n_done / (time.time() - t0)
                eta_h = (len(jobs) - n_done) / rate / 3600 if rate else float("inf")
                print(f"[runner] {n_done}/{len(jobs)} ({rate:.2f}건/초, 오류 {n_err}, 남은 시간 ~{eta_h:.1f}h)", flush=True)
    db.commit()
    db.close()
    print(f"[runner] 완료: {n_done}건 (오류 {n_err}건), {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
