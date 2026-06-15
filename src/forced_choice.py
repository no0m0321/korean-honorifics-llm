"""T2-5: forced-choice logprob 채점 — 생성·파싱을 완전히 우회한 순수 추론 신호.

COPA에서 선택지 '1'/'2'(및 '정답: 1/2' 시퀀스)의 누적 로그우도를 직접 비교해
argmax로 채점한다. 생성 텍스트·파서를 거치지 않으므로 형식 붕괴와 추론 붕괴를 분리한다.
하게체에서 생성기반 −8%p가 forced-choice에서도 유지되면 '진짜 추론 손상',
사라지면 '형식·파싱 붕괴'로 확정된다.

HF transformers 백엔드(8bit/bf16, MPS). 모델별로 순차 실행.

사용: .venv/bin/python -m src.forced_choice --model exaone3.5 [--limit 1000] [--levels 1,2,3,4,5,6]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

HF_REPO = {
    "exaone3.5": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "llama3.1": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS copa_forced (
    model TEXT, level INT, template INT, item_id TEXT,
    lp1 REAL, lp2 REAL, pred TEXT, gold TEXT, correct INT,
    created_at TEXT,
    PRIMARY KEY (model, level, template, item_id)
);
"""


def load_model(name):
    repo = HF_REPO[name]
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    return tok, model, dev


@torch.no_grad()
def seq_logprob(tok, model, dev, prompt_ids, cont_text):
    """prompt에 이어지는 cont_text의 누적 로그우도."""
    cont_ids = tok.encode(cont_text, add_special_tokens=False)
    ids = torch.tensor([prompt_ids + cont_ids], device=dev)
    logits = model(ids).logits[0]  # [T, V]
    lp = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    start = len(prompt_ids)
    for k, tokid in enumerate(cont_ids):
        total += lp[start + k - 1, tokid].item()
    return total / max(len(cont_ids), 1)  # 길이 정규화


def build_prompt_ids(tok, instruction, block):
    msg = [{"role": "user", "content": f"{instruction}\n\n{block}"}]
    return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF_REPO))
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    ap.add_argument("--template", type=int, default=1)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    tpl = yaml.safe_load((ROOT / "prompts" / "templates.yaml").read_text(encoding="utf-8"))
    items = [json.loads(l) for l in (ROOT / "data/samples/copa.jsonl").read_text(encoding="utf-8").splitlines() if l][: args.limit]

    db = sqlite3.connect(ROOT / "data" / "responses.db")
    db.execute(SCHEMA)
    done = {(lv, iid) for lv, iid in db.execute(
        "SELECT level, item_id FROM copa_forced WHERE model=? AND template=?",
        (args.model, args.template))}

    print(f"[forced] {args.model} 로딩...", flush=True)
    tok, model, dev = load_model(args.model)
    print(f"[forced] device={dev}, 대상 {len(levels)*len(items)-len(done)}건", flush=True)

    t0 = time.time(); n = 0
    for lv in levels:
        instr = tpl["instructions"]["copa"][lv][args.template - 1]
        for it in items:
            if (lv, str(it["item_id"])) in done:
                continue
            block = tpl["input_blocks"]["copa"].format(**it)
            pids = build_prompt_ids(tok, instr, block)
            lp1 = seq_logprob(tok, model, dev, pids, "1")
            lp2 = seq_logprob(tok, model, dev, pids, "2")
            pred = "1" if lp1 >= lp2 else "2"
            gold = str(it["label"])
            db.execute("INSERT OR REPLACE INTO copa_forced VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (args.model, lv, args.template, str(it["item_id"]),
                        lp1, lp2, pred, gold, int(pred == gold),
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
            n += 1
            if n % 100 == 0:
                db.commit()
                r = n / (time.time() - t0)
                print(f"[forced] {n}건 ({r:.1f}/s)", flush=True)
    db.commit()
    print(f"[forced] {args.model} 완료 {n}건 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
