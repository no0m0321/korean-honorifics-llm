"""T1-5: 토크나이저 fertility 분석 — 왜 하게체만(하오체는 아니고) 무너지나.

각 모델 토크나이저로 (a) 단계별 지시문, (b) 단계 대표 종결어미를 인코딩해
형태소/음절당 subword 분절 수(fertility)를 측정한다. 헤드라인 미시 기제:
사어체 종결어미가 더 잘게 깨지는가? 하게체 -게/-소가 특이적으로 깨지는가?

게이트된 토크나이저는 공개 미러로 대체하고 출처를 기록한다.

실행: .venv/bin/python analysis/tokenizer_fertility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis" / "output"

# (모델 라벨, HF 토크나이저 repo, 출처 메모). 게이트된 것은 공개 미러 사용.
TOKENIZERS = [
    ("EXAONE 3.5", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "공식"),
    ("Qwen3", "Qwen/Qwen3-8B", "공식(공개)"),
    ("Llama 3.1", "NousResearch/Meta-Llama-3.1-8B-Instruct", "공개 미러(게이트 우회, 동일 BPE)"),
]

# 단계별 대표 종결어미(명령/평서) — style_classifier 규칙과 정합
LEVEL_ENDINGS = {
    1: ["분류해라", "답해라", "골라라", "써라", "분석해라"],
    2: ["분류해", "답해", "골라", "써", "분석해 줘"],
    3: ["분류해 보게", "답하게", "골라 보게나", "쓰게", "판단해 주게"],
    4: ["분류하시오", "답하시오", "고르시오", "쓰시오", "판단하시오"],
    5: ["분류해 주세요", "답해 주세요", "골라 주세요", "써 주세요", "분석해 주세요"],
    6: ["분류해 주십시오", "답해 주십시오", "골라 주십시오", "작성해 주십시오", "분석해 주십시오"],
}
LEVEL_NAMES = {1: "해라체", 2: "해체", 3: "하게체", 4: "하오체", 5: "해요체", 6: "하십시오체"}


def instructions_by_level():
    tpl = yaml.safe_load((ROOT / "prompts" / "templates.yaml").read_text(encoding="utf-8"))
    out = {lv: [] for lv in range(1, 7)}
    for task in ("nsmc", "copa", "summ"):
        for lv in range(1, 7):
            out[lv].extend(tpl["instructions"][task][lv])
    return out


def syllables(text):
    return sum(1 for ch in text if "가" <= ch <= "힣")


def main():
    from transformers import AutoTokenizer

    instr = instructions_by_level()
    report = {"per_model": {}, "notes": []}
    print(f"{'model':12s} {'level':12s} {'instr_fert':>10s} {'ending_fert':>11s} {'게/소_tok':>9s}")
    for label, repo, src in TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            report["notes"].append(f"{label}: 토크나이저 로드 실패 ({type(e).__name__}) — repo={repo}")
            print(f"{label:12s} 로드 실패: {type(e).__name__}")
            continue
        report["per_model"][label] = {"repo": repo, "source": src, "by_level": {}}
        for lv in range(1, 7):
            # 지시문 fertility: 음절당 subword 수
            i_tok = sum(len(tok.encode(s, add_special_tokens=False)) for s in instr[lv])
            i_syl = sum(syllables(s) for s in instr[lv])
            # 종결어미 fertility
            e_tok = sum(len(tok.encode(s, add_special_tokens=False)) for s in LEVEL_ENDINGS[lv])
            e_syl = sum(syllables(s) for s in LEVEL_ENDINGS[lv])
            report["per_model"][label]["by_level"][lv] = {
                "name": LEVEL_NAMES[lv],
                "instr_fertility": round(i_tok / i_syl, 3),
                "ending_fertility": round(e_tok / e_syl, 3),
            }
            print(f"{label:12s} {LEVEL_NAMES[lv]:12s} {i_tok/i_syl:10.3f} {e_tok/e_syl:11.3f}")
        # 하게체 핵심 형태소가 단일 토큰인지: '게', '소', '게나' 분절 검사
        probe = {}
        for m in ["보게", "보게나", "하게", "있소", "하시오", "습니다"]:
            ids = tok.encode(m, add_special_tokens=False)
            probe[m] = len(ids)
        report["per_model"][label]["morpheme_probe"] = probe
        print(f"  {label} 형태소 분절: " + ", ".join(f"{k}={v}" for k, v in probe.items()))
    (OUT / "tokenizer_fertility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT / 'tokenizer_fertility.json'}")


if __name__ == "__main__":
    main()
