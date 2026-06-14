"""T1-6: 거부·회피 경로 분리 — 하게체의 이중 해리.

COPA L3: parse_fail≈0이지만 정확도 붕괴 → 추론 손상.
NSMC L3: parse_fail 최고지만 acc_parsed_only는 인접 단계와 동일 → 출력 형식 회피.
단일 어체가 태스크별로 다른 실패 경로를 활성화함을 데이터로 증명한다.

NSMC 파싱 실패 응답을 유형 코딩(규칙 기반): 명사형/메타발화·유보/어체변형/기타.

실행: .venv/bin/python analysis/refusal_path.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import load_results  # noqa: E402

OUT = ROOT / "analysis" / "output"
LEVEL_NAMES = {1: "해라체", 2: "해체", 3: "하게체", 4: "하오체", 5: "해요체", 6: "하십시오체"}

# 메타발화/유보 표현(판단 회피)
_META = ["어렵", "애매", "모호", "판단하기", "알 수 없", "충분한", "맥락이", "정보가",
         "중립", "불분명", "확실하지", "단정", "더 구체적", "추가"]


def code_failure(text: str) -> str:
    if not text or not text.strip():
        return "빈응답"
    t = text.strip()
    if any(m in t for m in _META):
        return "메타발화·유보(판단회피)"
    # 명사형/개조식: 짧고 종결어미 없이 명사·구로 끝남
    if len(t) <= 25 and not re.search(r"[.!?]\s*$", t) and not re.search(r"(다|요|까|오|네)\s*$", t):
        return "명사형·개조식"
    # 어체 변형으로 라벨이 비표준 위치
    if ("긍정" in t or "부정" in t):
        return "어체변형·라벨매몰"
    return "기타"


def main():
    df = load_results("nsmc")
    df = df[df["model"].isin(["exaone3.5", "qwen3", "llama3.1"])].copy()

    # 1) 이중 해리 핵심 표: parse_fail vs acc_parsed_only (모델×단계)
    rows = []
    for (m, lv), g in df.groupby(["model", "level"]):
        parsed = g["parsed"]
        rows.append({
            "model": m, "level": lv, "어체": LEVEL_NAMES[lv],
            "parse_fail": round(1 - parsed.mean(), 4),
            "acc_all": round(g["correct"].mean(), 4),
            "acc_parsed_only": round(g.loc[parsed, "correct"].mean(), 4) if parsed.any() else float("nan"),
            "refusal": round(g["refused"].mean(), 4),
        })
    diss = pd.DataFrame(rows)
    diss.to_csv(OUT / "refusal_dissociation_nsmc.csv", index=False)
    print("== NSMC 이중해리: parse_fail이 오르는데 acc_parsed_only는 유지되나? ==")
    for m in ["exaone3.5", "qwen3", "llama3.1"]:
        sub = diss[diss["model"] == m].sort_values("level")
        pf = sub.set_index("어체")["parse_fail"]
        ap = sub.set_index("어체")["acc_parsed_only"]
        print(f"\n{m}:")
        print("  parse_fail :", {k: f"{v:.3f}" for k, v in pf.items()})
        print("  acc_parsed :", {k: f"{v:.3f}" for k, v in ap.items()})

    # 2) NSMC 파싱 실패 응답 유형 코딩 (모델×단계)
    fails = df[~df["parsed"]].copy()
    fails["fail_type"] = fails["response"].map(code_failure)
    ct = pd.crosstab([fails["model"], fails["level"].map(LEVEL_NAMES)], fails["fail_type"])
    ct.to_csv(OUT / "refusal_failtype_nsmc.csv")
    print("\n== NSMC 파싱실패 응답 유형 분포 (행: 모델×어체) ==")
    print(ct.to_string())

    # 3) COPA 대조: parse_fail은 낮은데 정확도가 떨어지는가
    copa = load_results("copa")
    copa = copa[copa["model"].isin(["exaone3.5", "qwen3", "llama3.1"])]
    crows = []
    for (m, lv), g in copa.groupby(["model", "level"]):
        crows.append({"model": m, "어체": LEVEL_NAMES[lv],
                      "parse_fail": round(1 - g["parsed"].mean(), 4),
                      "acc_parsed_only": round(g.loc[g["parsed"], "correct"].mean(), 4)})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT / "refusal_dissociation_copa.csv", index=False)
    print("\n== COPA 대조 (하게체 L3에서 parse_fail 낮은데 acc_parsed_only 떨어지면 = 추론손상) ==")
    for m in ["exaone3.5", "qwen3", "llama3.1"]:
        sub = cdf[cdf["model"] == m]
        print(f"  {m} 하게체: parse_fail={sub[sub['어체']=='하게체']['parse_fail'].values}, "
              f"acc_parsed={sub[sub['어체']=='하게체']['acc_parsed_only'].values}")
    print(f"\n저장: refusal_dissociation_{{nsmc,copa}}.csv, refusal_failtype_nsmc.csv")


if __name__ == "__main__":
    main()
