"""종속 변수 계산: 파싱(정답 추출), 거부 판정, 길이·어휘 다양성.

파싱·거부 규칙은 논문 부록 B와 동일 — 수정 시 논문도 함께 수정할 것.
"""
from __future__ import annotations

import re

# 부록 B: 거부 키워드 (1차 자동 판정 — 이후 수동 확인 병행)
REFUSAL_PATTERNS = [
    "죄송하지만", "죄송합니다만", "도와드릴 수 없", "도움을 드릴 수 없",
    "할 수 없습니다", "응답할 수 없", "답변할 수 없", "답변드릴 수 없",
    "수행할 수 없", "제공할 수 없",
]

_NSMC_RE = re.compile(r"정답\s*[:：]\s*\**\s*(긍정|부정)")
_NSMC_LEAD_RE = re.compile(r"^\s*\**\s*(긍정|부정)\b")
_COPA_RE = re.compile(r"정답\s*[:：]\s*\**\s*([12])")
_COPA_LEAD_RE = re.compile(r"^\s*\**\s*([12])(?!\d)")
_COPA_DIGIT_RE = re.compile(r"(?<![\d.])([12])(?!\d)")

# 파서 v1.1 (2026-06-11): 본 실험 후 사례 점검에서 '답 선행 + 설명 후행' 응답
# (예: "부정\n\n이 리뷰는 긍정적 요소와 부정적 요소가...", "1입니다.")이
# 실패로 집계되는 것을 발견하여 선두 답(leading answer) 규칙을 2단계로 추가.
# 모든 조건에 동일 적용. 개정 전(v1.0) 채점 결과는 저장소에 함께 보관.


def parse_nsmc(text: str) -> str | None:
    """3단계 파싱: 형식 매칭 → 선두 답 → 단독 출현. 실패 시 None."""
    if not text:
        return None
    m = _NSMC_RE.search(text)
    if m:
        return m.group(1)
    m = _NSMC_LEAD_RE.match(text)
    if m:
        return m.group(1)
    has_pos, has_neg = "긍정" in text, "부정" in text
    if has_pos != has_neg:
        return "긍정" if has_pos else "부정"
    return None


def parse_copa(text: str) -> str | None:
    """3단계 파싱: 형식 매칭 → 선두 답 → 비모호 단독 출현. 실패 시 None."""
    if not text:
        return None
    m = _COPA_RE.search(text)
    if m:
        return m.group(1)
    m = _COPA_LEAD_RE.match(text)
    if m:
        return m.group(1)
    sel = set(re.findall(r"선택지\s*([12])", text))
    if len(sel) == 1:
        return sel.pop()
    digits = set(_COPA_DIGIT_RE.findall(text))
    if len(digits) == 1:
        return digits.pop()
    return None


def is_refusal(text: str) -> bool:
    return bool(text) and any(p in text for p in REFUSAL_PATTERNS)


# ---------- 길이·다양성 ----------

def eojeol_count(text: str) -> int:
    return len(text.split())


def morphemes(text: str) -> list[str]:
    """형태소 목록 (kiwipiepy). 길이·다양성 지표의 토큰 단위."""
    from .style_classifier import _get_kiwi

    return [tok.form for tok in _get_kiwi().tokenize(text)]


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    return len(ngrams) / (len(tokens) - n + 1)


def ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def summ_failed(response: str | None) -> bool:
    """요약 실패: 빈 응답. (원문 복사 검출은 ROUGE 계산 단계에서 별도 처리)"""
    return not (response or "").strip()


def rouge_l_morpheme(pred: str, ref: str) -> float:
    """형태소 단위 ROUGE-L F1 (Kiwi 토큰화, LCS 기반)."""
    p, r = morphemes(pred), morphemes(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0
