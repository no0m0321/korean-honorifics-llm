"""Kiwi 어말어미(EF) 기반 출력 어체 분류기 (v1 — 인간 검증 전).

분류 절차 (논문 5장과 동일):
  1) 응답을 문장 단위로 분리 (kiwi.split_into_sents)
  2) 각 문장의 마지막 어말어미(EF) 형태를 규칙으로 어체에 사상 (긴 접미사 우선)
  3) 응답 어체 = 문장 판정 다수결, 동률이면 마지막 문장 우선
  4) EF가 없는 응답(명사형 종결·개조식)은 '판정불가'

검증: 무작위 200건 인간 라벨과 Cohen's κ ≥ 0.8 통과 기준 (미달 시 규칙 보완 후
새 200건으로 재검증). 규칙 수정 시 이 docstring에 변경 이력을 남길 것.

자가 테스트: python -m src.style_classifier
"""
from __future__ import annotations

from collections import Counter

from kiwipiepy import Kiwi

LEVELS = ["해라체", "해체", "하게체", "하오체", "해요체", "하십시오체"]
UNKNOWN = "판정불가"

# (접미사, 어체) — 긴 접미사 우선 매칭. 'ᆸ'은 Kiwi가 쓰는 받침 자모(예: 'ᆸ니다').
_RULES: list[tuple[str, str]] = [
    # 하십시오체
    ("습니다", "하십시오체"), ("ᆸ니다", "하십시오체"), ("습니까", "하십시오체"),
    ("ᆸ니까", "하십시오체"), ("십시오", "하십시오체"), ("ᆸ시오", "하십시오체"),
    ("ᆸ시다", "하십시오체"),
    # 해요체 (일반 '요' 종결은 최후순위 캐치올로 별도 처리)
    ("어요", "해요체"), ("아요", "해요체"), ("에요", "해요체"), ("예요", "해요체"),
    ("세요", "해요체"), ("셔요", "해요체"), ("네요", "해요체"), ("지요", "해요체"),
    ("죠", "해요체"), ("는데요", "해요체"), ("거든요", "해요체"), ("까요", "해요체"),
    # 하오체
    ("구려", "하오체"), ("시오", "하오체"), ("소", "하오체"),
    # 하게체
    ("게나", "하게체"), ("는가", "하게체"), ("네", "하게체"), ("게", "하게체"),
    ("세", "하게체"), ("나", "하게체"),
    # 해라체
    ("는다", "해라체"), ("ᆫ다", "해라체"), ("어라", "해라체"), ("아라", "해라체"),
    ("려무나", "해라체"), ("렴", "해라체"), ("냐", "해라체"), ("니", "해라체"),
    ("자", "해라체"), ("라", "해라체"), ("다", "해라체"),
    # 해체
    ("는데", "해체"), ("거든", "해체"), ("군", "해체"), ("지", "해체"),
    ("야", "해체"), ("어", "해체"), ("아", "해체"), ("대", "해체"),
    # 캐치올: 위에 안 걸린 '~요' 종결과 '~오' 종결
    ("요", "해요체"), ("오", "하오체"),
]
_RULES.sort(key=lambda r: len(r[0]), reverse=True)

# 하오체·하게체는 희소해서 Kiwi가 어말어미를 EC(연결어미)로 오분석하는 경우가 있다
# (예: '재미있소' -> 소/EC). 문장 끝 EC에 한해 아래 형태만 보수적으로 인정한다.
_EC_FALLBACK = {"소": "하오체", "구려": "하오체", "게나": "하게체"}

_kiwi: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def classify_ef(form: str) -> str | None:
    """어말어미 형태 하나를 어체로 사상."""
    for suffix, level in _RULES:
        if form.endswith(suffix):
            return level
    return None


def classify_sentences(text: str) -> list[str | None]:
    """문장별 어체 판정 목록."""
    kiwi = _get_kiwi()
    results: list[str | None] = []
    for sent in kiwi.split_into_sents(text, return_tokens=True):
        efs = [tok.form for tok in sent.tokens if tok.tag == "EF"]
        if efs:
            results.append(classify_ef(efs[-1]))
            continue
        # EF 부재 시: 문장부호 앞 마지막 형태소가 EC이면 폴백 규칙 확인
        morphs = [tok for tok in sent.tokens if not tok.tag.startswith("S")]
        if morphs and morphs[-1].tag == "EC":
            results.append(_EC_FALLBACK.get(morphs[-1].form))
        else:
            results.append(None)
    return results


def classify(text: str) -> str:
    """응답 수준 어체: 다수결, 동률은 마지막 문장 우선, 전부 판정 불가면 UNKNOWN."""
    per_sent = [r for r in classify_sentences(text) if r is not None]
    if not per_sent:
        return UNKNOWN
    counts = Counter(per_sent)
    top = counts.most_common()
    best, best_n = top[0]
    tied = [lvl for lvl, n in top if n == best_n]
    if len(tied) > 1:
        for r in reversed(per_sent):
            if r in tied:
                return r
    return best


if __name__ == "__main__":
    samples = [
        ("이 영화는 정말 재미있다.", "해라체"),
        ("이 영화 정말 재미있어.", "해체"),
        ("이 영화는 정말 재미있게 보았네.", "하게체"),
        ("이 영화는 정말 재미있소.", "하오체"),
        ("이 영화는 정말 재미있어요.", "해요체"),
        ("이 영화는 정말 재미있습니다.", "하십시오체"),
        ("정답: 긍정", "판정불가(명사형)"),
        ("이 리뷰는 긍정적입니다. 전반적으로 호평이에요.", "다수결/동률 처리 확인"),
    ]
    for text, expected in samples:
        print(f"{classify(text):8s} <- {text}   (기대: {expected})")
