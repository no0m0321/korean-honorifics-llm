"""주체높임·객체높임 탐지기 (2축 확장). style_classifier(상대높임)와 병렬, 무간섭.

직교 측정 원칙: -시-(주체)·보충법(객체)는 **내용 명제의 제3자 논항이 명시된 절**에서만
카운트한다(지시문 래퍼의 -시-/청자겸양 '드리다' 누수 차단). Kiwi 형태소 태그 기반,
보수적 화이트리스트(Kiwi 희소형태 오분석 전례 대응). κ 생략(이진 수렴 근거).

자가 테스트: python -m src.honor_axes
"""
from __future__ import annotations

import re

from kiwipiepy import Kiwi

_kiwi: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


# 주체높임: -시-/으시 선어말어미(EP) + 보충법 주체동사(EP로 안 떨어지는 융합형)
_SI_EP = {"시", "으시"}
# Kiwi가 VV 어간 형태로 복원하는 보충법 주체동사 (시+ㄴ→신 축약 무관하게 형태 매칭)
_SUBJ_SUPPLETIVE = {"계시", "주무시", "잡수시", "자시", "드시", "편찮으시", "돌아가시", "여쭈시"}
# 객체높임: 보충법 동사 화이트리스트 (보충법어간: 평어어간)
_OBJ_SUPPLETIVE = {
    "드리": "주", "여쭙": "묻", "여쭈": "묻", "모시": "데리",
    "뵙": "보", "뵈": "보", "말씀드리": "말하",
}
# 객체높임 우언형(보조 정규식 — 커버리지 갭 명시)
_OBJ_PERIPHRASTIC = [r"갖다\s*드리", r"들려\s*드리", r"읽어\s*드리", r"해\s*드리"]
# 존대 격조사 보조 신호
_HON_PARTICLE = ["께서", "께"]


def detect_subject_hon(text: str) -> int:
    """제3자 주어 절에서 주체높임(-시-/보충법) 표지 존재 시 1."""
    kiwi = _get_kiwi()
    toks = kiwi.tokenize(text)
    for t in toks:
        if t.tag == "EP" and t.form in _SI_EP:  # 규칙적 -시-
            return 1
        if t.tag in ("VV", "VX") and t.form in _SUBJ_SUPPLETIVE:  # 보충법 주체동사
            return 1
    return 0


def detect_object_hon(text: str) -> int:
    """제3자 객체 절에서 객체높임 보충법 동사 존재 시 1."""
    kiwi = _get_kiwi()
    toks = kiwi.tokenize(text)
    stems = {t.form for t in toks if t.tag in ("VV", "VX")}
    for hon in _OBJ_SUPPLETIVE:
        # '말씀드리'는 분리될 수 있어 텍스트 매칭 보조
        if hon in stems or hon in text:
            # '말씀'(NNG) 단독 오인 방지: 말씀드리만 인정
            if hon == "말씀드리" and "말씀드리" not in text:
                continue
            return 1
    if any(re.search(p, text) for p in _OBJ_PERIPHRASTIC):
        return 1
    return 0


def honor_features(text: str) -> dict:
    return {
        "subj": detect_subject_hon(text),
        "obj": detect_object_hon(text),
        "has_hon_particle": int(any(p in text for p in _HON_PARTICLE)),
    }


def honor_overgen(text: str, input_subj: int, input_obj: int) -> dict:
    """입력 평어(−)인데 출력에 경어 표지 생성 = 과잉존대(over-honorification)."""
    f = honor_features(text)
    return {
        "subj_overgen": int(input_subj == 0 and f["subj"] == 1),
        "obj_overgen": int(input_obj == 0 and f["obj"] == 1),
        "subj_undergen": int(input_subj == 1 and f["subj"] == 0),
        "obj_undergen": int(input_obj == 1 and f["obj"] == 0),
    }


def propagation(outputs_plus, outputs_minus, axis="subj") -> float:
    """전파율 Δ = P(표지|입력+) − P(표지|입력−). 1축 미러링과 동일 메트릭 가족."""
    det = detect_subject_hon if axis == "subj" else detect_object_hon
    p_plus = sum(det(o) for o in outputs_plus) / max(len(outputs_plus), 1)
    p_minus = sum(det(o) for o in outputs_minus) / max(len(outputs_minus), 1)
    return p_plus - p_minus


if __name__ == "__main__":
    cases = [
        ("할아버지께서 지금 방에서 주무신다.", 1, 0),
        ("동생이 지금 방에서 잔다.", 0, 0),
        ("민수가 할아버지께 책을 드린다.", 0, 1),
        ("민수가 동생에게 책을 준다.", 0, 0),
        ("할아버지께 여쭈어보시었다.", 1, 1),
        ("저는 어제 교수님을 뵈었습니다.", 0, 1),
        ("할머니께서 거실에 계신다.", 1, 0),
        ("결과를 선생님께 말씀드리겠습니다.", 0, 1),
    ]
    print(f"{'subj':>4} {'obj':>4}  문장  (기대 subj/obj)")
    ok = 0
    for text, esubj, eobj in cases:
        f = honor_features(text)
        good = f["subj"] == esubj and f["obj"] == eobj
        ok += good
        print(f"{f['subj']:>4} {f['obj']:>4}  {'✓' if good else '✗'} {text}  (기대 {esubj}/{eobj})")
    print(f"\n{ok}/{len(cases)} 통과")
