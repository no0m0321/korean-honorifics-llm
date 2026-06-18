"""2축 잔여 실험 데이터: HAJ(경어 적절성 판정) + NSMC 채널2(의뢰자 래퍼).

HAJ: 경어가 맞는/틀린 문장을 O/X 판정 + 교정. 생성↔판정 해리(H7) — 모델이
보충법을 생성 못 해도 위반을 판정할 수 있는가.
NSMCref: NSMC에 의뢰자 래퍼(REFERENT 존대/평어) + "결과를 ~께/에게 전하라"로
분류 정확도 직교(H5)와 출력 경어 전파를 동시 측정.

산출: data/samples/haj.jsonl, nsmcref.jsonl
실행: .venv/bin/python -m src.make_honor_data2
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# HAJ: (문장, 정답 O/X, 위반유형). 균형: 적절 30 / 주체위반 / 객체위반.
HAJ_OK = [
    "저는 어제 교수님을 뵈었습니다.", "할머니께서 진지를 드신다.",
    "민수가 할아버지께 선물을 드렸다.", "선생님께서 댁에 계신다.",
    "할아버지께서 신문을 읽으신다.", "동생이 선생님께 안부를 여쭈었다.",
    "어머니께서 일찍 주무신다.", "철수가 사장님께 보고서를 드렸다.",
    "교수님께서 다음 주에 오신다.", "제가 부장님을 모시고 갔습니다.",
    "할아버지께서 편찮으시다.", "지수가 원장님께 말씀을 여쭈었다.",
    "사장님께서 회의에 참석하신다.", "후배가 선생님을 뵙고 왔다.",
    "큰아버지께서 책을 보신다.",
]
HAJ_BAD_SUBJ = [  # (틀린문장, 교정)
    ("할머니께서 거실에 있다.", "할머니께서 거실에 계신다."),
    ("교수님이 강의실에 왔다.", "교수님께서 강의실에 오셨다."),
    ("할아버지가 밥을 먹는다.", "할아버지께서 진지를 드신다."),
    ("사장님이 지금 잔다.", "사장님께서 지금 주무신다."),
    ("선생님이 신문을 읽는다.", "선생님께서 신문을 읽으신다."),
    ("부장님이 내일 온다.", "부장님께서 내일 오신다."),
    ("원장님이 많이 아프다.", "원장님께서 많이 편찮으시다."),
    ("어머니가 회의에 간다.", "어머니께서 회의에 가신다."),
]
HAJ_BAD_OBJ = [
    ("민수가 할아버지에게 책을 줬다.", "민수가 할아버지께 책을 드렸다."),
    ("동생이 선생님을 봤다.", "동생이 선생님을 뵈었다."),
    ("나는 사장님에게 보고서를 줬다.", "나는 사장님께 보고서를 드렸다."),
    ("철수가 교수님에게 질문을 물었다.", "철수가 교수님께 질문을 여쭈었다."),
    ("지수가 할머니를 데리고 갔다.", "지수가 할머니를 모시고 갔다."),
    ("후배가 부장님에게 말을 했다.", "후배가 부장님께 말씀을 드렸다."),
    ("내가 원장님에게 선물을 줬다.", "내가 원장님께 선물을 드렸다."),
]


def make_haj():
    rows = []
    i = 0
    for s in HAJ_OK:
        rows.append({"item_id": f"haj{i}", "sentence": s, "label": "O", "vio": "none"}); i += 1
    for s, fix in HAJ_BAD_SUBJ:
        rows.append({"item_id": f"haj{i}", "sentence": s, "label": "X", "fix": fix, "vio": "subj"}); i += 1
    for s, fix in HAJ_BAD_OBJ:
        rows.append({"item_id": f"haj{i}", "sentence": s, "label": "X", "fix": fix, "vio": "obj"}); i += 1
    (ROOT / "data/samples/haj.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"HAJ: {len(rows)}건 (적절 {len(HAJ_OK)}, 주체위반 {len(HAJ_BAD_SUBJ)}, 객체위반 {len(HAJ_BAD_OBJ)})")


def make_nsmcref(n=200):
    src = [json.loads(l) for l in (ROOT / "data/samples/nsmc.jsonl").read_text(encoding="utf-8").splitlines() if l][:n]
    rows = [{"item_id": r["item_id"], "document": r["document"], "label": r["label"]} for r in src]
    (ROOT / "data/samples/nsmcref.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"NSMCref: {len(rows)}건 (NSMC 부분집합)")


if __name__ == "__main__":
    make_haj()
    make_nsmcref()
