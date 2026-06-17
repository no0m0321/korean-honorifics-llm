"""2축 확장 데이터 생성 (HSC 완성 태스크 + memo 미러링).

설계 결정 반영: 신설 HSC 핵심(상대높임=해요체 1종 고정), 전파·과잉존대 중심.
높임 대상은 항상 제3자(2인칭 주어 금지) → -시-가 청자높임으로 새지 않음.
채점은 gold 문자열이 아니라 honor_axes로 셀 조건 일치 여부를 측정(전파 중심).

inclusion gate: 무생물 주어·2인칭·압존 맥락 배제(인벤토리 자체가 3인칭 존대가능 인물).

산출: data/samples/hsc.jsonl, memo_summ.jsonl
실행: .venv/bin/python -m src.make_honor_data
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

HON_SUBJ = ["할아버지", "할머니", "교수님", "사장님", "선생님", "부장님", "아버지", "어머니", "원장님", "회장님"]
PLN_SUBJ = ["동생", "친구", "민수", "철수", "후배", "누나", "조카", "사촌", "지수", "현우"]
HON_RECIP = ["할아버지", "할머니", "교수님", "사장님", "선생님", "부장님", "원장님", "큰아버지"]
PLN_RECIP = ["동생", "친구", "민수", "철수", "후배", "조카", "지수", "현우"]

# 양도 동사(2×2 subj×obj 지원): 사전형, 객체높임 보충법 여부
DITRANS = [
    {"verb": "주다", "theme": ["선물", "책", "편지", "용돈", "꽃", "사진"]},
    {"verb": "묻다", "theme": ["길", "안부", "이유", "방법", "사정"]},
    {"verb": "전하다", "theme": ["소식", "말씀", "감사 인사", "결과"]},
    {"verb": "보내다", "theme": ["선물", "자료", "초대장", "편지"]},
]
# 주어높임 전용(자동사/형용사: 객체 없음)
SUBJ_ONLY = [
    {"verb": "자다", "adj": ["지금 방에서", "소파에서", "일찍"]},
    {"verb": "오다", "adj": ["3시에", "방금", "회의에"]},
    {"verb": "읽다", "adj": ["신문을", "그 책을", "편지를"]},
    {"verb": "먹다", "adj": ["저녁을", "약을", "과일을"]},
    {"verb": "아프다", "adj": ["요즘", "어제부터", "많이"]},
    {"verb": "말하다", "adj": ["천천히", "그렇게", "조용히"]},
]


def _sb(name, honored):
    return f"{name}께서" if honored else f"{name}{'가' if name[-1] in '수지우나' or _open(name) else '이'}".replace("께서", "께서")


def _open(s):
    # 받침 없는 끝글자면 '가', 있으면 '이' (간이 판정)
    last = s[-1]
    return (ord(last) - 0xAC00) % 28 == 0 if 0xAC00 <= ord(last) <= 0xD7A3 else True


def subj_particle(name, honored):
    if honored:
        return f"{name}께서"
    return f"{name}{'가' if _open(name) else '이'}"


def recip_particle(name, honored):
    return f"{name}께" if honored else f"{name}에게"


def obj_particle(noun):
    return f"{noun}{'를' if _open(noun) else '을'}"


def main():
    rng = random.Random(SEED)
    hsc = []
    iid = 0

    # 블록 A: 양도동사 2×2 (subj × obj) + 객체 굴절통제판
    base_ditrans = []
    for d in DITRANS:
        for _ in range(7):  # 동사당 7 상황 → 28 기본상황
            base_ditrans.append({"verb": d["verb"], "theme": rng.choice(d["theme"])})
    for b in base_ditrans:
        s_hon, s_pln = rng.choice(HON_SUBJ), rng.choice(PLN_SUBJ)
        r_hon, r_pln = rng.choice(HON_RECIP), rng.choice(PLN_RECIP)
        for subj in (0, 1):
            for obj_cond in ("minus", "plus", "plus_ctrl"):
                S = subj_particle(s_hon if subj else s_pln, subj)
                obj_honored = obj_cond != "minus"
                R = recip_particle(r_hon if obj_honored else r_pln, obj_honored)
                tag = " [동사 어간은 바꾸지 말고]" if obj_cond == "plus_ctrl" else ""
                sit = f"{S} {R} {obj_particle(b['theme'])} ({b['verb']})."
                hsc.append({
                    "item_id": f"hsc{iid}", "block": "A", "subj": subj,
                    "obj": 0 if obj_cond == "minus" else 1,
                    "obj_ctrl": int(obj_cond == "plus_ctrl"),
                    "situation": sit + tag, "verb": b["verb"]})
                iid += 1

    # 블록 A2: 주어높임 전용 (obj n/a)
    for d in SUBJ_ONLY:
        for _ in range(4):
            s_hon, s_pln = rng.choice(HON_SUBJ), rng.choice(PLN_SUBJ)
            adj = rng.choice(d["adj"])
            for subj in (0, 1):
                S = subj_particle(s_hon if subj else s_pln, subj)
                hsc.append({
                    "item_id": f"hsc{iid}", "block": "A2", "subj": subj, "obj": -1, "obj_ctrl": 0,
                    "situation": f"{S} {adj} ({d['verb']}).", "verb": d["verb"]})
                iid += 1

    (ROOT / "data/samples/hsc.jsonl").write_text(
        "\n".join(json.dumps(h, ensure_ascii=False) for h in hsc) + "\n", encoding="utf-8")
    print(f"HSC: {len(hsc)}건 (블록A 양도 {sum(h['block']=='A' for h in hsc)}, A2 주어전용 {sum(h['block']=='A2' for h in hsc)})")

    # memo 미러링: 입력 -시- ± 쌍. 자동 활용 대신 정확한 +시/−시 표면형 큐레이션.
    memo = []
    actors_h = ["부장님", "사장님", "교수님", "원장님", "선생님"]
    actors_p = ["김 대리", "이 사원", "박 주임", "최 씨", "후배"]
    # (−시 평어형, +시 존대형)
    EVENTS = [
        ("3시에 회의실로 온다고 했다", "3시에 회의실로 오신다고 하셨다"),
        ("내일 출장을 간다고 했다", "내일 출장을 가신다고 하셨다"),
        ("보고서를 검토한다고 했다", "보고서를 검토하신다고 하셨다"),
        ("점심에 도착한다고 했다", "점심에 도착하신다고 하셨다"),
        ("행사에 참석한다고 했다", "행사에 참석하신다고 하셨다"),
        ("회의를 준비한다고 했다", "회의를 준비하신다고 하셨다"),
        ("자료를 보낸다고 했다", "자료를 보내신다고 하셨다"),
        ("일찍 퇴근한다고 했다", "일찍 퇴근하신다고 하셨다"),
    ]
    for i in range(40):
        minus, plus = rng.choice(EVENTS)
        ah, ap = rng.choice(actors_h), rng.choice(actors_p)
        memo.append({"item_id": f"memo{i}_plus", "subj": 1, "memo": f"{ah}께서 {plus}."})
        memo.append({"item_id": f"memo{i}_minus", "subj": 0, "memo": f"{subj_particle(ap, 0)} {minus}."})
    (ROOT / "data/samples/memo.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in memo) + "\n", encoding="utf-8")
    print(f"memo: {len(memo)}건 (입력 -시- ± 쌍)")


if __name__ == "__main__":
    main()
