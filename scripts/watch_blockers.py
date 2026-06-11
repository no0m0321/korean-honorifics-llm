"""사람 액션 대기 조건 감시: 풀리는 순간 한 줄 출력 (Monitor용).

감시 대상: κ 라벨링 완료 / 휴먼 평가 시트 회수 / HF 로그인 / GEMINI_API_KEY / git remote
"""
import csv
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
seen: set[str] = set()


def emit(key: str, msg: str) -> None:
    if key not in seen:
        print(f"{key}: {msg}", flush=True)
        seen.add(key)


def filled(path: Path, col: str, need: int) -> bool:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return sum(1 for r in rows if (r.get(col) or "").strip()) >= need


while True:
    try:
        if filled(ROOT / "analysis/output/validation_sample_200.csv", "human_label", 200):
            emit("KAPPA_READY", "κ 검증 라벨 200건 완료 — κ 산출 가능")
    except Exception:
        pass
    for rater in "AB":
        try:
            if filled(ROOT / f"analysis/output/human_eval/평가시트_평가자{rater}.csv", "적절성(1-5)", 120):
                emit(f"EVAL_{rater}_READY", f"평가자{rater} 시트 채점 완료")
        except Exception:
            pass
    try:
        from huggingface_hub import whoami

        whoami()
        emit("HF_READY", "HuggingFace 로그인 감지 — HCX SEED 실행 가능")
    except Exception:
        pass
    try:
        out = subprocess.run(["zsh", "-lc", 'printf %s "$GEMINI_API_KEY"'],
                             capture_output=True, text=True, timeout=30).stdout
        if out.strip():
            emit("GEMINI_READY", "GEMINI_API_KEY 감지 — Gemini 서브셋 실행 가능")
    except Exception:
        pass
    try:
        if subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                          capture_output=True, timeout=30).returncode == 0:
            emit("REMOTE_READY", "git remote 연결 감지 — push 가능")
    except Exception:
        pass
    time.sleep(180)
