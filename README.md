# korean-honorifics-llm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20740359.svg)](https://doi.org/10.5281/zenodo.20740359)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**한국어 경어법이 LLM 성능·출력에 미치는 영향** — 상대·주체·객체 높임 3축을 프롬프트 변수로 분석.
저자: 김승우 (잠신고등학교) · 논문: [paper/](paper/) · DOI: [10.5281/zenodo.20740359](https://doi.org/10.5281/zenodo.20740359)

한국어 경어법(상대 높임법 6단계)이 LLM 성능과 출력 스타일에 미치는 영향 — 실험 코드.
연구 계획서 v2.1 기준 (NSMC 1,000 / KoBEST COPA 1,000 전체 / 뉴스 요약 100).

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Ollama 모델: `ollama pull exaone3.5:7.8b qwen3:8b llama3.1:8b`
(HyperCLOVA X SEED는 transformers 백엔드 — requirements.txt의 torch/transformers 주석 해제)

## 실행 순서

```bash
# 1) 데이터 추출 (seed=42 고정, sample_ids 즉시 커밋)
python -m src.data_prep

# 2) 검정력 시뮬레이션 (사전 등록 문서용)
python analysis/power_sim.py

# 3) 어체 분류기 자가 테스트
python -m src.style_classifier

# 4) 파일럿 (아이템 30건, 단계 1·5·6)
python -m src.runner --model exaone3.5 --task nsmc --levels 1,5,6 --templates 1 --limit 30

# 5) 본 실험 (모델당 ~24h, caffeinate로 잠자기 방지, 중단 시 같은 명령으로 재개)
caffeinate -i python -m src.runner --model exaone3.5 --task nsmc
caffeinate -i python -m src.runner --model exaone3.5 --task copa
caffeinate -i python -m src.runner --model exaone3.5 --task summ
# ... qwen3, llama3.1, hcx-seed 반복

# 6) 분석 (실험 완료 후)
python analysis/stats.py accuracy --task nsmc
```

## 구조

```
prompts/templates.yaml   # 54개 템플릿 (논문 부록 A와 동일 — 동시 수정)
src/adapters.py          # ollama / transformers / genai 통합 generate()
src/runner.py            # 배치 실행 + SQLite 캐시 + 재개 + ETA 로그
src/style_classifier.py  # Kiwi 어말어미 기반 어체 분류 (κ≥0.8 검증 대상)
src/metrics.py           # 파싱·거부 판정·Distinct-N·TTR (논문 부록 B와 동일)
src/data_prep.py         # 샘플 추출 (seed=42)
analysis/power_sim.py    # 시뮬레이션 검정력 분석
analysis/stats.py        # Cochran's Q, McNemar(Holm), 혼합모형, 부트스트랩 CI
data/responses.db        # 응답 캐시 (git 제외)
data/sample_ids/         # 추출 고정 목록 (git 포함 — 변경 금지)
```

## 사전 등록 원칙

가설(H1–H4)·분석 방법은 실험 실행 전에 고정되었다 (연구계획서 v2.1).
파싱 실패는 오답으로 처리하고 실패율을 조건별 종속 변수로 보고한다.
규모 축소는 fallback 매트릭스의 트리거 발동 시에만, 발동 사실과 함께 보고한다.
