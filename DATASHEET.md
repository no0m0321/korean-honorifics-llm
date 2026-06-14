# Datasheet — Korean Honorifics LLM Response Corpus

Gebru et al. (2021) 7-섹션 데이터시트. 데이터 라이선스: 응답 데이터 CC BY 4.0, 코드 Apache-2.0.

## 1. Motivation
프롬프트 공손도가 LLM 성능에 미치는 영향(Yin et al., 2024)을 한국어 상대 높임법 6단계로 확장하기 위해 수집. 한국어는 종결어미만 교체해 명제 내용을 고정한 채 공손도를 순수 조작할 수 있어, 공손 효과를 어휘·구문 교란에서 분리하는 통제 실험이 가능하다.

## 2. Composition
- **단위**: `(model, task, level, template, item_id)` 키의 모델 응답 1건.
- **규모**: 113,400 응답 (오류 0). = 3 모델 × [NSMC 1,000 + COPA 1,000 + 요약 100] × 6 단계 × 3 템플릿. 민감도: NSMC temp0.7×3 = 5,400, COPA(서브셋) temp0.7×3 = 1,800.
- **모델**: EXAONE 3.5 7.8B (c7c4e3d1ca22), Qwen3-8B (500a1f067a9f), Llama 3.1 8B (46e0c10c039e), 전부 Ollama 4bit.
- **단계**: 해라체·해체·하게체·하오체·해요체·하십시오체(상대 높임법). 단계당 패러프레이즈 템플릿 3종(부록 A, `prompts/templates.yaml`).
- **출처 데이터**: NSMC(영화 리뷰 감정, CC0 추정), KoBEST COPA(인과추론, `skt/kobest_v1`), 공개 뉴스 요약(HuggingFace).
- **라벨**: 정확도용 정답 라벨 동봉. 어체 분류기 인간 검증 라벨은 **미완**(`validation_sample_200.csv` 빈칸).

## 3. Collection Process
seed=42 고정. NSMC는 테스트 분할(50K)에서 긍/부 각 500 층화 추출(10자 미만·중복 제외), COPA는 테스트 분할 1,000 전체, 요약은 기사 500–2,000자 100건. greedy(temperature=0), 서버측 배치 추론(동시 4)로 결정성 보존(재실행 일치 검증 완료). SQLite 캐시로 중단 재개.

## 4. Preprocessing / Labeling
- 채점 파서 **v1.1** 사용(`src/metrics.py`). v1.0은 "1입니다."류 선두답 응답을 실패 처리해 하게체 효과를 과대추정(−19.7%p)했고, 선두답 규칙 추가 후 전수 재채점 → −8.0%p. **v1.1 채점본이 정본이며 v1.0 수치(−19.7%p)는 인용 금지.**
- 어체: Kiwi 어말어미(EF) 규칙 분류기(`src/style_classifier.py`).

## 5. Uses
- **적합**: 경어 단계 간 비교, 모델×단계 상호작용, 출력 어체 전이, 파싱/거부 분석.
- **주의**: NSMC/KoBEST는 공개 벤치마크라 학습 데이터 오염 가능성. **오염은 모든 경어 조건에 동일 적용되므로 단계 간 비교의 내적 타당성은 유지**되나, 모델×단계 상호작용(H4) 해석 시 모델별 오염도 차이 가능성에 유의.

## 6. Distribution
`data/release/*.jsonl.gz`로 동결 배포. GitHub 공개 + Zenodo DOI 예정.

## 7. Maintenance
유지: 저자(swxvno0m@gmail.com). `make reproduce`로 응답→표·그림 재현(kiwipiepy 버전 고정 필요). 이슈·PR은 GitHub.
