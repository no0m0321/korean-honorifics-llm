"""데이터 추출 (v2.1 규칙, seed=42 고정).

- NSMC : 공식 테스트 분할(GitHub raw)에서 긍정 500 + 부정 500 층화 추출
- COPA : KoBEST 테스트 분할 1,000건 전체 (추출 없음)
- summ : 공개 뉴스 요약 데이터셋에서 기사 500~2,000자 필터 후 100건

산출물:
  data/samples/{nsmc,copa,summ}.jsonl   — runner 입력
  data/sample_ids/{task}.txt            — 추출 즉시 고정, 이후 변경 금지

실행: python -m src.data_prep
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
NSMC_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"


def _write(task: str, rows: list[dict]) -> None:
    samples = ROOT / "data" / "samples"
    ids_dir = ROOT / "data" / "sample_ids"
    samples.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)
    with open(samples / f"{task}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (ids_dir / f"{task}.txt").write_text(
        "\n".join(str(r["item_id"]) for r in rows) + "\n", encoding="utf-8")
    print(f"[data_prep] {task}: {len(rows)}건 저장")


def prep_nsmc(n_per_class: int = 500) -> None:
    raw = urllib.request.urlopen(NSMC_URL, timeout=60).read().decode("utf-8")
    rows = []
    seen = set()
    for line in raw.splitlines()[1:]:  # header: id \t document \t label
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        iid, doc, label = parts
        doc = doc.strip()
        if len(doc) < 10 or doc in seen:  # 너무 짧거나 중복인 리뷰 제외
            continue
        seen.add(doc)
        rows.append({"item_id": iid, "document": doc, "label": "긍정" if label == "1" else "부정"})
    rng = random.Random(SEED)
    pos = [r for r in rows if r["label"] == "긍정"]
    neg = [r for r in rows if r["label"] == "부정"]
    picked = rng.sample(pos, n_per_class) + rng.sample(neg, n_per_class)
    rng.shuffle(picked)
    _write("nsmc", picked)


def prep_copa() -> None:
    rows = [dict(r) for r in _load_kobest_copa_test()]
    out = []
    qmap = {"cause": "원인", "effect": "결과"}
    for i, r in enumerate(rows):
        out.append({
            "item_id": str(i),
            "premise": r["premise"],
            "question": qmap.get(str(r["question"]).strip(), str(r["question"]).strip()),
            "alternative_1": r["alternative_1"],
            "alternative_2": r["alternative_2"],
            "label": str(int(r["label"]) + 1),  # 0/1 -> "1"/"2"
        })
    _write("copa", out)  # 테스트 분할 전체 — 추출 없음


def _load_kobest_copa_test():
    try:
        from datasets import load_dataset

        return load_dataset("skt/kobest_v1", "copa", split="test")
    except Exception as e:  # 스크립트 기반 로딩 실패 시 parquet 변환본 폴백
        print(f"[data_prep] load_dataset 실패({type(e).__name__}) — parquet 변환본으로 폴백")
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = api.list_repo_files("skt/kobest_v1", repo_type="dataset",
                                    revision="refs/convert/parquet")
        targets = [f for f in files if "copa" in f and "test" in f and f.endswith(".parquet")]
        if not targets:
            raise SystemExit("KoBEST COPA parquet 파일을 찾지 못했습니다") from e
        frames = [pd.read_parquet(hf_hub_download(
            "skt/kobest_v1", f, repo_type="dataset", revision="refs/convert/parquet"))
            for f in targets]
        return pd.concat(frames).to_dict("records")


def prep_summ(n: int = 100, min_len: int = 500, max_len: int = 2000) -> None:
    from datasets import load_dataset

    ds = None
    for split in ("test", "validation", "train"):
        try:
            ds = load_dataset("daekeun-ml/naver-news-summarization-ko", split=split)
            break
        except ValueError:
            continue
    if ds is None:
        raise SystemExit("뉴스 요약 데이터셋 로딩 실패")
    pool = [
        {"item_id": str(i), "document": r["document"], "summary": r["summary"],
         "category": r.get("category", "")}
        for i, r in enumerate(ds)
        if min_len <= len(r["document"]) <= max_len
    ]
    rng = random.Random(SEED)
    _write("summ", rng.sample(pool, n))


if __name__ == "__main__":
    prep_nsmc()
    prep_copa()
    prep_summ()
    print("[data_prep] 완료 — data/sample_ids/ 를 git에 커밋해 고정하세요.")
