"""사전 등록 통계 분석 (논문 6장·표 6과 동일).

  - 정확도 단계 간: Cochran's Q -> 사후 pairwise McNemar (Holm 보정)
  - 단계 x 모델 상호작용(H4): 로지스틱 혼합효과 (statsmodels BinomialBayesMixedGLM)
  - 연속 지표: 선형 혼합효과(mixedlm), 비모수 대안 Friedman
  - 모든 평균: 부트스트랩 95% CI (재표집 10,000회)

실행 (실험 완료 후):
  python analysis/stats.py accuracy --task nsmc   # 모델x단계 정확도 표 + Cochran's Q
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402

PARSERS = {"nsmc": metrics.parse_nsmc, "copa": metrics.parse_copa}


def load_results(task: str, db_path: Path | None = None) -> pd.DataFrame:
    """responses.db + 정답 라벨 -> (model, level, template, item_id, correct) 프레임."""
    db = sqlite3.connect(db_path or ROOT / "data" / "responses.db")
    df = pd.read_sql_query(
        "SELECT model, level, template, item_id, response FROM responses "
        "WHERE task = ? AND response IS NOT NULL", db, params=(task,))
    gold = {
        str(json.loads(line)["item_id"]): json.loads(line)["label"]
        for line in (ROOT / "data" / "samples" / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    parser = PARSERS[task]
    df["pred"] = df["response"].map(parser)
    df["gold"] = df["item_id"].map(gold)
    df["parsed"] = df["pred"].notna()
    df["refused"] = df["response"].map(metrics.is_refusal)
    df["correct"] = (df["pred"] == df["gold"]).astype(int)  # 파싱 실패 = 오답 처리(사전 등록)
    return df


def bootstrap_ci(values: np.ndarray, reps: int = 10_000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cochrans_q(wide: pd.DataFrame) -> tuple[float, float]:
    """행=아이템, 열=조건(단계)의 0/1 행렬에 대한 Cochran's Q."""
    from statsmodels.stats.contingency_tables import cochrans_q as _cq

    res = _cq(wide.to_numpy())
    return float(res.statistic), float(res.pvalue)


def pairwise_mcnemar_holm(wide: pd.DataFrame) -> pd.DataFrame:
    """모든 단계 쌍 McNemar + Holm 보정."""
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    cols = list(wide.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = wide[cols[i]], wide[cols[j]]
            tbl = pd.crosstab(a, b).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            p = mcnemar(tbl.to_numpy(), exact=True).pvalue
            rows.append({"pair": f"{cols[i]} vs {cols[j]}", "p_raw": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    return out.sort_values("p_holm")


def mixed_logit_interaction(df: pd.DataFrame):
    """H4: correct ~ level * model + (1|item) — 베이즈 근사 로지스틱 혼합모형.

    statsmodels에는 빈도주의 로지스틱 혼합모형이 없어 BinomialBayesMixedGLM을 쓴다.
    (대안: R lme4::glmer 또는 pymer4 — 결과 교차 확인용)
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.copy()
    d["level"] = d["level"].astype("category")
    d["model"] = d["model"].astype("category")
    model = BinomialBayesMixedGLM.from_formula(
        "correct ~ C(level) * C(model)", {"item": "0 + C(item_id)"}, d)
    return model.fit_vb()


def friedman(wide: pd.DataFrame) -> tuple[float, float]:
    from scipy.stats import friedmanchisquare

    stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
    return float(stat), float(p)


def accuracy_report(task: str) -> None:
    df = load_results(task)
    if df.empty:
        raise SystemExit(f"{task}: responses.db에 결과가 없습니다")
    out_dir = ROOT / "analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for model, g in df.groupby("model"):
        accs = g.groupby("level")["correct"].mean()
        cell = []
        for lv in sorted(accs.index):
            vals = g.loc[g["level"] == lv, "correct"].to_numpy()
            lo, hi = bootstrap_ci(vals)
            cell.append(f"L{lv}: {accs[lv]:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"{model}: " + " | ".join(cell))
        # 단계 간 검정 (아이템x템플릿 단위 대응, 템플릿 평균이 아닌 관측 단위 유지)
        wide = g.pivot_table(index=["item_id", "template"], columns="level",
                             values="correct", aggfunc="first").dropna()
        q, p = cochrans_q(wide)
        lines.append(f"  Cochran's Q = {q:.2f}, p = {p:.2g}, n_obs = {len(wide)}")
        if p < 0.05:
            lines.append(pairwise_mcnemar_holm(wide).to_string(index=False))
        lines.append(f"  파싱 실패율: {1 - g['parsed'].mean():.3%}, 거부율: {g['refused'].mean():.3%}")
    report = "\n".join(lines)
    print(report)
    (out_dir / f"accuracy_{task}.txt").write_text(report, encoding="utf-8")
    df.to_csv(out_dir / f"scored_{task}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["accuracy"])
    ap.add_argument("--task", required=True, choices=["nsmc", "copa"])
    args = ap.parse_args()
    accuracy_report(args.task)
