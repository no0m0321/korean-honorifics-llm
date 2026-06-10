"""최종 분석: 전 지표 집계 + H4 상호작용 검정 + 그림 + κ 검증 샘플.

실행: .venv/bin/python analysis/final_analysis.py
산출: analysis/output/*.csv, figures/*.png, validation_sample_200.csv
"""
from __future__ import annotations

import json
import signal
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis" / "output"
FIG = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

from analysis.stats import bootstrap_ci, load_results  # noqa: E402
from src import metrics  # noqa: E402
from src.style_classifier import classify  # noqa: E402

MODELS = ["exaone3.5", "qwen3", "llama3.1"]
LEVEL_NAMES = {1: "해라체", 2: "해체", 3: "하게체", 4: "하오체", 5: "해요체", 6: "하십시오체"}
STYLE_ORDER = ["해라체", "해체", "하게체", "하오체", "해요체", "하십시오체", "판정불가"]


def db():
    return sqlite3.connect(ROOT / "data" / "responses.db")


# ---------- 1. 정확도 표 (CI 포함) + 파싱·거부 ----------
def accuracy_tables():
    for task in ("nsmc", "copa"):
        df = load_results(task)
        rows = []
        for m in MODELS:
            g = df[df["model"] == m]
            for lv in range(1, 7):
                vals = g.loc[g["level"] == lv, "correct"].to_numpy()
                lo, hi = bootstrap_ci(vals, reps=10_000)
                sub = g[g["level"] == lv]
                rows.append({
                    "model": m, "level": lv, "acc": vals.mean(),
                    "ci_lo": lo, "ci_hi": hi,
                    "parse_fail": 1 - sub["parsed"].mean(),
                    "refusal": sub["refused"].mean(),
                    "acc_parsed_only": sub.loc[sub["parsed"], "correct"].mean(),
                })
        pd.DataFrame(rows).to_csv(OUT / f"final_accuracy_{task}.csv", index=False)
        print(f"[1] {task} 정확도 표 저장")


# ---------- 2. 요약: ROUGE·스타일·전이 ----------
def summ_tables():
    refs = {json.loads(l)["item_id"]: json.loads(l)["summary"]
            for l in (ROOT / "data" / "samples" / "summ.jsonl").read_text(encoding="utf-8").splitlines() if l}
    d = pd.read_sql_query(
        "SELECT model, level, template, item_id, response FROM responses "
        "WHERE task='summ' AND response IS NOT NULL", db())
    d["rouge_l"] = d.apply(lambda r: metrics.rouge_l_morpheme(r["response"], refs[r["item_id"]]), axis=1)
    d["style"] = d["response"].map(classify)
    d["eojeol"] = d["response"].map(metrics.eojeol_count)
    morphs = d["response"].map(metrics.morphemes)
    d["distinct1"] = morphs.map(lambda t: metrics.distinct_n(t, 1))
    d["distinct2"] = morphs.map(lambda t: metrics.distinct_n(t, 2))
    d["ttr"] = morphs.map(metrics.ttr)

    agg = d.groupby(["model", "level"]).agg(
        rouge_l=("rouge_l", "mean"), eojeol=("eojeol", "mean"),
        distinct1=("distinct1", "mean"), distinct2=("distinct2", "mean"), ttr=("ttr", "mean"),
    ).round(4)
    agg.to_csv(OUT / "final_summ_metrics.csv")
    print("[2] 요약 지표 저장")
    for m in MODELS:
        mat = pd.crosstab(d.loc[d["model"] == m, "level"], d.loc[d["model"] == m, "style"],
                          normalize="index").reindex(columns=STYLE_ORDER, fill_value=0)
        (mat * 100).round(1).to_csv(OUT / f"final_transition_{m}.csv")
    print("[2] 어체 전이 매트릭스 3종 저장")
    return d


# ---------- 3. H4: level x model 상호작용 (혼합모형 -> GEE 폴백) ----------
class _Timeout(Exception):
    pass


def interaction_test():
    results = {}
    for task in ("nsmc", "copa"):
        df = load_results(task)
        df = df[df["model"].isin(MODELS)].copy()
        df["level"] = df["level"].astype("category")
        df["model"] = pd.Categorical(df["model"], categories=MODELS)
        used = None
        try:  # 사전 등록 1순위: 로지스틱 혼합효과 (변분 근사) — 300초 제한
            signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
            signal.alarm(300)
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

            md = BinomialBayesMixedGLM.from_formula(
                "correct ~ C(level) * C(model)", {"item": "0 + C(item_id)"}, df)
            fit = md.fit_vb()
            signal.alarm(0)
            used = "BinomialBayesMixedGLM(VB)"
            names = list(fit.model.exog_names)
            ix = [i for i, n in enumerate(names) if ":" in n]
            # VB 사후분포 기반 Wald 근사 (상호작용 항 동시 검정)
            beta = fit.fe_mean[ix]
            sd = fit.fe_sd[ix]
            from scipy import stats as st

            chi2 = float(np.sum((beta / sd) ** 2))
            p = float(st.chi2.sf(chi2, len(ix)))
        except (_Timeout, MemoryError, Exception) as e:  # noqa: BLE001
            signal.alarm(0)
            print(f"[3] {task}: 혼합모형 불가({type(e).__name__}) → GEE 폴백")
            import statsmodels.formula.api as smf
            from statsmodels.genmod.cov_struct import Exchangeable
            from statsmodels.genmod.families import Binomial

            md = smf.gee("correct ~ C(level) * C(model)", groups="item_id", data=df,
                         cov_struct=Exchangeable(), family=Binomial())
            fit = md.fit()
            used = "GEE(exchangeable, cluster=item)"
            names = [n for n in fit.model.exog_names if ":" in n]
            w = fit.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
            chi2, p = float(w.statistic), float(w.pvalue)
        results[task] = {"method": used, "chi2": round(chi2, 1),
                         "df": 10, "p": p}
        print(f"[3] {task}: {used} 상호작용 chi2={chi2:.1f}, p={p:.3g}")
    (OUT / "final_interaction_h4.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


# ---------- 4. 그림 ----------
def figures(summ_df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["AppleGothic", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task, title in zip(axes, ("nsmc", "copa"), ("NSMC (classification)", "KoBEST COPA (reasoning)")):
        t = pd.read_csv(OUT / f"final_accuracy_{task}.csv")
        for m, marker in zip(MODELS, "os^"):
            g = t[t["model"] == m]
            ax.errorbar(g["level"], g["acc"], yerr=[g["acc"] - g["ci_lo"], g["ci_hi"] - g["acc"]],
                        marker=marker, capsize=3, label=m)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([LEVEL_NAMES[i] for i in range(1, 7)], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_level.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, m in zip(axes, MODELS):
        mat = pd.read_csv(OUT / f"final_transition_{m}.csv", index_col=0)
        im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, fontsize=7)
        ax.set_yticks(range(6))
        ax.set_yticklabels([LEVEL_NAMES[i] for i in range(1, 7)], fontsize=7)
        ax.set_title(f"{m} (summary)", fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iloc[i, j]
                if v >= 1:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 50 else "black")
    fig.colorbar(im, ax=axes, fraction=0.02, label="%")
    fig.savefig(FIG / "style_transition.png", dpi=150, bbox_inches="tight")
    print("[4] 그림 2종 저장")


# ---------- 5. κ 검증용 층화 샘플 200 ----------
def validation_sample():
    d = pd.read_sql_query(
        "SELECT model, task, level, template, item_id, response FROM responses "
        "WHERE response IS NOT NULL AND task='summ'", db())
    # 모델(3) x 단계(6) 셀당 11~12건 -> 200건
    rng = np.random.default_rng(42)
    picks = []
    for (m, lv), g in d.groupby(["model", "level"]):
        n = 12 if len(picks) < 100 else 11
        picks.append(g.iloc[rng.choice(len(g), size=min(n, len(g)), replace=False)])
    sample = pd.concat(picks).head(200).copy()
    sample["human_label"] = ""  # 라벨러 기입란
    sample["classifier_label"] = sample["response"].map(classify)
    cols = ["model", "task", "level", "template", "item_id", "response", "human_label"]
    sample[cols].to_csv(OUT / "validation_sample_200.csv", index=False)
    sample.to_csv(OUT / "validation_sample_200_with_classifier.csv", index=False)  # 채점 대조용(라벨링 후 열람)
    print(f"[5] κ 검증 샘플 {len(sample)}건 저장 (human_label 칸 채운 뒤 κ 산출)")


if __name__ == "__main__":
    accuracy_tables()
    sd = summ_tables()
    interaction_test()
    figures(sd)
    validation_sample()
    print("최종 분석 완료 — analysis/output/, analysis/figures/ 확인")
