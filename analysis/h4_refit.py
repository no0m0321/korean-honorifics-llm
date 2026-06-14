"""T1-1: H4 상호작용을 제대로 된 로지스틱 혼합모형으로 재추정.

사전등록 1순위(BinomialBayesMixedGLM)가 VB(.fit_vb)에서 수치 퇴화했으나,
같은 모형을 MAP(.fit_map, 라플라스 근사)로 적합하면 퇴화하지 않을 수 있다.
item과 template를 교차 무선효과로 분리한다(현 GEE는 둘을 단일 상관으로 뭉갬).

비교 출력: VB(실패 재현) / GEE(기존) / MAP-MixedGLM(신규, 교차 무선효과).
R lme4가 설치되면 h4_refit_lme4.R로 4번째 열을 교차검증한다.

실행: .venv/bin/python analysis/h4_refit.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import load_results  # noqa: E402

OUT = ROOT / "analysis" / "output"
MODELS = ["exaone3.5", "qwen3", "llama3.1"]


def gee(df):
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    fit = smf.gee("correct ~ C(level) * C(model)", groups="item_id", data=df,
                  cov_struct=Exchangeable(), family=Binomial()).fit()
    names = [n for n in fit.model.exog_names if ":" in n]
    w = fit.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
    return {"method": "GEE(exchangeable, item)", "chi2": float(w.statistic),
            "df": len(names), "p": float(w.pvalue)}


def map_mixedglm(df):
    """교차 무선효과(item + template) 로지스틱 혼합모형, MAP 적합."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.copy()
    d["item_id"] = d["item_id"].astype(str)
    d["template"] = d["template"].astype(str)
    vc = {"item": "0 + C(item_id)", "template": "0 + C(template)"}
    model = BinomialBayesMixedGLM.from_formula("correct ~ C(level) * C(model)", vc, d)
    res = model.fit_map()
    names = list(model.exog_names)
    ix = [i for i, n in enumerate(names) if ":" in n]
    # MAP 점추정 + 관측정보행렬 기반 Wald 동시검정
    beta = np.asarray(res.params)[ix]
    try:
        cov = np.asarray(res.cov_params())[np.ix_(ix, ix)]
    except Exception:  # noqa: BLE001
        cov = None
    if cov is not None and np.all(np.isfinite(cov)) and np.linalg.cond(cov) < 1e12:
        chi2 = float(beta @ np.linalg.solve(cov, beta))
        from scipy import stats as st
        p = float(st.chi2.sf(chi2, len(ix)))
        ok = True
    else:
        chi2, p, ok = float("nan"), float("nan"), False
    return {"method": "MAP MixedGLM(item+template crossed)", "chi2": chi2,
            "df": len(ix), "p": p, "well_conditioned": ok}


def main():
    results = {}
    for task in ("nsmc", "copa"):
        df = load_results(task)
        df = df[df["model"].isin(MODELS)].copy()
        row = {"n_obs": int(len(df)), "n_items": int(df["item_id"].nunique())}
        row["gee"] = gee(df)
        try:
            row["map_mixed"] = map_mixedglm(df)
        except Exception as e:  # noqa: BLE001
            row["map_mixed"] = {"error": f"{type(e).__name__}: {e}"}
        results[task] = row
        g, m = row["gee"], row["map_mixed"]
        print(f"\n== {task.upper()} (n={row['n_obs']}, items={row['n_items']}) ==")
        print(f"  GEE        : chi2({g['df']})={g['chi2']:.1f}, p={g['p']:.3g}")
        if "error" in m:
            print(f"  MAP mixed  : {m['error']}")
        else:
            print(f"  MAP mixed  : chi2({m['df']})={m['chi2']:.1f}, p={m['p']:.3g}, "
                  f"well_conditioned={m['well_conditioned']}")
    (OUT / "h4_refit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT / 'h4_refit.json'}")


if __name__ == "__main__":
    main()
