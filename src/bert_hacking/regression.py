"""Logit regressions of metadata on gold vs predicted labels, and the resulting I / II / S / M errors.

For each covariate column and each of its values v: logit(1[covariate == v] ~ 1 + label).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

SIGNIFICANCE = 0.05


def logit(y: pd.Series, x: pd.Series, optimizer: str = "bfgs") -> dict:
    """Fit logit(y ~ 1 + x) on boolean series and return the key statistics."""
    exog = sm.add_constant(x.to_numpy().astype(int))
    try:
        fitted_model = sm.Logit(y.to_numpy().astype(int), exog)
        res = fitted_model.fit(maxiter=100, method=optimizer, disp=0)
        return {
            "success": True,
            "optimizer": optimizer,
            "Pseudo R-squared": res.prsquared,
            "Covariate Names": fitted_model.exog_names,
            "Coef": res.params.tolist(),
            "Std err": res.bse.tolist(),
            "z": res.tvalues.tolist(),
            "pvalues": res.pvalues.tolist(),
            "Conf Int": res.conf_int().tolist(),
            "Log-Likelihood": res.llf,
            "LL-Null": res.llnull,
            "LLR p-value": res.llr_pvalue,
            "AIC": res.aic,
            "BIC": res.bic,
            "N obs": res.nobs,
        }
    except Exception as e:
        return {"success": False, "optimizer": optimizer, "error": repr(e)}


def _imbalance(labels: pd.Series) -> float:
    """|P(label) - P(not label)|."""
    return abs(labels.mean() - (~labels).mean())


def errors(gold: dict, pred: dict, gold_labels: pd.Series, pred_labels: pd.Series) -> dict:
    """Type I / II / S / M errors of the predicted-label regression w.r.t. the gold one.

    Type M follows PROTOCOLE_EXP.md: relative change in label imbalance, when both
    regressions are significant with the same sign. Values are None when undefined.
    """
    if not (gold["success"] and pred["success"]):
        return {}
    # add_constant skips the constant when x is constant, in which case there is no "x1"
    if "x1" not in gold["Covariate Names"] or "x1" not in pred["Covariate Names"]:
        return {"error": "can't find x1"}
    gold_p, gold_coef = (gold[key][gold["Covariate Names"].index("x1")] for key in ("pvalues", "Coef"))
    pred_p, pred_coef = (pred[key][pred["Covariate Names"].index("x1")] for key in ("pvalues", "Coef"))
    gold_significant, pred_significant = gold_p < SIGNIFICANCE, pred_p < SIGNIFICANCE
    both_significant = gold_significant and pred_significant

    gold_imbalance = _imbalance(gold_labels)
    error_m = (float(both_significant and gold_coef * pred_coef >= 0)
               * abs(_imbalance(pred_labels) / gold_imbalance - 1)
               if gold_imbalance != 0 else np.nan)

    def unless_nan(value, *inputs):
        return None if np.isnan(inputs).any() else value

    return {
        "h_t": unless_nan(gold_significant, gold_p),
        "error_type_1": unless_nan(pred_significant and not gold_significant, gold_p, pred_p),
        "error_type_2": unless_nan(gold_significant and not pred_significant, gold_p, pred_p),
        "error_type_S": unless_nan(both_significant and gold_coef * pred_coef < 0,
                                   gold_p, pred_p, gold_coef, pred_coef),
        "error_type_M": unless_nan(error_m, gold_p, pred_p, gold_coef, pred_coef, error_m),
    }


def hypotheses(df: pd.DataFrame, covariates: list[str], min_size: int = 0) -> list[tuple[str, object]]:
    """(covariate, value) pairs to test: values with at least `min_size` documents, most
    frequent first. Missing values are never tested."""
    return [(col, value) for col in covariates
            for value, count in df[col].value_counts().items() if count >= min_size]


def gold_regressions(df: pd.DataFrame, label: str, covariates: list[str],
                     min_size: int = 0) -> dict[tuple[str, object], dict]:
    """`df` has columns GS-LABEL and the covariates. Defines the hypotheses of the task."""
    gold_labels = df["GS-LABEL"] == label
    return {(col, value): logit(df[col] == value, gold_labels)
            for col, value in hypotheses(df, covariates, min_size)}


def regress_run(df: pd.DataFrame, label: str, gold: dict[tuple[str, object], dict]) -> list[dict]:
    """`df` has columns GS-LABEL, PRED-LABEL and the covariates. One record per hypothesis
    of `gold`."""
    gold_labels, pred_labels = df["GS-LABEL"] == label, df["PRED-LABEL"] == label
    results = []
    for col, value in gold:
        result = {"regression_col": col, "regression_unique_value": value}
        if not gold[col, value]["success"]:
            # No point in comparing with a failed gold-standard regression
            results.append({**result, "h_t": None})
            continue
        pred = logit(df[col] == value, pred_labels)
        results.append({
            **result,
            "errors": errors(gold[col, value], pred, gold_labels, pred_labels),
            "pred_reg_results": pred,
        })
    return results
