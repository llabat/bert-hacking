import os
import json

import pandas as pd 
import numpy as np 
import statsmodels.api as sm

from . import ensure_no_na, CustomLogger

def perform_regression(
    y_column: pd.Series, 
    x_column: pd.Series, 
    optimizer: str = "bfgs"
) -> dict[str:str|float|list[float]]:
    """
    Perform logit regression and returns key values
    """
    Y = y_column.to_numpy().astype(int) # [1, 0, 0, 1, ...]
    X = x_column.to_numpy().astype(int) # [1, 0, 1, 0, ...]
    X = sm.add_constant(X)
    try: 
        model = sm.Logit(Y,X, )
        res = model.fit(maxiter=100, method=optimizer, disp=0)
        if len(model.exog_names) == 1:
            #TODELETE
            logger = CustomLogger()
            logger("x_column.value_counts()", type="REGRESSION")
            logger(x_column.value_counts(), type="REGRESSION")
            logger("y_column.value_counts()", type="REGRESSION")
            logger(y_column.value_counts(), type="REGRESSION")
            
        return {
            "success":True,
            "optimizer": optimizer,
            "Pseudo R-squared": res.prsquared,
            "Covariate Names": model.exog_names,
            "Coef": res.params.tolist(),
            "Std err": res.bse.tolist(),
            "z": res.tvalues.tolist(),          # z-statistics
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
        return {
            "success":False,
            "optimizer": optimizer,
            "Pseudo R-squared": None,
            "Covariate Names": None, 
            "Coef": None,
            "Std err": None,
            "z": None,          # z-statistics
            "pvalues": None,
            "Conf Int": None,
            "Log-Likelihood": None,
            "LL-Null": None,
            "LLR p-value": None,
            "AIC": None,
            "BIC": None,
            "N obs": None,
        } 
    
def assess_errors(
        pred_reg_results: dict, 
        gold_reg_results: dict, 
        pred_labels_binarised : pd.Series,
        gold_labels_binarised: pd.Series,
    ) -> dict:
    """"""
    if not (pred_reg_results["success"] and gold_reg_results["success"]):
        return {}
    
    try: 
        gold_index_x1 = gold_reg_results["Covariate Names"].index("x1")
        pred_index_x1 = pred_reg_results["Covariate Names"].index("x1")
    except:
        #TODELETE
        logger = CustomLogger()
        logger("gold: ", gold_reg_results["Covariate Names"], type="REGRESSION")
        logger("pred: ", pred_reg_results["Covariate Names"], type="REGRESSION")
        return {"error": "can't find x1"}
    if gold_index_x1 not in [0,1]: raise ValueError(f"Issue with gold_index_x1, found {gold_index_x1}, should be either 0 or 1")
    if pred_index_x1 not in [0,1]: raise ValueError(f"Issue with pred_index_x1, found {pred_index_x1}, should be either 0 or 1")

    gold_p_value = gold_reg_results["pvalues"][gold_index_x1]
    pred_p_value = pred_reg_results["pvalues"][pred_index_x1]
    gold_coef = gold_reg_results["Coef"][gold_index_x1]
    pred_coef = pred_reg_results["Coef"][pred_index_x1]
    gold_delta = abs(
        gold_labels_binarised.astype(bool).mean(skipna=False) 
        - 
        (~gold_labels_binarised).astype(bool).mean(skipna=False)
    )
    pred_delta = abs(
        pred_labels_binarised.astype(bool).mean(skipna=False) 
        - 
        (~pred_labels_binarised).astype(bool).mean(skipna=False)
    )

    gold_significant = gold_p_value < 0.05
    pred_significant = pred_p_value < 0.05

    error_M_value = float(
        int(gold_significant and pred_significant and (gold_coef * pred_coef >= 0))
        *
        abs( -1 + pred_delta / gold_delta)
    )

    return {
        "h_t" : (gold_significant 
            if not np.isnan([gold_p_value]).any() 
            else None),
        "error_type_1": (
            pred_significant and not gold_significant
            if not np.isnan([gold_p_value, pred_p_value]).any() 
            else None),
        "error_type_2": (gold_significant and not pred_significant
            if not np.isnan([gold_p_value, pred_p_value]).any() 
            else None),
        "error_type_S": (gold_significant and pred_significant and (gold_coef * pred_coef < 0)
            if not np.isnan([gold_p_value, pred_p_value, gold_coef, pred_coef]).any() 
            else None),
        "error_type_M": (error_M_value
            if not np.isnan([gold_p_value, pred_p_value, gold_coef, pred_coef, pred_delta, gold_delta]).any() 
            else None),
    }

def run_regression_and_assess_errors(
    df_regression: pd.DataFrame, 
    run_info: dict, 
    regression_col :str, 
    regression_unique_value:str,
) -> dict:
    pred_labels_binarised = df_regression["PRED-LABEL"] == run_info["dichotomization_label"]
    gold_labels_binarised = df_regression["GS-LABEL"] == run_info["dichotomization_label"]

    gold_reg_results_cache_file = (
        f"{run_info['dataset_name']}-{run_info['dichotomization_label']}-"
        f"{regression_col}-{regression_unique_value}.json"
    )

    if gold_reg_results_cache_file in os.listdir("./.cache"):
        with open(f"./.cache/{gold_reg_results_cache_file}") as file:
            gold_reg_results = json.load(file)
    else: 
        gold_reg_results = perform_regression(
            y_column = df_regression[regression_col] == regression_unique_value, 
            x_column = gold_labels_binarised
        )
        with open(f"./.cache/{gold_reg_results_cache_file}", "w") as file:
                json.dump(gold_reg_results, file, ensure_ascii=True)
    
    if not gold_reg_results["success"]: 
        # No point in computing the regression if the gold standard one failed
        return {"h_t": None}

    pred_reg_results = perform_regression(
        y_column = df_regression[regression_col] == regression_unique_value, 
        x_column = pred_labels_binarised
    )
    
    errors =  assess_errors(
        pred_reg_results, 
        gold_reg_results,
        pred_labels_binarised,
        gold_labels_binarised
    )
    return {
        "errors" : errors, 
        "pred_reg_results":ensure_no_na(pred_reg_results)
    }