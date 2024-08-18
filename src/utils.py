
import pandas as pd
import re
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import statsmodels.api as sm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def transform_y(y):
    # Clip values to avoid log(0) or log(1)
    y_clipped = np.clip(y, 0.001, 0.999)
    return inverse_sigmoid(y_clipped)


def inverse_transform_y(y):
    return sigmoid(y)


def sigmoid_transformed_x(x, *p):
    p = np.array(p)
    return sigmoid(np.dot(x, p))


def format_linear_func_form(weights, metric_names, bias=None, eps=5e-3):
    terms = []
    for weight, name in zip(weights, metric_names):
        if np.abs(weight) <= eps:
            continue
        sign = "+" if weight > 0 else "-"
        abs_weight = abs(weight)
        if np.abs(abs_weight - 1.0) <= eps:
            term = f"{sign} {name}"
        else:
            term = f"{sign} {abs_weight:.2f}{name}"
        terms.append(term.strip())

    if terms:
        terms[0] = terms[0].lstrip("+").strip()
        expression = " ".join(terms)
    else:
        expression = ""

    if bias is not None and np.abs(bias) > eps:
        if bias > 0:
            expression += f" + {bias:.2f}"
        elif bias < 0:
            expression += f" - {-bias:.2f}"

    return expression


def fit_model(X, y, model_type):
    X = sm.add_constant(X)

    if model_type == "linear":
        model = sm.OLS(y, X).fit()
        fit_func = lambda x: model.predict(sm.add_constant(x))
        weights = model.params.values[1:]
        bias = model.params.values[0]
        func_form = format_linear_func_form(weights, X.columns[1:], bias)
    elif model_type == "sigmoid":
        p0 = np.ones(X.shape[1]) * 1e-2
        popt, _ = curve_fit(
            sigmoid_transformed_x, X.values, y.values, p0=p0, maxfev=10000
        )
        fit_func = lambda x: sigmoid_transformed_x(sm.add_constant(x), *popt)
        weights = popt[1:]
        bias = popt[0]
        linear_form = format_linear_func_form(weights, X.columns[1:], bias)
        func_form = f"sigmoid({linear_form})"

    return fit_func, func_form


def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def clean_column_name(name):
    return re.sub(r"\W+", "_", name).strip("_")


def summarize_weights(model):
    if isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        return pd.DataFrame(
            {"coef": model.params, "std_err": model.bse, "p_value": model.pvalues}
        )
    elif isinstance(model, sm.regression.mixed_linear_model.MixedLMResultsWrapper):
        fixed_effects = pd.DataFrame(
            {
                "coef": model.fe_params,
                "std_err": model.bse_fe,
                # "p_value": model.pvalues_fe,
            }
        )
        random_effects = []
        for group, effects in model.random_effects.items():
            re_df = pd.DataFrame(effects, columns=["coef"])
            re_df["group"] = group
            re_df["variable"] = re_df.index
            random_effects.append(re_df)

        random_effects_df = pd.concat(random_effects, ignore_index=True)
        random_effects_df["std_err"] = np.nan
        random_effects_df["p_value"] = np.nan

        return pd.concat(
            [
                fixed_effects.reset_index().rename(columns={"index": "variable"}),
                random_effects_df,
            ],
            keys=["Fixed Effects", "Random Effects"],
        )
    else:
        return pd.DataFrame()
