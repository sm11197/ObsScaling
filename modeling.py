import pandas as pd
import re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf


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


def compare_models(train_df, test_df, benchmark):
    X = train_df[["FLOPs (1E21)"]]
    y = train_df[benchmark]
    X_test = test_df[["FLOPs (1E21)"]]
    y_test = test_df[benchmark]

    # Logistic Regression (as an alternative to Beta Regression)
    # log_reg = LogisticRegression()
    # log_reg.fit(X, y)
    # y_pred_log = log_reg.predict(X)
    # rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))

    def clean_column_name(name):
        return re.sub(r'\W+', '_', name).strip('_')
    
    models = {}
    for model_type in ["linear", "sigmoid"]:
        fit_func, func_form = fit_model(X, y, model_type)
        y_pred = fit_func(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # models[model_type] = {"fit_func": fit_func, "func_form": func_form, "rmse": rmse}
        models[f"FLOPS only {model_type}"] = rmse

    # Linear Mixed-Effects Model
    train_df.columns = [clean_column_name(col) for col in train_df.columns]
    test_df.columns = [clean_column_name(col) for col in test_df.columns]
    flops_column = clean_column_name('FLOPs (1E21)')
    mixed_model = smf.mixedlm(
        f"{benchmark} ~ {flops_column}", data=train_df, groups=train_df["Model_Family"]
    )
    mixed_model_fit = mixed_model.fit()
    y_pred_mixed = mixed_model_fit.predict(test_df)
    rmse_mixed = np.sqrt(mean_squared_error(y_test, y_pred_mixed))

    # GLM on all benchmark scores
    formula = f"{benchmark} ~ " + " + ".join(
        [col for col in train_df.columns if col.endswith("_minus_1")]
    )
    glm_model = smf.glm(formula=formula, data=train_df, family=sm.families.Gaussian())
    glm_results = glm_model.fit()
    y_pred_glm = glm_results.predict(test_df)
    rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm))

    y_transformed = transform_y(y)
    # Linear Mixed-Effects Model
    train_df["y_transformed"] = y_transformed
    mixed_model = smf.mixedlm(
        f"y_transformed ~ {flops_column}", data=train_df, groups=train_df["Model_Family"]
    )
    mixed_model_fit = mixed_model.fit()
    y_pred_mixed = mixed_model_fit.predict(test_df)
    y_pred_mixed_transformed = inverse_transform_y(y_pred_mixed)
    rmse_mixed_transformed = np.sqrt(mean_squared_error(y_test, y_pred_mixed_transformed))

    # GLM on all benchmark scores
    benchmark_cols = [col for col in train_df.columns if col.endswith("_minus_1")]
    formula = "y_transformed ~ " + " + ".join(benchmark_cols)
    glm_model = smf.glm(formula=formula, data=train_df, family=sm.families.Gaussian())
    glm_results = glm_model.fit()
    y_pred_glm = glm_results.predict(test_df)
    y_pred_glm_transformed = inverse_transform_y(y_pred_glm)
    rmse_glm_transformed = np.sqrt(mean_squared_error(y_test, y_pred_glm_transformed))

    return {
        **models,
        "Mixed-Effects Model RMSE": rmse_mixed,
        "Mixed-Effects Model RMSE with inverse sigmoid transform": rmse_mixed_transformed,
        "GLM on All Benchmarks RMSE": rmse_glm,
        "GLM on All Benchmarks RMSE with inverse sigmoid transform": rmse_glm_transformed,
    }


def split_train_test(data):
    # Find the largest model by FLOPs for each family
    data = data.reset_index(drop=True)

    # Find the largest model by FLOPs for each family, ignoring empty groups
    largest_models = data.loc[
        data.groupby("Model Family", observed=True)["FLOPs (1E21)"].idxmax()
    ]

    # Create test set with the largest models
    print(data)
    # largest_models = data.groupby("Model Family")
    print(largest_models)
    # Create test set with the largest models
    test_df = largest_models.copy()

    # Create train set with all other models
    train_df = data[~data.index.isin(test_df.index)].copy()

    return train_df, test_df


def consolidate_model_family(family):
    if "OPENLLAMA" in family.upper():
        return "OPENLLAMA"
    elif "CODELLAMA" in family.upper():
        return "CODELLAMA"
    elif "LLAMA" in family:
        return "LLAMA"
    elif "QWEN" in family:
        return "QWEN"
    else:
        return family


def main():
    # Data Processing
    # Read in the data
    base_llm_benchmark_eval = pd.read_csv(
        "eval_results/base_llm_benchmark_pca_imputed.csv"
    )
    base_llm_emergent_eval = pd.read_csv(
        "eval_results/base_llm_emergent_capability_eval.csv"
    )

    # Split first column by /
    base_llm_benchmark_eval[["Repo", "Model"]] = base_llm_benchmark_eval[
        "Model"
    ].str.split("/", expand=True)
    base_llm_emergent_eval[["Repo", "Model"]] = base_llm_emergent_eval[
        "Model"
    ].str.split("/", expand=True)

    # Convert model & model family names to uppercase
    print(base_llm_benchmark_eval)
    base_llm_benchmark_eval["Model"] = base_llm_benchmark_eval["Model"].str.upper()
    base_llm_benchmark_eval["Model Family"] = base_llm_benchmark_eval[
        "Model Family"
    ].str.upper()
    base_llm_emergent_eval["Model"] = base_llm_emergent_eval["Model"].str.upper()

    emergent_benchmarks = [
        col for col in base_llm_emergent_eval.columns if col not in ["Model", "Repo"]
    ]

    # Consolidate LLAMA and QWEN model families

    base_llm_benchmark_eval["Model Family"] = base_llm_benchmark_eval[
        "Model Family"
    ].apply(consolidate_model_family)

    # Merge datasets by Model, all. Suffix "benchmark" and "emergent"
    base_llm = pd.merge(
        base_llm_benchmark_eval,
        base_llm_emergent_eval,
        on="Model",
        how="outer",
        suffixes=(".benchmark", ".emergent"),
    )

    # Convert character columns to categorical
    for col in base_llm.select_dtypes(include=["object"]).columns:
        base_llm[col] = base_llm[col].astype("category")

    # Count the number of models in each family
    family_counts = base_llm["Model Family"].value_counts()
    print("Model families:")
    print(family_counts)

    # Identify families with at least 2 models
    valid_families = family_counts[family_counts > 2].index
    print("Model families with more than 2 models:")
    print(valid_families)

    # Filter the dataset to keep only the valid families
    base_llm = base_llm[base_llm["Model Family"].isin(valid_families)]

    # Re-factor Model.Family to remove unused levels
    base_llm["Model Family"] = pd.Categorical(base_llm["Model Family"])

    # Remove ipa_transliterate_2_bleu column if it exists
    if "ipa_transliterate_2_bleu" in base_llm.columns:
        base_llm = base_llm.drop("ipa_transliterate_2_bleu", axis=1)

    # Print summary stats
    print(base_llm.describe())

    # Print the number of models remaining
    print(f"Number of models remaining: {len(base_llm)}")

    # Print the remaining model families and their counts
    remaining_family_counts = base_llm["Model Family"].value_counts()
    print("Remaining model families and their counts:")
    print(remaining_family_counts)

    # Get n-1 score
    def get_minus_1_normalized(model_index, data, benchmark_column):
        current_family = data.iloc[model_index]["Model Family"]
        current_flops = data.iloc[model_index]["FLOPs (1E21)"]
        current_score = data.iloc[model_index][benchmark_column]

        smaller_models = data[data["FLOPs (1E21)"] < current_flops].sort_values(
            "FLOPs (1E21)", ascending=False
        )

        closest_within_family = smaller_models[
            smaller_models["Model Family"] == current_family
        ].head(1)
        closest_global = smaller_models.head(1)

        if not closest_within_family.empty:
            target_model = closest_within_family
        elif not closest_global.empty:
            target_model = closest_global
        else:
            target_model = pd.DataFrame(
                {"FLOPs (1E21)": [current_flops], benchmark_column: [current_score]}
            )

        minus_1_flops = target_model["FLOPs (1E21)"].values[0]
        minus_1_score = target_model[benchmark_column].values[0]

        return minus_1_score * (np.log(current_flops) / np.log(minus_1_flops))

    # Define non-benchmark columns
    non_benchmark_columns = [
        "Model",
        "Model Family",
        "Model Size (B)",
        "Pretraining Data Size (T)",
        "FLOPs (1E21)",
        "Repo.benchmark",
        "Repo.emergent",
    ]

    # Identify benchmark columns by exclusion
    benchmark_columns = [
        col for col in base_llm.columns if col not in non_benchmark_columns
    ]

    # Apply the 'get_minus_1_normalized' function across all benchmarks
    for benchmark in benchmark_columns:
        new_column_name = f"{benchmark}_minus_1"
        base_llm[new_column_name] = [
            get_minus_1_normalized(i, base_llm, benchmark) for i in range(len(base_llm))
        ]

    # Modeling (basic structure)
    def pca_impute(train_df, response_vars, test_df=None, n_components=2):
        # Implement PCA imputation logic here
        pass

    def fit_model(formula, data):
        # Implement model fitting logic here
        pass

    def calculate_model_metrics(model, test_data, response_formula):
        # Implement model metrics calculation here
        pass

    # Define cutoff for FLOPs to split the data
    cutoff_flops = 8.4 * 10

    train_df, test_df = split_train_test(base_llm)
    print(test_df)

    # Split data into training and testing sets
    # train_df = base_llm_clean[base_llm_clean["FLOPs (1E21)"] <= cutoff_flops]
    # test_df = base_llm_clean[base_llm_clean["FLOPs (1E21)"] > cutoff_flops]

    # Prepare for modeling
    performance_metrics = pd.DataFrame(columns=["benchmark", "model_type", "rmse"])

    for benchmark in emergent_benchmarks:
        benchmark_minus_1 = f"{benchmark}_minus_1"
        train_df_benchmark = train_df.dropna(
            subset=[benchmark, "FLOPs (1E21)", "Model Family", benchmark_minus_1]
        )
        test_df_benchmark = test_df.dropna(
            subset=[benchmark, "FLOPs (1E21)", "Model Family", benchmark_minus_1]
        )

        results = compare_models(train_df_benchmark, test_df_benchmark, benchmark)
        print(f"Results for {benchmark}:")
        for key, value in results.items():
            print(f"{key}: {value.item()}")

        # Add results to performance_metrics DataFrame
        # for model_type, rmse in results.items():
        #     performance_metrics = performance_metrics.append(
        #         {"benchmark": benchmark, "model_type": model_type, "rmse": rmse},
        #         ignore_index=True,
            # )

    # Output results
    print("\nOverall Performance Metrics:")
    print(performance_metrics)

    # Optional: Save results to CSV
    performance_metrics.to_csv("regression_comparison_results.csv", index=False)
    # Remove NAs

    # base_llm_clean = base_llm.dropna(
    #     subset=[response_formula, "FLOPs (1E21)", "Model Family", benchmark_minus_1]
    # )

    # Perform modeling steps here (not implemented in detail)

    # Output results
    print(performance_metrics)


if __name__ == "__main__":
    main()
