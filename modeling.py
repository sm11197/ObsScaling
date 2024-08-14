import pandas as pd
import argparse
import re
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


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


def compare_models(train_df, test_df, benchmark):

    train_df.columns = [clean_column_name(col) for col in train_df.columns]
    test_df.columns = [clean_column_name(col) for col in test_df.columns]
    # transform flops into log scale
    train_df[["FLOPs_1E21"]] = np.log(train_df[["FLOPs_1E21"]])
    test_df[["FLOPs_1E21"]] = np.log(test_df[["FLOPs_1E21"]])
    X_train = train_df[["FLOPs_1E21"]]
    y_train = train_df[benchmark]
    X_test = test_df[["FLOPs_1E21"]]
    y_test = test_df[benchmark]

    results = {}
    predictions = {}

    for model_type in ["linear", "sigmoid"]:
        fit_func, func_form = fit_model(X_train, y_train, model_type)
        y_pred_train = fit_func(X_train)
        y_pred_test = fit_func(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = calculate_r2(y_test, y_pred_test)
        results[f"FLOPS only {model_type}"] = {"rmse": rmse, "r2": r2}
        predictions[f"FLOPS only {model_type}"] = {
            "train": (y_train, y_pred_train),
            "test": (y_test, y_pred_test),
        }

    # Linear Mixed-Effects Model
    y_transformed = transform_y(y_train)
    train_df["y_transformed"] = y_transformed
    mixed_model = smf.mixedlm(
        f"y_transformed ~ FLOPs_1E21", data=train_df, groups=train_df["Model_Family"]
    )
    mixed_model_fit = mixed_model.fit()
    y_pred_mixed_train = inverse_transform_y(mixed_model_fit.predict(train_df))
    y_pred_mixed_test = inverse_transform_y(mixed_model_fit.predict(test_df))
    rmse_mixed = np.sqrt(mean_squared_error(y_test, y_pred_mixed_test))
    r2_mixed = calculate_r2(y_test, y_pred_mixed_test)
    results["Mixed-Effects Model"] = {"rmse": rmse_mixed, "r2": r2_mixed}
    predictions["Mixed-Effects Model"] = {
        "train": (y_train, y_pred_mixed_train),
        "test": (y_test, y_pred_mixed_test),
    }

    # GLM on all benchmark scores
    train_df[[f"{benchmark}_minus_1_scaling_factor"]] = np.log(
        train_df[[f"{benchmark}_minus_1_scaling_factor"]]
    )
    test_df[[f"{benchmark}_minus_1_scaling_factor"]] = np.log(
        test_df[[f"{benchmark}_minus_1_scaling_factor"]]
    )
    benchmark_cols = [col for col in train_df.columns if col.endswith("_minus_1")]
    benchmark_cols = [col for col in benchmark_cols if "PC" in col]
    benchmark_cols.append(f"{benchmark}_minus_1_scaling_factor")
    benchmark_cols.append(f"{benchmark}_minus_1")
    formula = f"y_transformed ~ " + " + ".join(benchmark_cols)
    glm_model = smf.glm(formula=formula, data=train_df, family=sm.families.Gaussian())
    glm_results = glm_model.fit()

    glm_weights = summarize_weights(glm_results)
    print(glm_weights)
    y_pred_glm_train = inverse_transform_y(glm_results.predict(train_df))
    y_pred_glm_test = inverse_transform_y(glm_results.predict(test_df))
    rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm_test))
    r2_glm = calculate_r2(y_test, y_pred_glm_test)
    results["GLM on All Benchmarks"] = {"rmse": rmse_glm, "r2": r2_glm}
    predictions["GLM on All Benchmarks"] = {
        "train": (y_train, y_pred_glm_train),
        "test": (y_test, y_pred_glm_test),
    }

    # GLM with mixed effects for compute correction
    fixed_effects = " + ".join(benchmark_cols)

    formula = f"y_transformed ~ {fixed_effects}"

    combined_model = smf.mixedlm(
        formula,
        data=train_df,
        groups=train_df["Model_Family"],
        re_formula=f"~{benchmark}_minus_1_scaling_factor",
    )
    combined_model_fit = combined_model.fit()
    combined_weights = summarize_weights(combined_model_fit)
    print(combined_weights)
    y_pred_glm_train = inverse_transform_y(combined_model_fit.predict(train_df))
    y_pred_glm_test = inverse_transform_y(combined_model_fit.predict(test_df))
    rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm_test))
    r2_glm = calculate_r2(y_test, y_pred_glm_test)
    results["GLM on All Benchmarks with compute correction"] = {
        "rmse": rmse_glm,
        "r2": r2_glm,
    }
    predictions["GLM on All Benchmarks with compute correction"] = {
        "train": (y_train, y_pred_glm_train),
        "test": (y_test, y_pred_glm_test),
    }

    return results, predictions


def plot_predictions(results, predictions, benchmark, save_path):
    plt.figure(figsize=(20, 15))
    for i, (model_name, pred_data) in enumerate(predictions.items()):
        plt.subplot(2, 3, i + 1)

        # Plot training data
        plt.scatter(
            pred_data["train"][0],
            pred_data["train"][1],
            color="blue",
            alpha=0.5,
            label="Train",
        )

        # Plot test data
        plt.scatter(
            pred_data["test"][0],
            pred_data["test"][1],
            color="red",
            alpha=0.5,
            label="Test",
        )

        # Plot the perfect prediction line
        min_val = min(pred_data["train"][0].min(), pred_data["test"][0].min())
        max_val = max(pred_data["train"][0].max(), pred_data["test"][0].max())
        plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f'{model_name}\nR² = {results[model_name]["r2"]:.4f}')
        plt.legend()

    plt.tight_layout()
    plt.suptitle(f"Predicted vs Actual Values for {benchmark}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path)
    plt.close()


# def compare_models(train_df, test_df, benchmark):
#     X = train_df[["FLOPs (1E21)"]]
#     y = train_df[benchmark]
#     X_test = test_df[["FLOPs (1E21)"]]
#     y_test = test_df[benchmark]
#
#     # Logistic Regression (as an alternative to Beta Regression)
#     # log_reg = LogisticRegression()
#     # log_reg.fit(X, y)
#     # y_pred_log = log_reg.predict(X)
#     # rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
#
#     def clean_column_name(name):
#         return re.sub(r'\W+', '_', name).strip('_')
#
#     models = {}
#     for model_type in ["linear", "sigmoid"]:
#         fit_func, func_form = fit_model(X, y, model_type)
#         y_pred = fit_func(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         # models[model_type] = {"fit_func": fit_func, "func_form": func_form, "rmse": rmse}
#         models[f"FLOPS only {model_type}"] = rmse
#
#     # Linear Mixed-Effects Model
#     train_df.columns = [clean_column_name(col) for col in train_df.columns]
#     test_df.columns = [clean_column_name(col) for col in test_df.columns]
#     flops_column = clean_column_name('FLOPs (1E21)')
#     mixed_model = smf.mixedlm(
#         f"{benchmark} ~ {flops_column}", data=train_df, groups=train_df["Model_Family"]
#     )
#     mixed_model_fit = mixed_model.fit()
#     y_pred_mixed = mixed_model_fit.predict(test_df)
#     rmse_mixed = np.sqrt(mean_squared_error(y_test, y_pred_mixed))
#
#     # GLM on all benchmark scores
#     formula = f"{benchmark} ~ " + " + ".join(
#         [col for col in train_df.columns if col.endswith("_minus_1")]
#     )
#     glm_model = smf.glm(formula=formula, data=train_df, family=sm.families.Gaussian())
#     glm_results = glm_model.fit()
#     y_pred_glm = glm_results.predict(test_df)
#     rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm))
#
#     y_transformed = transform_y(y)
#     # Linear Mixed-Effects Model
#     train_df["y_transformed"] = y_transformed
#     mixed_model = smf.mixedlm(
#         f"y_transformed ~ {flops_column}", data=train_df, groups=train_df["Model_Family"]
#     )
#     mixed_model_fit = mixed_model.fit()
#     y_pred_mixed = mixed_model_fit.predict(test_df)
#     y_pred_mixed_transformed = inverse_transform_y(y_pred_mixed)
#     rmse_mixed_transformed = np.sqrt(mean_squared_error(y_test, y_pred_mixed_transformed))
#
#     # GLM on all benchmark scores
#     benchmark_cols = [col for col in train_df.columns if col.endswith("_minus_1")]
#     formula = "y_transformed ~ " + " + ".join(benchmark_cols)
#     glm_model = smf.glm(formula=formula, data=train_df, family=sm.families.Gaussian())
#     glm_results = glm_model.fit()
#     y_pred_glm = glm_results.predict(test_df)
#     y_pred_glm_transformed = inverse_transform_y(y_pred_glm)
#     rmse_glm_transformed = np.sqrt(mean_squared_error(y_test, y_pred_glm_transformed))
#
#     return {
#         **models,
#         "Mixed-Effects Model RMSE": rmse_mixed,
#         "Mixed-Effects Model RMSE with inverse sigmoid transform": rmse_mixed_transformed,
#         "GLM on All Benchmarks RMSE": rmse_glm,
#         "GLM on All Benchmarks RMSE with inverse sigmoid transform": rmse_glm_transformed,
#     }


def split_train_test(data, flops_cutoff=None):
    # Find the largest model by FLOPs for each family
    data = data.reset_index(drop=True)
    if flops_cutoff:
        # train is all models with flops less than the cutoff
        train_df = data[data["FLOPs (1E21)"] <= flops_cutoff].copy()
        # test is all models with flops greater than the cutoff
        test_df = data[data["FLOPs (1E21)"] > flops_cutoff].copy()
    else:
        # Find the largest model by FLOPs for each family, ignoring empty groups
        largest_models = data.loc[
            data.groupby("Model Family", observed=True)["FLOPs (1E21)"].idxmax()
        ]

        # Create test set with the largest models
        # print(data)
        # largest_models = data.groupby("Model Family")
        # print(largest_models)
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


def get_minus_1_normalized(model, data, benchmark_column, flops_cutoff=None):
    if isinstance(model, (pd.Series, dict)):
        current_family = model["Model Family"]
        current_flops = model["FLOPs (1E21)"]
        current_score = model[benchmark_column]
    else:
        raise ValueError("model must be a pandas Series or dict")

    if flops_cutoff:
        largest_flops = min(current_flops, flops_cutoff)
    else:
        largest_flops = current_flops

    smaller_models = data[data["FLOPs (1E21)"] < largest_flops].sort_values(
        "FLOPs (1E21)", ascending=False
    )

    closest_within_family = smaller_models[
        smaller_models["Model Family"] == current_family
    ].head(1)

    if not closest_within_family.empty:
        target_model = closest_within_family
    elif flops_cutoff:
        if current_flops < flops_cutoff:
            target_model = pd.DataFrame(
                {"FLOPs (1E21)": [current_flops], benchmark_column: [current_score]}
            )
        else:
            print(f"Model not found for {model['Model']}")
            print(f"Current flops: {current_flops}, largest flops: {largest_flops}")
            print(model)
            target_model = smaller_models.head(1)
    else:
        target_model = pd.DataFrame(
                {"FLOPs (1E21)": [current_flops], benchmark_column: [current_score]}
            )
    
    minus_1_flops = target_model["FLOPs (1E21)"].values[0]
    minus_1_score = target_model[benchmark_column].values[0]
    scaling_factor = current_flops / minus_1_flops
    return minus_1_score, scaling_factor


def augment_dataset(train_df, benchmark_columns, flops_cutoff):
    augmented_data = []

    for _, row in train_df.iterrows():
        current_family = row["Model Family"]
        current_flops = row["FLOPs (1E21)"]

        # Get all models in the same family with lower FLOPS
        if flops_cutoff:
            largest_flops = min(current_flops, flops_cutoff)
        else:
            largest_flops = current_flops

        smaller_models = train_df[train_df["FLOPs (1E21)"] < largest_flops].sort_values(
            "FLOPs (1E21)", ascending=False
        )

        closest_within_family = smaller_models[
            smaller_models["Model Family"] == current_family
        ]
        # remove row with the largest flops
        closest_within_family = closest_within_family[
            closest_within_family["FLOPs (1E21)"] != closest_within_family["FLOPs (1E21)"].max()
        ]
        for _, target_model in closest_within_family.iterrows():
            for benchmark in benchmark_columns:
                minus_1_score = target_model[benchmark]
                minus_1_flops = target_model["FLOPs (1E21)"]
                scaling_factor = current_flops / minus_1_flops
                augmented_row = row.copy()
                augmented_row[f"{benchmark}_minus_1"] = minus_1_score
                augmented_row[f"{benchmark}_minus_1_scaling_factor"] = scaling_factor
            augmented_data.append(augmented_row)

    return pd.DataFrame(augmented_data)


def main():
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment the dataset with additional data points",
    )
    parser.add_argument(
        "--flops_cutoff",
        default=84.0,
        type=float,
        help="FLOPs cutoff for splitting the data",
    )
    args = parser.parse_args()
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
    # print(base_llm_benchmark_eval)
    base_llm_benchmark_eval["Model"] = base_llm_benchmark_eval["Model"].str.upper()
    base_llm_benchmark_eval["Model Family"] = base_llm_benchmark_eval[
        "Model Family"
    ].str.upper()
    base_llm_emergent_eval["Model"] = base_llm_emergent_eval["Model"].str.upper()

    emergent_benchmarks = [
        col
        for col in base_llm_emergent_eval.columns
        if col not in ["Model", "Repo"] and "bleu" not in col
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
    # print(base_llm.describe())

    # Print the number of models remaining
    print(f"Number of models remaining: {len(base_llm)}")

    # Print the remaining model families and their counts
    remaining_family_counts = base_llm["Model Family"].value_counts()
    print("Remaining model families and their counts:")
    print(remaining_family_counts)

    # Get n-1 score
    # return minus_1_score * (np.log(current_flops) / np.log(minus_1_flops))

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

    # Define cutoff for FLOPs to split the data
    flops_cutoff = args.flops_cutoff
    # Apply the 'get_minus_1_normalized' function across all benchmarks
    for benchmark in benchmark_columns:
        new_column_name = f"{benchmark}_minus_1"
        minus_1_factors = [
            get_minus_1_normalized(base_llm.iloc[i], base_llm, benchmark, flops_cutoff)
            for i in range(len(base_llm))
        ]
        # minus_1_factors = [
        #     get_minus_1_normalized(i, base_llm, benchmark, flops_cutoff)
        #     for i in range(len(base_llm))
        # ]
        base_llm[new_column_name] = [factor[0] for factor in minus_1_factors]
        base_llm[f"{new_column_name}_scaling_factor"] = [
            factor[1] for factor in minus_1_factors
        ]

    train_df, test_df = split_train_test(base_llm, flops_cutoff)
    print(test_df)
    # Augment the dataset
    if args.augment:
        augmented_df = augment_dataset(train_df, benchmark_columns, flops_cutoff)

        # Combine train_df and augmented_df, ensuring augmented data is not more than 50% of total
        total_train_size = len(train_df)
        max_augmented_size = total_train_size  # This ensures 50% split

        if len(augmented_df) > max_augmented_size:
            augmented_df = augmented_df.sample(n=max_augmented_size, random_state=42)

        combined_train_df = pd.concat([train_df, augmented_df], ignore_index=True)

        print("Original train set size:", len(train_df))
        print("Augmented set size:", len(augmented_df))
        print("Combined train set size:", len(combined_train_df))
    else:
        combined_train_df = train_df

    # Split data into training and testing sets
    # train_df = base_llm_clean[base_llm_clean["FLOPs (1E21)"] <= cutoff_flops]
    # test_df = base_llm_clean[base_llm_clean["FLOPs (1E21)"] > cutoff_flops]

    # Prepare for modeling
    all_performance_metrics = []

    for benchmark in emergent_benchmarks:
        benchmark_minus_1 = f"{benchmark}_minus_1"
        combined_train_df_benchmark = combined_train_df.dropna(
            subset=[benchmark, "FLOPs (1E21)", "Model Family", benchmark_minus_1]
        )
        test_df_benchmark = test_df.dropna(
            subset=[benchmark, "FLOPs (1E21)", "Model Family", benchmark_minus_1]
        )

        results, predictions = compare_models(
            combined_train_df_benchmark, test_df_benchmark, benchmark
        )
        print(f"Results for {benchmark}:")
        for model_type, metrics in results.items():
            print(
                f"{model_type}: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}"
            )

        # Add results to performance_metrics DataFrame
        for model_type, metrics in results.items():
            all_performance_metrics.append(
                pd.DataFrame(
                    {
                        "benchmark": benchmark,
                        "model_type": model_type,
                        "rmse": metrics["rmse"],
                    },
                    index=[0],
                )
                # {"benchmark": benchmark, "model_type": model_type, "rmse": rmse},
            )
        plot_predictions(
            results, predictions, benchmark, f"scatter_plot_{benchmark}.png"
        )

    # Output results
    performance_metrics = pd.concat(all_performance_metrics, ignore_index=True)
    print("\nOverall Performance Metrics:")
    rmse_summary = performance_metrics.groupby("model_type").agg(
        {"rmse": ["mean", "median"]}
    )
    rmse_summary.columns = ["mean_rmse", "median_rmse"]
    rmse_summary = rmse_summary.sort_values("mean_rmse")

    print("\nRMSE Summary Across All Benchmarks:")
    print(rmse_summary)

    # Optional: Save results to CSV
    performance_metrics.to_csv("regression_comparison_results.csv", index=False)
    rmse_summary.to_csv("rmse_summary.csv")

    # Optionally, you can also create a bar plot to visualize the RMSE summary

    plt.figure(figsize=(12, 6))
    rmse_summary.plot(
        kind="bar", ylabel="RMSE", title="Mean and Median RMSE Across All Benchmarks"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("rmse_summary.png")
    plt.close()
    # Optional: Save results to CSV
    performance_metrics.to_csv("regression_comparison_results.csv", index=False)
    # Remove NAs


if __name__ == "__main__":
    main()
