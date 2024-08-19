import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import arviz as az
import pymc as pm

from src.utils import (
    transform_y,
    inverse_transform_y,
    summarize_weights,
    calculate_r2,
)


def create_and_fit_beta_model(train_df, test_df, benchmark):
    # Prepare data
    y = train_df[benchmark].values  # Assuming this is already accuracy data
    X = train_df["FLOPs_1E21"].values

    # Create a mapping for group codes
    unique_groups = train_df["Model_Family"].unique()
    group_mapping = {group: i for i, group in enumerate(unique_groups)}

    # Map the group codes
    groups = np.array([group_mapping[group] for group in train_df["Model_Family"]])
    n_groups = len(unique_groups)

    print(f"Number of unique groups: {n_groups}")
    print(f"Group codes range: {groups.min()} to {groups.max()}")
    print(f"Shape of y: {y.shape}")
    print(f"Shape of X: {X.shape}")

    # Create model
    with pm.Model() as model:
        # Priors
        beta0 = pm.Normal("beta0", mu=0, sigma=5)
        beta1 = pm.Normal("beta1", mu=0, sigma=5)
        sigma_u = pm.HalfCauchy("sigma_u", beta=5)
        sigma_beta_u = pm.HalfCauchy("sigma_beta_u", beta=5)

        # Random effects
        u = pm.Normal("u", mu=0, sigma=sigma_u, shape=n_groups)
        beta_u = pm.Normal("beta_u", mu=0, sigma=sigma_beta_u, shape=n_groups)

        # Expected value of outcome
        mu = pm.math.invlogit(beta0 + beta1 * X + u[groups] + beta_u[groups] * X)

        # Precision parameter
        phi = pm.Gamma("phi", alpha=1, beta=0.1)
        alpha = mu * phi
        beta = (1 - mu) * phi

        # Likelihood (Beta distribution)
        # Add a small epsilon to avoid exact 0 or 1 values
        eps = 1e-6
        y_adj = y * (1 - 2 * eps) + eps
        # y_obs = pm.Beta('y_obs', alpha=mu*phi, beta=(1-mu)*phi, observed=y_adj)
        y_obs = pm.Beta("y_obs", alpha=alpha, beta=beta, observed=y_adj)

        # Fit model
        trace = pm.sample(4000, tune=4000, return_inferencedata=True)

    # Predict for train data
    with model:
        mu_train = pm.math.invlogit(beta0 + beta1 * X + u[groups] + beta_u[groups] * X)
        alpha_train = mu_train * phi
        beta_train = (1 - mu_train) * phi
        # y_pred_train = pm.Beta('y_pred_train', alpha=mu_train*phi, beta=(1-mu_train)*phi)
        y_pred_train = pm.Beta("y_pred_train", alpha=alpha_train, beta=beta_train)
        posterior_pred_train = pm.sample_posterior_predictive(
            trace, var_names=["y_pred_train"]
        )

    # Extract predictions
    y_pred_train_samples = posterior_pred_train.posterior_predictive["y_pred_train"]
    y_pred_train_mean = y_pred_train_samples.mean(dim=("chain", "draw")).values

    # Predict for test data
    with model:
        X_new = test_df["FLOPs_1E21"].values
        groups_new = np.array(
            [group_mapping.get(group, -1) for group in test_df["Model_Family"]]
        )

        # Handle any new groups in test data
        groups_new[groups_new == -1] = n_groups
        if -1 in groups_new:
            u = pm.Normal.dist(mu=0, sigma=sigma_u, shape=n_groups + 1)

        print(f"Test group codes range: {groups_new.min()} to {groups_new.max()}")

        mu_pred = pm.math.invlogit(
            beta0 + beta1 * X_new + u[groups_new] + beta_u[groups_new] * X_new
        )
        alpha_pred = mu_pred * phi
        beta_pred = (1 - mu_pred) * phi
        # y_pred = pm.Beta('y_pred', alpha=mu_pred*phi, beta=(1-mu_pred)*phi)
        y_pred = pm.Beta("y_pred", alpha=alpha_pred, beta=beta_pred)
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y_pred"])

    y_pred_samples = posterior_pred.posterior_predictive["y_pred"]
    y_pred_mean = y_pred_samples.mean(dim=("chain", "draw")).values
    y_pred_hdi = az.hdi(y_pred_samples)

    return trace, y_pred_mean, y_pred_train_mean


def create_and_fit_linear_mixed_effects_model(train_df, test_df):
    # Prepare data
    y = train_df["y_transformed"].values
    X = train_df["FLOPs_1E21"].values
    unique_groups = train_df["Model_Family"].unique()
    group_mapping = {group: i for i, group in enumerate(unique_groups)}

    # Map the group codes
    groups = np.array([group_mapping[group] for group in train_df["Model_Family"]])
    n_groups = len(unique_groups)

    print(f"Number of unique groups: {n_groups}")
    print(f"Group codes range: {groups.min()} to {groups.max()}")
    print(f"Shape of y: {y.shape}")
    print(f"Shape of X: {X.shape}")

    # Create model
    with pm.Model() as model:
        # Priors
        beta0 = pm.Normal("beta0", mu=0, sigma=5)
        beta1 = pm.Normal("beta1", mu=0, sigma=5)
        sigma_e = pm.HalfCauchy("sigma_e", beta=5)
        sigma_u = pm.HalfCauchy("sigma_u", beta=5)
        sigma_beta_u = pm.HalfCauchy("sigma_beta_u", beta=5)

        # Random effects
        u = pm.Normal("u", mu=0, sigma=sigma_u, shape=n_groups)
        beta_u = pm.Normal("beta_u", mu=0, sigma=sigma_beta_u, shape=n_groups)

        # Expected value of outcome
        mu = beta0 + beta1 * X + u[groups] + beta_u[groups] * X

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y)

        # Fit model
        trace = pm.sample(4000, tune=4000, return_inferencedata=True)

    # Predict for train data
    with model:
        # mu_train = beta0 + beta1 * X + u[groups]
        mu_train = beta0 + beta1 * X + u[groups] + beta_u[groups] * X
        y_pred_train = pm.Normal("y_pred_train", mu=mu_train, sigma=sigma_e)
        posterior_pred_train = pm.sample_posterior_predictive(
            trace, var_names=["y_pred_train"]
        )

    y_pred_train_samples = posterior_pred_train.posterior_predictive["y_pred_train"]
    y_pred_train_mean = y_pred_train_samples.mean(dim=("chain", "draw")).values
    y_pred_train_hdi = az.hdi(y_pred_train_samples)

    # Predict
    with model:
        X_new = test_df["FLOPs_1E21"].values
        groups_new = np.array(
            [group_mapping.get(group, -1) for group in test_df["Model_Family"]]
        )

        # Handle any new groups in test data
        groups_new[groups_new == -1] = n_groups
        if -1 in groups_new:
            u = pm.Normal.dist(mu=0, sigma=sigma_u, shape=n_groups + 1)
        # groups_new = pd.Categorical(test_df['Model_Family'], categories=train_df['Model_Family'].unique()).codes
        mu_pred = beta0 + beta1 * X_new + u[groups_new] + beta_u[groups_new] * X_new
        y_pred = pm.Normal("y_pred", mu=mu_pred, sigma=sigma_e)
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y_pred"])

    # Extract predictions
    y_pred_samples = posterior_pred.posterior_predictive["y_pred"]
    y_pred_mean = y_pred_samples.mean(dim=("chain", "draw")).values
    y_pred_hdi = az.hdi(y_pred_samples)

    # Inverse transform predictions
    y_pred_mean = inverse_transform_y(y_pred_mean)
    y_pred_train_mean = inverse_transform_y(y_pred_train_mean)
    # y_pred_lower = inverse_transform_y(y_pred_hdi.sel(hdi='lower').values)
    # y_pred_upper = inverse_transform_y(y_pred_hdi.sel(hdi='higher').values)

    return trace, y_pred_mean, y_pred_train_mean  # , y_pred_lower, y_pred_upper


def pymc_models(train_df, test_df, benchmark, results, predictions):

    y_train = train_df[benchmark]
    y_test = test_df[benchmark]
    # Use the model
    # trace, y_pred_mean, y_pred_lower, y_pred_upper = create_and_fit_pymc_model(train_df, test_df)
    trace, y_pred_mean, y_pred_train_mean = create_and_fit_linear_mixed_effects_model(
        train_df, test_df
    )
    # Analyze results
    az.plot_trace(trace)
    az.summary(trace, var_names=["beta0", "beta1", "sigma_e", "sigma_u"])

    # Calculate metrics
    rmse_pymc = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2_pymc = calculate_r2(y_test, y_pred_mean)

    results["PyMC Model"] = {"rmse": rmse_pymc, "r2": r2_pymc}
    predictions["PyMC Model"] = {
        "train": (y_train, y_pred_train_mean),
        "test": (y_test, y_pred_mean),
    }

    trace, y_pred_mean, y_pred_train_mean = create_and_fit_beta_model(
        train_df, test_df, benchmark
    )
    # Analyze results
    az.plot_trace(trace)
    # az.summary(trace, var_names=["beta0", "beta1", "sigma_e", "sigma_u"])

    # Calculate metrics
    rmse_pymc = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2_pymc = calculate_r2(y_test, y_pred_mean)

    results["PyMC Beta Regression Model"] = {"rmse": rmse_pymc, "r2": r2_pymc}
    predictions["PyMC Beta Regression Model"] = {
        "train": (y_train, y_pred_train_mean),
        "test": (y_test, y_pred_mean),
    }
    return results, predictions


def summarize_pymc_weights(trace, benchmark_cols):
    summary = az.summary(trace, var_names=["beta"])
    weights = pd.DataFrame(
        {
            "variable": benchmark_cols,
            "mean": summary["mean"],
            "sd": summary["sd"],
            "hdi_3%": summary["hdi_3%"],
            "hdi_97%": summary["hdi_97%"],
        }
    )
    return weights


def create_and_fit_glm(train_df, test_df, benchmark_cols):
    X_train = train_df[benchmark_cols].values
    y_train = train_df["y_transformed"].values
    X_test = test_df[benchmark_cols].values

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=X_train.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=5)

        mu = pm.math.dot(X_train, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    # Predict for train data
    with model:
        y_pred_train = pm.Normal("y_pred_train", mu=mu, sigma=sigma)
        posterior_pred_train = pm.sample_posterior_predictive(
            trace, var_names=["y_pred_train"]
        )

    # Predict for test data
    with model:
        mu_test = pm.math.dot(X_test, beta)
        y_pred_test = pm.Normal("y_pred_test", mu=mu_test, sigma=sigma)
        posterior_pred_test = pm.sample_posterior_predictive(
            trace, var_names=["y_pred_test"]
        )

    y_pred_train_mean = (
        posterior_pred_train.posterior_predictive["y_pred_train"]
        .mean(dim=("chain", "draw"))
        .values
    )
    y_pred_test_mean = (
        posterior_pred_test.posterior_predictive["y_pred_test"]
        .mean(dim=("chain", "draw"))
        .values
    )

    return trace, y_pred_train_mean, y_pred_test_mean


def create_and_fit_mixed_effects_glm(train_df, test_df, benchmark_cols, benchmark):
    X_train = train_df[benchmark_cols].values
    y_train = train_df["y_transformed"].values
    X_test = test_df[benchmark_cols].values

    # Identify the index of the scaling factor column
    scaling_factor_col = f"{benchmark}_minus_1_scaling_factor"
    scaling_factor_index = benchmark_cols.index(scaling_factor_col)

    unique_groups = train_df["Model_Family"].unique()
    group_mapping = {group: i for i, group in enumerate(unique_groups)}

    # Map the group codes
    groups_train = np.array(
        [group_mapping[group] for group in train_df["Model_Family"]]
    )
    n_groups = len(unique_groups)
    # groups_train = pd.Categorical(train_df["Model_Family"]).codes
    # n_groups = len(np.unique(groups_train))

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=X_train.shape[1])
        sigma_u = pm.HalfCauchy("sigma_u", beta=5)
        sigma_beta_u = pm.HalfCauchy("sigma_beta_u", beta=5)
        u = pm.Normal("u", mu=0, sigma=sigma_u, shape=n_groups)
        beta_u = pm.Normal("beta_u", mu=0, sigma=sigma_beta_u, shape=n_groups)
        sigma = pm.HalfCauchy("sigma", beta=5)

        mu = (
            pm.math.dot(X_train, beta)
            + u[groups_train]
            + beta_u[groups_train] * X_train[:, scaling_factor_index]
        )
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    # Predict for train data
    with model:
        y_pred_train = pm.Normal("y_pred_train", mu=mu, sigma=sigma)
        posterior_pred_train = pm.sample_posterior_predictive(
            trace, var_names=["y_pred_train"]
        )

    # Predict for test data
    with model:
        groups_test = np.array(
            [group_mapping.get(group, -1) for group in test_df["Model_Family"]]
        )

        # Handle any new groups in test data
        groups_test[groups_test == -1] = n_groups
        if -1 in groups_test:
            u = pm.Normal.dist(mu=0, sigma=sigma_u, shape=n_groups + 1)
        mu_test = (
            pm.math.dot(X_test, beta)
            + u[groups_test]
            + beta_u[groups_test] * X_test[:, scaling_factor_index]
        )
        y_pred_test = pm.Normal("y_pred_test", mu=mu_test, sigma=sigma)
        posterior_pred_test = pm.sample_posterior_predictive(
            trace, var_names=["y_pred_test"]
        )

    y_pred_train_mean = (
        posterior_pred_train.posterior_predictive["y_pred_train"]
        .mean(dim=("chain", "draw"))
        .values
    )
    y_pred_test_mean = (
        posterior_pred_test.posterior_predictive["y_pred_test"]
        .mean(dim=("chain", "draw"))
        .values
    )

    return trace, y_pred_train_mean, y_pred_test_mean


def benchmark_models_pymc(train_df, test_df, benchmark, results, predictions):
    y_train = train_df[benchmark]
    y_test = test_df[benchmark]

    # Prepare data
    # train_df["y_transformed"] = transform_y(y_train)
    # test_df["y_transformed"] = transform_y(y_test)
    # train_df[[f"{benchmark}_minus_1_scaling_factor"]] = np.log(
    #     train_df[[f"{benchmark}_minus_1_scaling_factor"]]
    # )
    # test_df[[f"{benchmark}_minus_1_scaling_factor"]] = np.log(
    #     test_df[[f"{benchmark}_minus_1_scaling_factor"]]
    # )
    benchmark_cols = [col for col in train_df.columns if col.endswith("_minus_1")]
    benchmark_cols = [col for col in benchmark_cols if "PC" in col]
    benchmark_cols.append(f"{benchmark}_minus_1_scaling_factor")
    benchmark_cols.append(f"{benchmark}_minus_1")

    # GLM on all benchmark scores
    glm_trace, y_pred_glm_train, y_pred_glm_test = create_and_fit_glm(
        train_df, test_df, benchmark_cols
    )

    y_pred_glm_train = inverse_transform_y(y_pred_glm_train)
    y_pred_glm_test = inverse_transform_y(y_pred_glm_test)

    rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm_test))
    r2_glm = calculate_r2(y_test, y_pred_glm_test)

    results["PyMC GLM on All Benchmarks"] = {"rmse": rmse_glm, "r2": r2_glm}
    predictions["PyMC GLM on All Benchmarks"] = {
        "train": (y_train, y_pred_glm_train),
        "test": (y_test, y_pred_glm_test),
    }

    glm_weights = summarize_pymc_weights(glm_trace, benchmark_cols)
    print("GLM Weights:", glm_weights)

    # GLM with mixed effects for compute correction
    mixed_trace, y_pred_mixed_train, y_pred_mixed_test = (
        create_and_fit_mixed_effects_glm(train_df, test_df, benchmark_cols, benchmark)
    )

    y_pred_mixed_train = inverse_transform_y(y_pred_mixed_train)
    y_pred_mixed_test = inverse_transform_y(y_pred_mixed_test)

    rmse_mixed = np.sqrt(mean_squared_error(y_test, y_pred_mixed_test))
    r2_mixed = calculate_r2(y_test, y_pred_mixed_test)

    results["PyMC Mixed-Effects GLM"] = {"rmse": rmse_mixed, "r2": r2_mixed}
    predictions["PyMC Mixed-Effects GLM"] = {
        "train": (y_train, y_pred_mixed_train),
        "test": (y_test, y_pred_mixed_test),
    }

    mixed_weights = summarize_pymc_weights(mixed_trace, benchmark_cols)
    print("Mixed-Effects GLM Weights:", mixed_weights)

    # Analyze results
    az.plot_trace(glm_trace)
    az.summary(glm_trace, var_names=["beta", "sigma"])
    az.plot_trace(mixed_trace)
    az.summary(mixed_trace, var_names=["beta", "sigma", "sigma_u"])

    return results, predictions
