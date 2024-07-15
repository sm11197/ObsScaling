# args[1] <- "~/git/predicting-capabilities/ObsScaling/" # nolint: commented_code_linter, line_length_linter.
main <- function() { # nolint: function_name_linter
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  args <- commandArgs(trailingOnly = TRUE)
  #### Libraries ####
  ## Install missing packages, and devtools if you haven't already
  install_and_load <- function(pkg) {
    if (!require(pkg, character.only = TRUE)) {
      ## Set a CRAN mirror if not already set
      if (length(getOption("repos")) == 0 || getOption("repos")["CRAN"] == "@CRAN@") {
        options(repos = c(CRAN = "https://cloud.r-project.org"))
      }

      ## Try to install the package
      tryCatch(
        {
          install.packages(pkg, dependencies = TRUE)
        },
        error = function(e) {
          message(paste("Failed to install package:", pkg))
          message("Error message:", e$message)
          return(FALSE)
        }
      )

      ## Try to load the package
      if (!require(pkg, character.only = TRUE)) {
        message(paste("Package", pkg, "is not available and could not be installed."))
        return(FALSE)
      }
    }
    return(TRUE)
  }


  ## List of required packages
  packages <- c("caret", "dplyr", "ggplot2", "gtools", "lme4", "pROC", "missMDA", "glmmTMB")


  ## Try to install and load each package
  installed_packages <- sapply(packages, install_and_load)
  if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools")
  }
  library(devtools)
  tryCatch(
    {
      install_version("TMB", version = "1.9.11", repos = "https://cloud.r-project.org")
      detach("package:glmmTMB", unload = TRUE)
      remove.packages("glmmTMB")
      install.packages("glmmTMB")
    },
    error = function(e) {
      message("Error installing packages: ", e$message)
    }
  )

  ## Check if all packages were successfully installed and loaded
  if (!all(installed_packages)) {
    message("Not all required packages could be installed. Please check the error messages above.")
    message("Proceeding with available packages...")
  }
  ## Load packages
  library(caret)
  library(dplyr)
  library(ggplot2)
  library(gtools)
  library(lme4)
  library(pROC)
  library(missMDA)

  #### DATA PREP/EXPLORATION ####
  ## Read in the data
  setwd(args[1])
  base_llm_benchmark_eval <- read.csv(file.path(
    getwd(),
    "/eval_results/base_llm_benchmark_pca_imputed.csv"
  ))
  base_llm_emergent_eval <- read.csv(file.path(
    getwd(),
    "/eval_results/base_llm_emergent_capability_eval.csv"
  ))
  ## Split first column by /
  base_llm_benchmark_eval$Repo <- sapply(
    strsplit(as.character(base_llm_benchmark_eval$Model), "/"), "[", 1
  )
  base_llm_benchmark_eval$Model <- sapply(
    strsplit(as.character(base_llm_benchmark_eval$Model), "/"), "[", 2
  )
  base_llm_emergent_eval$Repo <- sapply(
    strsplit(as.character(base_llm_emergent_eval$Model), "/"), "[", 1
  )
  base_llm_emergent_eval$Model <- sapply(
    strsplit(as.character(base_llm_emergent_eval$Model), "/"), "[", 2
  )
  ## For consistency, convert model & model family names to uppercase
  base_llm_benchmark_eval$Model <- toupper(
    base_llm_benchmark_eval$Model
  )
  base_llm_benchmark_eval$Model.Family <- toupper(
    base_llm_benchmark_eval$Model.Family
  )
  base_llm_emergent_eval$Model <- toupper(
    base_llm_emergent_eval$Model
  )

  emergent_benchmarks <- names(base_llm_emergent_eval)[!names(base_llm_emergent_eval) %in% c("Model", "Repo")]

  ## Consolidate LLAMA and QWEN model families ####
  consolidate_model_family <- function(family) {
    if (grepl("OPENLLAMA", family, ignore.case = TRUE)) {
      return("OPENLLAMA")
    } else if (grepl("CODELLAMA", family, ignore.case = TRUE)) {
      return("CODELLAMA")
    } else if (grepl("LLAMA", family)) {
      return("LLAMA")
    } else if (grepl("QWEN", family)) {
      return("QWEN")
    } else {
      return(family)
    }
  }
  base_llm_benchmark_eval$Model.Family <- sapply(
    base_llm_benchmark_eval$Model.Family,
    consolidate_model_family
  )
  ## Merge datasets by Model, all. Suffix "benchmark" and "emergent"
  base_llm <- merge(
    base_llm_benchmark_eval, base_llm_emergent_eval,
    by = "Model", all = TRUE, suffixes = c(".benchmark", ".emergent")
  )
  ## Convert character columns to factors for modeling
  base_llm[sapply(base_llm, is.character)] <- lapply(
    base_llm[sapply(base_llm, is.character)], as.factor
  )

  ## Count the number of models in each family ####
  family_counts <- table(base_llm$Model.Family)
  print("Model families:")
  print(family_counts)
  ## Identify families with at least 2 models
  valid_families <- names(family_counts[family_counts > 2])
  print("Model families with more than 2 models:")
  print(valid_families)
  ## Filter the dataset to keep only the valid families
  base_llm <- base_llm[base_llm$Model.Family %in% valid_families, ]
  ## Re-factor Model.Family to remove unused levels
  base_llm$Model.Family <- factor(base_llm$Model.Family)
  # Divide ipa_transliterate_2_bleu by 100
  if ("ipa_transliterate_2_bleu" %in% names(base_llm)) {
    base_llm$ipa_transliterate_2_bleu <- base_llm$ipa_transliterate_2_bleu / 100
  }
  # Remove ipa_transliterate_2_bleu column if it exists TODO fix this by running this benchmark
  base_llm$ipa_transliterate_2_bleu <- NULL
  ## Summary stats
  print(summary(base_llm))
  ## Print the number of models remaining
  print(paste("Number of models remaining:", nrow(base_llm)))
  ## Print the remaining model families and their counts
  remaining_family_counts <- table(base_llm$Model.Family)
  print("Remaining model families and their counts:")
  print(remaining_family_counts)


  ## If you want to make a plot like I shared to group + emergent benchmarks
  # install.packages("PerformanceAnalytics") # nolint: commented_code_linter.
  # library(PerformanceAnalytics) # nolint: commented_code_linter.
  ## Correlations of numeric  columns
  # PerformanceAnalytics::chart.Correlation(
  #   base_llm[, sapply(base_llm, is.numeric)], # nolint: commented_code_linter.
  #   histogram = TRUE, method = "pearson"
  # )
  ## You can check help for more options, but basically first argument = data
  ## (probably should manipulate this to zoom in to specific relationships)

  ## Get n-1 score ####
  ## Sort the data frame by Model.Family and FLOPs
  base_llm <- base_llm %>%
    arrange(Model.Family, FLOPs..1E21.) # nolint: object_usage_linter.
  ## Function to find n-1 benchmark score of the smaller model
  get_minus_1_normalized <- function(model_index, data, benchmark_column) {
    current_family <- data$Model.Family[model_index]
    current_flops <- data$FLOPs..1E21.[model_index]
    current_score <- data[[benchmark_column]][model_index]
    ## Filter to smaller than current model_index flops
    smaller_models <- data %>%
      filter(FLOPs..1E21. < current_flops) %>% # nolint: object_usage_linter.
      arrange(desc(FLOPs..1E21.))
    ## First, try to locate the closest smaller model in the same family
    closest_within_family <- smaller_models %>%
      filter(Model.Family == current_family) %>% # nolint: object_usage_linter.
      slice(1)
    ## Second, if no model in family is found, use the closest global model
    closest_global <- smaller_models %>%
      slice(1)
    ## Determine the correct model source (within family or global)
    ## fallback to current if no smaller exists
    if (nrow(closest_within_family) > 0) {
      target_model <- closest_within_family
    } else if (nrow(closest_global) > 0) {
      target_model <- closest_global
    } else {
      ## Fallback to current model's values if no smaller models exist
      target_model <- tibble(
        `FLOPs..1E21.` = current_flops,
        !!benchmark_column := current_score # nolint: object_usage_linter.
      )
    }
    ## Calculate and return the normalized score directly
    minus_1_flops <- target_model$FLOPs..1E21.[1]
    minus_1_score <- target_model[[benchmark_column]][1]
    ## Final normalized score calculation
    return(minus_1_score) * (log(current_flops) / log(minus_1_flops))
  }
  ## Single column use. Apply the function to create 'GSM8K_minus_1'
  # base_llm$GSM8K_minus_1 <- sapply(1:nrow(base_llm), function(idx) {
  #  get_minus_1(idx, base_llm, "GSM8K") # nolint: commented_code_linter.
  # })

  ## Define non-benchmark columns
  non_benchmark_columns <- c(
    "Model", "Model.Family",
    "Model.Size..B.", "Pretraining.Data.Size..T.", "FLOPs..1E21.",
    "Repo.benchmark", "Repo.emergent"
  )
  ## Identify benchmark columns by exclusion
  benchmark_columns <- setdiff(names(base_llm), non_benchmark_columns)

  ## Number of trials (assumed constant across benchmarks bc we're using acc)
  number_of_trials <- 100 # Replace as needed
  ## Apply the 'get_minus_1_normalized' function across all benchmarks
  for (benchmark in benchmark_columns) {
    new_column_name <- paste(benchmark, "minus_1", sep = "_")
    base_llm[[new_column_name]] <- sapply(
      seq_len(nrow(base_llm)),
      function(idx) {
        get_minus_1_normalized(idx, base_llm, benchmark)
      }
    )
  }
  # base_llm : 107 rows, 71 cols 06/19 || 06/26 base_llm 101, 71

  ### MODELING ####
  ## Define the cutoff for FLOPs to split the data
  cutoff_flops <- 8.4 * 10
  ## Function for PCA imputation
  pca_impute <- function(train_df, response_vars, test_df = NULL, n_components = 2, max_iter = 1000, tol = 1e-4, boundary = NULL, verbose = TRUE) {
    if (!is.null(boundary)) {
      if (!is.numeric(boundary) || length(boundary) != 2) {
        stop("Boundary should be a numeric vector of length 2")
      }
    }

    # Separate response variables and non-response numeric variables
    response_train_df <- train_df %>% select(all_of(response_vars))
    numeric_train_df <- train_df %>%
      select(-all_of(response_vars)) %>%
      select_if(is.numeric)
    non_numeric_train_df <- train_df %>% select(!where(is.numeric), -all_of(response_vars))

    if (verbose) {
      cat("Original train_df dimensions:", dim(train_df), "\n")
      cat("Numeric train_df dimensions excluding response:", dim(numeric_train_df), "\n")
    }

    standardize <- function(data) {
      scale(data, center = TRUE, scale = TRUE)
    }

    inverse_standardize <- function(data, mean, sd) {
      data * sd + mean
    }

    train_mean <- colMeans(numeric_train_df, na.rm = TRUE)
    train_sd <- apply(numeric_train_df, 2, sd, na.rm = TRUE)
    train_scaled <- standardize(numeric_train_df)

    if (verbose) {
      na_ratio <- sum(is.na(train_scaled)) / prod(dim(train_scaled))
      cat(sprintf("Missing values in training data: %.2f%%\n", na_ratio * 100))
    }

    train_imputed <- imputePCA(train_scaled, ncp = n_components, method = "Regularized", maxiter = max_iter, threshold = tol)$completeObs
    train_final <- inverse_standardize(train_imputed, train_mean, train_sd)

    if (!is.null(boundary)) {
      train_final <- pmax(pmin(train_final, boundary[2]), boundary[1])
    }

    train_final_df <- as.data.frame(train_final)
    names(train_final_df) <- names(numeric_train_df)
    rownames(train_final_df) <- rownames(numeric_train_df)

    train_final_df <- bind_cols(non_numeric_train_df, train_final_df, response_train_df)

    if (!is.null(test_df)) {
      response_test_df <- test_df %>% select(all_of(response_vars))
      numeric_test_df <- test_df %>%
        select(-all_of(response_vars)) %>%
        select_if(is.numeric)
      non_numeric_test_df <- test_df %>% select(!where(is.numeric), -all_of(response_vars))

      if (verbose) {
        cat("Original test_df dimensions:", dim(test_df), "\n")
        cat("Numeric test_df dimensions excluding response:", dim(numeric_test_df), "\n")
      }

      test_scaled <- scale(numeric_test_df, center = train_mean, scale = train_sd)

      if (verbose) {
        na_ratio_test <- sum(is.na(test_scaled)) / prod(dim(test_scaled))
        cat(sprintf("Missing values in test data: %.2f%%\n", na_ratio_test * 100))
      }

      test_imputed <- imputePCA(test_scaled, ncp = n_components, method = "Regularized", maxiter = max_iter, threshold = tol)$completeObs
      test_final <- inverse_standardize(test_imputed, train_mean, train_sd)

      if (!is.null(boundary)) {
        test_final <- pmax(pmin(test_final, boundary[2]), boundary[1])
      }

      test_final_df <- as.data.frame(test_final)
      names(test_final_df) <- names(numeric_test_df)
      rownames(test_final_df) <- rownames(numeric_test_df)

      test_final_df <- bind_cols(non_numeric_test_df, test_final_df, response_test_df)
    } else {
      test_final_df <- NULL
    }

    return(list(train_data = train_final_df, test_data = test_final_df))
  }
  ## Function to fit the fixed-effects model with error handling

  fit_model_fixed <- function(formula, data) {
    model <- tryCatch(
      {
        glm(formula, data = data, family = quasibinomial(link = "logit"))
      },
      warning = function(w) {
        message("Warning in model fitting: ", w$message)
        return(NA)
      },
      error = function(e) {
        message("Error in model fitting: ", e$message)
        return(NA)
      },
      finally = {
        message("Attempted model: ", deparse(formula))
      }
    )
    return(model)
  }
  ## Function to fit the mixed-effects model with error handling

  library(glmmTMB)
  fit_model <- function(formula, data) {
    model <- tryCatch(
      {
        print(formula)
        glmmTMB(formula,
          data = data,
          family = beta_family(link = "logit"),
          control = glmmTMBControl(optimizer = nlminb)
        )
      },
      warning = function(w) {
        message("Warning in model fitting: ", w$message)
        return(w)
      },
      error = function(e) {
        message("Error in model fitting: ", e$message)
        return(e)
      },
      finally = {
        message("Attempted model: ", deparse(formula))
      }
    )
    return(model)
  }

  ## Function to get model metrics with error handling
  calculate_model_metrics <- function(model, test_data, response_formula) {
    if (is.null(model) || inherits(model, "try-error") || all(is.na(model))) {
      message("Model is null, an error, or all NA")
      return(rep(NA, 3))
    }
    tryCatch(
      {
        if (inherits(model, "glm")) {
          aic_value <- NA # AIC is not available for quasi-models
          test_predictions <- predict(model, newdata = test_data, type = "response")
        } else if (inherits(model, "glmmTMB")) {
          aic_value <- AIC(model)
          test_predictions <- predict(model, newdata = test_data, type = "response")
        } else {
          warng("Unexpected model class: ", class(model))
          return(c(NA, NA, NA))
        }
        actual_values <- test_data[[response_formula]]
        rmse <- sqrt(mean((actual_values - test_predictions)^2))
        print(paste("RMSE:", rmse))
        r_squared <- cor(actual_values, test_predictions)^2
        print(paste("R-squared:", r_squared))
        return(c(aic_value, r_squared, rmse))
      },
      error = function(e) {
        warning("Error in calculate_model_metrics: ", e$message)
        # print("Error occurred with model of class:", paste(class(model), collapse = ", "))
        # print("Response formula:", response_formula)
        # print("Test data columns:", names(test_data))
        return(rep(NA, 3))
      }
    )
  }

  ## Container for best results
  # best_models <- list() # nolint: commented_code_linter
  ## Container for performance metrics
  performance_metrics <- data.frame(
    benchmark = character(),
    model_type = character(),
    aic = numeric(),
    binary_accuracy = numeric(),
    rmse = numeric(),
    stringsAsFactors = FALSE
  )

  ## Get all combinations of benchmarks for formulas
  benchmark_minus_1_columns <- paste0(benchmark_columns, "_minus_1")
  all_combinations <- do.call(c, lapply(1:1, function(k) {
    combn(benchmark_minus_1_columns, k, simplify = FALSE)
  })) # use 1:16 for all combinations except empty set

  emergent_benchmark_columns <- benchmark_columns[benchmark_columns %in% emergent_benchmarks]

  for (benchmark in emergent_benchmark_columns[length(emergent_benchmark_columns):1]) {
    ## Dynamic responses (Y) based on current benchmark
    response_formula <- benchmark
    response <- benchmark
    # response_formula <- paste(benchmark, "successes", sep = "_")
    # response_failures <- paste(benchmark, "failures", sep = "_")

    benchmark_minus_1 <- paste0(benchmark, "_minus_1")

    ## Remove NAs # TODO:dynamic depending on formula
    base_llm_clean <- base_llm[complete.cases(base_llm[c(
      response_formula,
      "FLOPs..1E21.", "Model.Family", benchmark_minus_1
    )]), ]

    ## Split data into training and testing sets # dynamic as well?
    train_df <- base_llm %>% filter(FLOPs..1E21. <= cutoff_flops) # nolint: object_usage_linter
    test_df <- base_llm %>% filter(FLOPs..1E21. > cutoff_flops) # nolint: object_usage_linter

    ## Perform PCA imputation on sets without family
    response_vars <- c(response_formula, response_failures)
    result <- pca_impute(train_df, response_vars, test_df, n_components = 2, max_iter = 100, tol = 1e-4, boundary = c(0, 100), verbose = TRUE)

    ## Change sets to imputed sets, add family back
    train_data <- result$train_data
    test_data <- result$test_data

    ## Model 0: Base model with FLOPs only (fixed-effects)
    formula_base <- as.formula(paste(response, "~ 1 + log(FLOPs..1E21. + 0.001)"))
    model_base <- fit_model_fixed(formula_base, train_data)
    base_metrics <- calculate_model_metrics(model_base, test_data, response_formula) # nolint: object_usage_linter
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops Only",
      base_metrics[1], base_metrics[2], base_metrics[3]
    )

    ## Model 1: Base model with FLOPs only + random family intercepts
    formula_base <- as.formula(paste(
      response,
      "~ log(FLOPs..1E21. + 0.001) + (1|Model.Family)"
    ))
    print(response)
    print(benchmark)
    model_base <- fit_model(formula_base, train_data)
    base_metrics <- calculate_model_metrics(model_base, test_data, response_formula) # nolint: object_usage_linter
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops Only + intercept",
      base_metrics[1], base_metrics[2], base_metrics[3]
    )

    ## Model 1.5: Base model with FLOPs only + random family intercepts & slopes
    formula_base <- as.formula(paste(
      response,
      "~ log(FLOPs..1E21. + 0.001) + (1 + log(FLOPs..1E21. + 0.001)|Model.Family)"
    ))
    model_base <- fit_model(formula_base, train_data)
    base_metrics <- calculate_model_metrics(
      model_base,
      test_data, response_formula
    )
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops Only + intercept-slope",
      base_metrics[1], base_metrics[2], base_metrics[3]
    )

    ## Model 2: Base model with FLOPs + benchmark_minus_1
    formula_minus_1 <- as.formula(paste0(
      response,
      " ~ log(FLOPs..1E21. + 0.001) + ", benchmark_minus_1
    ))
    model_minus_1 <- fit_model_fixed(formula_minus_1, train_data)
    minus_1_metrics <- calculate_model_metrics(
      model_minus_1,
      test_data, response_formula
    )
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops + n-1",
      minus_1_metrics[1], minus_1_metrics[2], minus_1_metrics[3]
    )

    ## TODO: generalize the above?

    ## Grid search #### 06/26, FLOPs, family intercepts,
    formula_components <- "log(FLOPs..1E21.) + (1|Model.Family)"
    # formulas <- lapply(seq_along(all_combinations), function(comb) {
    #  components <- paste(all_combinations[comb], collapse = " + ")
    #  formula_str <- paste(response, "~", formula_components, "+", components)
    #  as.formula(formula_str)
    # })
    formulas <- lapply(all_combinations, function(comb) {
      components <- paste(comb, collapse = " + ")
      formula_str <- paste(response, "~", formula_components, "+", components)
      as.formula(formula_str)
    })
    for (formula in formulas) {
      model_dynamic <- fit_model(formula, train_data)
      dynamic_metrics <- calculate_model_metrics(
        model_dynamic,
        test_data, response_formula
      )
      if (!is.na(dynamic_metrics[1])) {
        print("Writing to performance metrics...")
        print(gsub("\\s+", " ", paste(deparse(formula), collapse = "")))
        performance_metrics[nrow(performance_metrics) + 1, ] <- c(
          benchmark,
          gsub("\\s+", " ", paste(deparse(formula), collapse = "")), #
          dynamic_metrics[1],
          dynamic_metrics[2],
          dynamic_metrics[3]
        )
      }
    }

    formula_components <- "1 + log(FLOPs..1E21.)"
    formulas <- lapply(all_combinations, function(comb) {
      components <- paste(comb, collapse = " + ")
      formula_str <- paste(response, "~", formula_components, "+", components)
      as.formula(formula_str)
    })
    for (formula in formulas) {
      model_dynamic <- fit_model_fixed(formula, train_data)
      dynamic_metrics <- calculate_model_metrics(
        model_dynamic,
        test_data, response_formula
      )
      print("Writing to performance metrics...")
      print(gsub("\\s+", " ", paste(deparse(formula), collapse = "")))
      performance_metrics[nrow(performance_metrics) + 1, ] <- c(
        benchmark,
        gsub("\\s+", " ", paste(deparse(formula), collapse = "")), #
        dynamic_metrics[1],
        dynamic_metrics[2],
        dynamic_metrics[3]
      )
    }

    ## Determine the best model for this benchmark
    ## commented out bc we're retaining all results now,
    ## performance metrics out of loop
    # if (any(!is.na(performance_metrics$rmse))) {
    #   best_model <- performance_metrics %>%
    #     filter(
    #       rmse == min(rmse, na.rm = TRUE), # nolint: object_usage_linter
    #     ) # nolint: object_usage_linter.
    #   best_models[[benchmark]] <- best_model # nolint: commented_code_linter
    # } else {
    #   best_models[[benchmark]] <- NA # nolint: commented_code_linter
    #   message("All models failed for benchmark: ", benchmark) # nolint: commented_code_linter
    # }
    print("writing...")
    write.csv2(performance_metrics, file.path(getwd(), "performance_metrics.csv"))
  }

  # print(best_models) # nolint: commented_code_linter
  # print(performance_metrics) # nolint: commented_code_linter

  # Optionally, plot ROC curve for the last evaluated benchmark
  # plot(roc_obj) # nolint: commented_code_linter

  write.csv2(performance_metrics, file.path(getwd(), "performance_metrics_PCA_imputed.csv"))
  # Ensure everything is numerical where necessary
  performance_metrics <- performance_metrics %>%
    mutate(
      aic = as.numeric(aic), # nolint: object_usage_linter
      binary_accuracy = as.numeric(binary_accuracy), # nolint: object_usage_linter
      rmse = as.numeric(rmse) # nolint: object_usage_linter
    )

  # Selecting the best model types by lowest AIC for each benchmark
  best_model_type <- performance_metrics %>%
    group_by(benchmark) %>%
    filter(rmse == min(rmse, na.rm = TRUE)) %>% # nolint: object_usage_linter
    select(benchmark, model_type, aic, binary_accuracy, rmse) # nolint: object_usage_linter

  print(best_model_type)

  # Create factor for model_type with "Other" category for non-matching types
  best_model_type <- performance_metrics %>%
    mutate(
      model_type = case_when(
        model_type %in% c(
          "Flops Only",
          "Flops Only + intercept", "Flops Only + intercept-slope",
          "Flops + n-1"
        ) ~ model_type,
        TRUE ~ "Other"
      ),
      model_type = factor(model_type, levels = c( # nolint: object_usage_linter
        "Flops Only",
        "Flops Only + intercept", "Flops Only + intercept-slope",
        "Flops + n-1", "Other"
      ))
    )

  # Compute average performance metrics for each model_type
  avg_performance_metrics <- best_model_type %>%
    group_by(model_type) %>% # nolint: object_usage_linter
    summarise(
      avg_aic = mean(aic, na.rm = TRUE), # nolint: object_usage_linter
      avg_binary_accuracy = mean(binary_accuracy, na.rm = TRUE), # nolint: object_usage_linter
      avg_rmse = mean(rmse, na.rm = TRUE) # nolint: object_usage_linter
    )

  print(avg_performance_metrics)

  # Function to extract unique predictors from formulas
  extract_predictors <- function(formula) {
    predictors <- unlist(strsplit(formula, "\\s*\\+\\s*"))
    predictors <- predictors[!grepl("log\\(FLOPs|\\(1 \\| Model\\.Family\\)", predictors)]
    return(predictors)
  }

  # Apply the function and create a data frame of predictors and corresponding RMSEs
  extracted_predictors <- do.call(rbind, lapply(1:nrow(performance_metrics), function(i) {
    data.frame(
      predictor = extract_predictors(performance_metrics$model_type[i]),
      avg_rmse = performance_metrics$avg_rmse[i]
    )
  }))

  # Calculate the average RMSE for each unique predictor
  avg_rmse_per_predictor <- extracted_predictors %>%
    group_by(predictor) %>%
    summarise(avg_rmse = mean(avg_rmse, na.rm = TRUE)) %>%
    arrange(avg_rmse)

  print(avg_rmse_per_predictor)
}

main()
