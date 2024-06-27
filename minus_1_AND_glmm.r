# args[1] <- "~/git/predicting-capabilities/ObsScaling/" # nolint: commented_code_linter, line_length_linter.
main <- function() { # nolint: function_name_linter
  args <- commandArgs(trailingOnly = TRUE)
  #### Libraries ####
  ## List of required packages in alphabetical order
  packages <- c("caret", "dplyr", "ggplot2", "gtools", "lme4", "pROC")
  ## Function to install missing packages and load them
  install_and_load <- function(pkg) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
  ## Apply the function to the list of packages
  lapply(packages, install_and_load)
  ## Load packages
  library(caret)
  library(dplyr)
  library(ggplot2)
  library(gtools)
  library(lme4)
  library(pROC)

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
  ## Identify families with at least 2 models
  valid_families <- names(family_counts[family_counts >= 2])
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
    ## Dynamically create successes and failures columns as well for response
    base_llm[[paste(benchmark, "successes", sep = "_")]] <-
      round(base_llm[[benchmark]] * number_of_trials)
    base_llm[[paste(benchmark, "failures", sep = "_")]] <-
      number_of_trials - base_llm[[paste(benchmark, "successes", sep = "_")]]
  }
  # base_llm : 107 rows, 71 cols 06/19 || 06/26 base_llm 101, 71

  ### MODELING ####
  ## Define the cutoff for FLOPs to split the data
  cutoff_flops <- 8.4 * 10
  ## Function to fit the fixed-effects model with error handling
  fit_model_fixed <- function(formula, data) {
    model <- tryCatch(
      {
        glm(formula,
          data = data,
          family = binomial(link = "logit")
        )
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
  fit_model <- function(formula, data) {
    model <- tryCatch(
      {
        glmer(formula,
          data = data,
          family = binomial(link = "logit"),
          control = glmerControl(optimizer = "bobyqa")
        )
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
  ## Function to get model metrics with error handling
  calculate_model_metrics <- function(model, test_data, response_formula) {
    if (is.null(model) || inherits(model, "try-error") || all(is.na(model))) {
      return(rep(NA, 3))
    }
    tryCatch(
      {
        if (inherits(model, "glm")) {
          aic_value <- AIC(model)
          test_predictions <- predict(model,
            newdata = test_data, type = "response"
          )
        } else if (inherits(model, "glmerMod")) {
          aic_value <- AIC(model)
          test_predictions <- predict(model,
            newdata = test_data, re.form = NA, type = "response"
          )
        } else {
          warning("Unexpected model class")
          return(c(NA, NA, NA))
        }
        actual_values <- test_data[[response_formula]] / 100
        accuracy <- mean(round(actual_values, 1) == round(test_predictions, 1))
        rmse <- sqrt(mean((actual_values - test_predictions)^2))
        return(c(aic_value, accuracy, rmse))
      },
      error = function(e) {
        warning("Error in calculate_model_metrics: ", e$message)
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
  all_combinations <- do.call(c, lapply(1:8, function(k) {
    combn(benchmark_minus_1_columns, k, simplify = FALSE)
  })) # use 1:16 for all combinations except empty set

  for (benchmark in benchmark_columns[length(benchmark_columns):1]) {
    ## Dynamic responses (Y) based on current benchmark
    response_formula <- paste(benchmark, "successes", sep = "_")
    response_failures <- paste(benchmark, "failures", sep = "_")
    response <- paste("cbind(", response_formula, ",", response_failures, ")",
      sep = ""
    )
    benchmark_minus_1 <- paste0(benchmark, "_minus_1")

    ## Remove NAs # TODO:dynamic depending on formula
    base_llm_clean <- base_llm[complete.cases(base_llm[c(
      response_formula, response_failures,
      "FLOPs..1E21.", "Model.Family", benchmark_minus_1
    )]), ]

    ## Split data into training and testing sets # dynamic as well?
    train_data <- base_llm_clean %>% filter(FLOPs..1E21. <= cutoff_flops) # nolint: object_usage_linter
    test_data <- base_llm_clean %>% filter(FLOPs..1E21. > cutoff_flops) # nolint: object_usage_linter

    ## Model 0: Base model with FLOPs only (fixed-effects)
    formula_base <- as.formula(paste(response, "~ 1 + log(FLOPs..1E21.)"))
    model_base <- fit_model_fixed(formula_base, train_data)
    base_metrics <- calculate_model_metrics(model_base, test_data, response_formula) # nolint: object_usage_linter
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops Only",
      base_metrics[1], base_metrics[2], base_metrics[3]
    )

    ## Model 1: Base model with FLOPs only + random family intercepts
    formula_base <- as.formula(paste(
      response,
      "~ log(FLOPs..1E21.) + (1|Model.Family)"
    ))
    model_base <- fit_model(formula_base, train_data)
    base_metrics <- calculate_model_metrics(model_base, test_data, response_formula) # nolint: object_usage_linter
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops Only + intercept",
      base_metrics[1], base_metrics[2], base_metrics[3]
    )

    ## Model 1.5: Base model with FLOPs only + random family intercepts & slopes
    formula_base <- as.formula(paste(
      response,
      "~ log(FLOPs..1E21.) + (1 + log(FLOPs..1E21.)|Model.Family)"
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

    ## Model 2: Base model with FLOPs, random family intercept, benchmark_minus_1 # nolint
    formula_minus_1 <- as.formula(paste0(
      response,
      " ~ log(FLOPs..1E21.) + (1|Model.Family) + ", benchmark_minus_1
    ))
    model_minus_1 <- fit_model(formula_minus_1, train_data)
    minus_1_metrics <- calculate_model_metrics(
      model_minus_1,
      test_data, response_formula
    )
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(
      benchmark, "Flops + intercept + n-1",
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
        performance_metrics[nrow(performance_metrics) + 1, ] <- c(
        benchmark,
        deparse(formula),
        dynamic_metrics[1],
        dynamic_metrics[2],
        dynamic_metrics[3]
        )
      }
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
  }

  # print(best_models) # nolint: commented_code_linter
  # print(performance_metrics) # nolint: commented_code_linter

  # Optionally, plot ROC curve for the last evaluated benchmark
  # plot(roc_obj) # nolint: commented_code_linter

  write.csv2(performance_metrics, file.path(getwd(), "performance_metrics.csv"))
}

main()
