# args <- c() # nolint: commented_code_linter, commented_code_linter.
# args[1] <- "~/git/predicting-capabilities/ObsScaling/" # nolint: commented_code_linter, line_length_linter.
main <- function() { # nolint: cyclocomp_linter.
  args <- commandArgs(trailingOnly = TRUE)
  #### Libraries, if not yet installed use install.packages("package_name")
  library(lme4)
  library(dplyr)
  library(ggplot2)
  library(pROC)
  library(caret)
  library(gtools)

  #### DATA PREP/EXPLROATION
  ## Read in the data
  setwd(args[1])
  base_llm_benchmark_eval <- read.csv(file.path(
    getwd(),
    "/eval_results/base_llm_benchmark_eval.csv"
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
  ## Merge datasets by Model, all. Suffix "benchmark" and "emergent"
  base_llm <- merge(
    base_llm_benchmark_eval, base_llm_emergent_eval,
    by = "Model", all = TRUE, suffixes = c(".benchmark", ".emergent")
  )
  ## Convert character columns to factors for modeling
  base_llm[sapply(base_llm, is.character)] <- lapply(
    base_llm[sapply(base_llm, is.character)], as.factor
  )
  ## Summary stats
  print(summary(base_llm))

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

  ## Get n-1 score
  ## Sort the data frame by Model.Family and FLOPs
  base_llm <- base_llm %>%
    arrange(Model.Family, FLOPs..1E21.) # nolint: object_usage_linter.
  ## Function to find n-1 benchmark score of the smaller model
  get_minus_1_normalized <- function(model_index, data, benchmark_column) {
    current_family <- data$Model.Family[model_index]
    current_flops <- data$FLOPs..1E21.[model_index]
    current_score <- data[[benchmark_column]][model_index]
    # Filter to smaller than current model_index flops
    smaller_models <- data %>%
      filter(FLOPs..1E21. < current_flops) %>% # nolint: object_usage_linter.
      arrange(desc(FLOPs..1E21.))
    # First, try to locate the closest smaller model in the same family
    closest_within_family <- smaller_models %>%
      filter(Model.Family == current_family) %>% # nolint: object_usage_linter.
      slice(1)
    # Second, if no model in family is found, use the closest global model
    closest_global <- smaller_models %>%
      slice(1)
    # Determine the correct model source (within family or global)
    # fallback to current if no smaller exists
    if (nrow(closest_within_family) > 0) {
      target_model <- closest_within_family
    } else if (nrow(closest_global) > 0) {
      target_model <- closest_global
    } else {
      # Fallback to current model's values if no smaller models exist
      target_model <- tibble(
        `FLOPs..1E21.` = current_flops,
        !!benchmark_column := current_score # nolint: object_usage_linter.
      )
    }
    # Calculate and return the normalized score directly
    minus_1_flops <- target_model$FLOPs..1E21.[1]
    minus_1_score <- target_model[[benchmark_column]][1]
    # Final normalized score calculation
    return(minus_1_score) * (log(current_flops) / log(minus_1_flops))
  }
  ## Single column use. Apply the function to create 'GSM8K_minus_1'
  # base_llm$GSM8K_minus_1 <- sapply(1:nrow(base_llm), function(idx) {
  #  get_minus_1(idx, base_llm, "GSM8K") # nolint: commented_code_linter.
  # })

  # Define non-benchmark columns
  non_benchmark_columns <- c(
    "Model", "Model.Family",
    "Model.Size..B.", "Pretraining.Data.Size..T.", "FLOPs..1E21.",
    "Repo.benchmark", "Repo.emergent"
  )
  # Identify benchmark columns by exclusion
  benchmark_columns <- setdiff(names(base_llm), non_benchmark_columns)
  # Number of trials (assuming it's a constant across benchmarks)
  # Apply the 'get_minus_1_normalized' function across all benchmarks
  number_of_trials <- 100 # Replace as needed
  for (benchmark in benchmark_columns) {
    new_column_name <- paste(benchmark, "minus_1", sep = "_")
    base_llm[[new_column_name]] <- sapply(
      seq_len(nrow(base_llm)),
      function(idx) {
        get_minus_1_normalized(idx, base_llm, benchmark)
      }
    )
    # Dynamically create successes and failures columns as well
    base_llm[[paste(benchmark, "successes", sep = "_")]] <-
      round(base_llm[[benchmark]] * number_of_trials)
    base_llm[[paste(benchmark, "failures", sep = "_")]] <-
      number_of_trials - base_llm[[paste(benchmark, "successes", sep = "_")]]
  }
  # base_llm : 107 rows, 71 cols 06/19

  ### MODELING
  ## REMOVED: Example of how to fit a single mixed effects model

  # TODO: grid search of all possible predictive benchmarks. Done: 06/12 flops, n-1 # nolint
  # Define the cutoff for FLOPs to split the data
  cutoff_flops <- 8.4 * 10
  # Function to fit the model with error handling
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

  # Function to calculate model metrics
  calculate_model_metrics <- function(model) {
    if (!is.na(model) && class(model) != "try-error") {
      aic_value <- AIC(model)
      test_predictions <- predict(model,
        newdata = test_data, re.form = NA, type = "response"
      )
      accuracy <- sum(round(test_data[[response_formula]] / 100, 1) == round(test_predictions, 1)) / length(test_predictions)
      rmse <- sqrt(mean((round(test_data[[response_formula]] / 100, 2) - round(test_predictions / 100, 2))^2))
      # r_squared <- cor(round(test_data[[response_formula]] / 100, 2), round(test_predictions / 100, 2))^2
      return(c(aic_value, accuracy, rmse))
    } else {
      return(rep(NA, 3))
    }
  }

  # # Container for results
  # best_models <- list()

  # Container for performance metrics
  performance_metrics <- data.frame(
    benchmark = character(),
    model_type = character(),
    aic = numeric(),
    binary_accuracy = numeric(),
    rmse = numeric(),
    stringsAsFactors = FALSE
  )

  benchmark_minus_1_columns <- paste0(benchmark_columns, "_minus_1")
  all_combinations <- do.call(c, lapply(1:3, function(k) {
    combn(benchmark_minus_1_columns, k, simplify = FALSE)
  })) # use 1:16 for all combinations except empty set

  for (benchmark in benchmark_columns) {
    # Dynamic responses based on current benchmark
    response_formula <- paste(benchmark, "successes", sep = "_")
    response_failures <- paste(benchmark, "failures", sep = "_")
    response <- paste("cbind(", response_formula, ",", response_failures, ")", sep = "")
    benchmark_minus_1 <- paste0(benchmark, "_minus_1")

    # Remove NAs
    base_llm_clean <- base_llm[complete.cases(base_llm[c(response_formula, response_failures, "FLOPs..1E21.", "Model.Family", benchmark_minus_1)]), ]

    # Split data into training and testing sets
    train_data <- base_llm_clean %>% filter(FLOPs..1E21. <= cutoff_flops) # nolint: object_usage_linter
    test_data <- base_llm_clean %>% filter(FLOPs..1E21. > cutoff_flops) # nolint: object_usage_linter

    # Model 1: Base model with FLOPs only
    formula_base <- as.formula(paste(response, "~ log(FLOPs..1E21.) + (1|Model.Family)"))
    model_base <- fit_model(formula_base, train_data)
    base_metrics <- calculate_model_metrics(model_base)
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(benchmark, "Flops Only", base_metrics[1], base_metrics[2], base_metrics[3])

    # Model 2: Base model with FLOPs and benchmark minus 1
    formula_minus_1 <- as.formula(paste0(response, " ~ log(FLOPs..1E21.) + (1|Model.Family) + ", benchmark_minus_1))
    model_minus_1 <- fit_model(formula_minus_1, train_data)
    minus_1_metrics <- calculate_model_metrics(model_minus_1)
    performance_metrics[nrow(performance_metrics) + 1, ] <- c(benchmark, "Flops + n-1", minus_1_metrics[1], minus_1_metrics[2], minus_1_metrics[3])

    # Grid search: Add each of the other benchmarks minus 1 incrementally
    formula_components <- "log(FLOPs..1E21.) + (1|Model.Family)"

    formulas <- lapply(seq_along(all_combinations), function(comb) {
      components <- paste(comb, collapse = " + ")
      formula_str <- paste(response, "~", formula_components, "+", components)
      as.formula(formula_str)
    })

    for (formula in formulas) {
      model_dynamic <- fit_model(formula, train_data)
      dynamic_metrics <- calculate_model_metrics(model_dynamic)
      if (!is.na(dynamic_metrics[1])) {
        performance_metrics[nrow(performance_metrics) + 1, ] <- c(
          benchmark,
          paste(formula[3]),
          dynamic_metrics[1],
          dynamic_metrics[2],
          dynamic_metrics[3]
        )
      }
    }

    # Determine the best model for this benchmark
    # if (any(!is.na(performance_metrics$aic))) {
    #   best_model <- performance_metrics %>%
    #     filter(
    #       aic == min(aic, na.rm = TRUE),
    #     ) # nolint: object_usage_linter.
    #   best_models[[benchmark]] <- best_model
    # } else {
    #   best_models[[benchmark]] <- NA
    #   message("All models failed for benchmark: ", benchmark)
    # }
  }

  # print(best_models)
  # print(performance_metrics)

  # Optionally, plot ROC curve for the last evaluated benchmark
  # plot(roc_obj)

  # Done? train/test split
}

main()
