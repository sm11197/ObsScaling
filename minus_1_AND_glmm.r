# args <- c() # nolint: commented_code_linter, commented_code_linter.
# args[1] <- "~/git/predicting-capabilities/ObsScaling/" # nolint: commented_code_linter, line_length_linter.
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  #### Libraries, if not yet installed use install.packages
  library(lme4)
  library(dplyr)
  library(ggplot2)

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
    return(minus_1_score * log(current_flops) / log(minus_1_flops))
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
  # Apply the 'get_minus_1_normalized' function across all benchmarks
  for (benchmark_column in benchmark_columns) {
    new_column_name <- paste(benchmark_column, "minus_1", sep = "_")
    base_llm[[new_column_name]] <- sapply(
      seq_len(nrow(base_llm)),
      function(idx) {
        get_minus_1_normalized(idx, base_llm, benchmark_column)
      }
    )
  }
  # base_llm : 107 rows, 39 cols 06/12

  ## Relationship Check
  ## just eyeballing making it more linear for now
  # plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$GSM8K + 0.01)) # nolint
  # plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$FLOPs..1E21.)) # nolint
  # plot(base_llm_clean$XWinograd, log10(base_llm_clean$XWinograd_minus_1 + 0.01)) # nolint

  ### MODELING
  ## Example of how to fit a single mixed effects model

  # mixed_model <- lmer(
  #   arithmetic_3ds_2_acc ~ log(FLOPs..1E21.) +
  #     (1 | Model.Family) + log(GSM8K_minus_1 + 0.01), # nolint: commented_code_linter, line_length_linter.
  #   data = base_llm_clean, REML = FALSE
  # )
  # Done: binomial GLMM
  # TODO: replace with number of questions in benchmark?
  base_llm_clean$successes <- round(base_llm_clean$XWinograd * 100)
  base_llm_clean$failures <- 100 - base_llm_clean$successes
  # mixed_model <- glmer(
  #   cbind(successes, failures) ~ log(FLOPs..1E21.) +
  #     (1 | Model.Family) + log(XWinograd_minus_1), # nolint: commented_code_linter, line_length_linter.
  #   data = base_llm_clean, # nolint: commented_code_linter.
  #   family = binomial(link = "logit"), # nolint: commented_code_linter.
  #   control = glmerControl(optimizer = "bobyqa") # nolint: commented_code_linter, line_length_linter.
  # )
  # Done: random slopes as well as intercepts
  mixed_model <- glmer(
    cbind(successes, failures) ~ log(FLOPs..1E21.) +
      (1 | Model.Family) + log(XWinograd_minus_1 + 0.01),
    data = base_llm_clean,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa")
  )
  # Check Random Effects
  print(ranef(mixed_model)) # Check the random effects output
  summary(mixed_model)
  # Plotting to assess possible issues
  fitted.values <- fitted(mixed_model)
  residuals <- resid(mixed_model, type = "pearson")
  # Basic residual plot
  plot(fitted.values, residuals)
  abline(h = 0, col = "red")
  # Post-fit prediction check (you'll need actual observed data here)
  observed <- with(base_llm_clean, cbind(successes, failures))
  predicted <- predict(mixed_model, type = "response", re.form = NA)
  # Plot observed vs. predicted (logistic predictions)
  observed_prop <- observed[, 1] / rowSums(observed)
  plot(observed_prop, predicted,
    xlab = "Observed XWinograd", ylab = "Predicted XWinograd",
    main = "XWinograd", sub = "XWinograd ~ log(FLOPs..1E21.) +
    (1 | Model.Family) + log(XWinograd_minus_1)",
    cex.sub = 0.75, xlim = c(0.4, 0.9), ylim = c(0.4, 0.9)
  )
  abline(a = 0, b = 1, col = "red") # Ideal line where observed equals predicted

  # TODO: grid search of all possible predictive models
  # Calculating AIC and guarding against errors
  calculate_model_aic <- function(model) {
    if (class(model) != "try-error") {
      aic_value <- AIC(model)
      return(aic_value)
    } else {
      return(NA)
    }
  }

  # Helper function to fit model with tryCatch
  fit_model <- function(formula, data) {
    model <- tryCatch(
      {
        # Proper glmer call within a tryCatch
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

  # Container for results
  best_models <- list()

  # Number of trials (assuming it's a constant across benchmarks)
  number_of_trials <- 100 # Replace as needed if you have actual data

  for (benchmark in benchmark_columns) {
    # Dynamically create successes and failures columns
    base_llm_clean[[paste(benchmark, "successes", sep = "_")]] <-
      round(base_llm_clean[[benchmark]] * number_of_trials)
    base_llm_clean[[paste(benchmark, "failures", sep = "_")]] <-
      number_of_trials - base_llm_clean[[paste(benchmark,
        "successes",
        sep = "_"
      )]]

    aic_scores <- data.frame(
      model_type = character(),
      aic = numeric(),
      stringsAsFactors = FALSE
    )

    # Dynamic responses based on current benchmark
    response_formula <- paste(benchmark, "successes", sep = "_")
    response_failures <- paste(benchmark, "failures", sep = "_")
    response <- paste("cbind(", response_formula, ",",
      response_failures, ")",
      sep = ""
    )

    # Model 1: Base model
    formula_base <- as.formula(paste(response, "~ (1|Model.Family) +
    log(", benchmark, " + 0.01)"))
    model_base <- fit_model(formula_base, base_llm_clean)
    aic_scores[nrow(aic_scores) + 1, ] <- c(
      "Base",
      calculate_model_aic(model_base)
    )

    # Model 2: Including FLOPs # TODO, flops only
    formula_flops <- as.formula(paste(
      response,
      "~ log(FLOPs..1E21.) + (1|Model.Family) + log(", benchmark, " + 0.01)"
    ))
    model_flops <- fit_model(formula_flops, base_llm_clean)
    aic_scores[nrow(aic_scores) + 1, ] <- c(
      "FLOPs",
      calculate_model_aic(model_flops)
    )

    # Determine the best model for this benchmark
    if (any(!is.na(aic_scores$aic))) {
      best_model <- aic_scores %>%
        filter(aic == min(aic, na.rm = TRUE)) # nolint: object_usage_linter.
      best_models[[benchmark]] <- best_model
    } else {
      best_models[[benchmark]] <- NA
      message("All models failed for benchmark: ", benchmark)
    }
  }

  # Let's choose one benchmark to work with, say 'benchmark_i'
  benchmark_i <- "MMLU" # the target benchmark

  # Dynamically create scores and failures for this benchmark
  number_of_trials <- 100 # Assuming 100 trials; replace or modify as needed
  base_llm_clean[[paste(benchmark_i, "successes", sep = "_")]] <-
    round(base_llm_clean[[benchmark_i]] * number_of_trials)
  base_llm_clean[[paste(benchmark_i, "failures", sep = "_")]] <-
    number_of_trials - base_llm_clean[[paste(benchmark_i,
      "successes",
      sep = "_"
    )]]

  # List of possible predictors: other benchmarks' minus_1 scores
  predictor_benchmarks <- grep("minus_1", names(base_llm_clean), value = TRUE)
  predictor_benchmarks <- setdiff(
    predictor_benchmarks,
    paste(benchmark_i, "minus_1", sep = "_")
  ) # Exclude its own minus_1

  # Initialize stores for results
  best_models <- list()
  aic_scores <- vector("numeric", length(predictor_benchmarks))

  # Define response for model
  response <- paste("cbind(", paste(benchmark_i, "successes", sep = "_"),
    ",", paste(benchmark_i, "failures", sep = "_"), ")",
    sep = ""
  )

  # Grid Search: Try each minus 1 score as a predictor
  for (k in seq_along(predictor_benchmarks)) {
    formula <- as.formula(paste(
      response, "~",
      paste("log(", predictor_benchmarks[k], "+0.01) + (1|Model.Family)")
    ))
    model <- tryCatch(
      {
        glmer(formula, data = base_llm_clean, family = binomial(link = "logit"))
      },
      error = function(e) {
        message(
          "Error with model using predictor ",
          predictor_benchmarks[k], ": ", e$message
        )
        return(NA)
      }
    )

    # Store model if successful
    if (!is.na(model)) {
      aic_scores[k] <- AIC(model)
      best_models[[predictor_benchmarks[k]]] <- list(
        formula = formula,
        aic = AIC(model),
        model = model
      )
    }
  }

  # Determine best model from models that completed successfully
  if (length(best_models) > 0) {
    best_fit_index <- which.min(aic_scores)
    best_fit_model <- best_models[[best_fit_index]]
    print(best_fit_model)
  } else {
    message("No successful model fits were found.")
  }

  # Print out the best models for each benchmark
  print(best_models)

  # Extract the AICs and find the minimum
  min_aic <- min(sapply(best_models, function(x) x$aic))
  best_model_name <- names(which.min(sapply(best_models, function(x) x$aic)))

  # Output the best model details
  cat("Best Model is based on predictor:", best_model_name, "\n")
  cat("It has the lowest AIC of:", min_aic, "\n")
  print(best_models[[best_model_name]])

  plot(base_llm_clean$ipa_transliterate_2_bleu, base_llm_clean$MMLU)

  # TODO: train/test split



  # Create the scatter plot with facets for each Model.Family
  ggplot(
    base_llm,
    aes(x = FLOPs..1E21., y = MMLU) # nolint: object_usage_linter.
  ) +
    geom_point() + # Add points
    facet_wrap(~Model.Family, scales = "free") + # Add facets
    labs(
      x = "FLOPs (1E21)", y = "MMLU Scores",
      title = "Relationship between FLOPs and MMLU by Model.Family"
    ) +
    theme_minimal()

  # Create the scatter plot coloring points by Model.Family
  ggplot(
    base_llm,
    aes(
      x = FLOPs..1E21., y = MMLU, # nolint: object_usage_linter.
      color = Model.Family # nolint: object_usage_linter.
    )
  ) +
    geom_point() + # Adds colored points based on Model.Family
    labs(
      x = "FLOPs (1E21)", y = "MMLU Scores",
      title = "Relationship between FLOPs and MMLU by Model.Family"
    ) +
    theme_minimal() # Applies a minimal theme
}

main()
