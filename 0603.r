### Libraries, if not yet installed use install.packages
library(lme4)
library(dplyr)

### DATA PREP/EXPLROATION
## Read in the data
base_llm_benchmark_eval <- read.csv("~/git/predicting-capabilities/ObsScaling/eval_results/base_llm_benchmark_eval.csv")
base_llm_emergent_capability_eval <- read.csv("~/git/predicting-capabilities/ObsScaling/eval_results/base_llm_emergent_capability_eval.csv")

## Split first column by /
base_llm_benchmark_eval$Repo <- sapply(strsplit(as.character(base_llm_benchmark_eval$Model), "/"), "[", 1)
base_llm_benchmark_eval$Model <- sapply(strsplit(as.character(base_llm_benchmark_eval$Model), "/"), "[", 2)

base_llm_emergent_capability_eval$Repo <- sapply(strsplit(as.character(base_llm_emergent_capability_eval$Model), "/"), "[", 1)
base_llm_emergent_capability_eval$Model <- sapply(strsplit(as.character(base_llm_emergent_capability_eval$Model), "/"), "[", 2)

## For consistency, convert model & model family names to uppercase
base_llm_benchmark_eval$Model <- toupper(base_llm_benchmark_eval$Model)
base_llm_benchmark_eval$Model.Family <- toupper(base_llm_benchmark_eval$Model.Family)

base_llm_emergent_capability_eval$Model <- toupper(base_llm_emergent_capability_eval$Model)

## Merge datasets by Model, all rows should be included. Suffix "benchmark" and "emergent"
base_llm <- merge(base_llm_benchmark_eval, base_llm_emergent_capability_eval, by = "Model", all = TRUE, suffixes = c("_benchmark", "_emergent"))
## Convert character columns to factors to use the columns when modeling with sapply
base_llm[sapply(base_llm, is.character)] <- lapply(base_llm[sapply(base_llm, is.character)], as.factor)

## Summary stats
summary(base_llm)

## If you want to make a plot like I shared to group, including emergent benchmarks
# install.packages("PerformanceAnalytics")
# library(PerformanceAnalytics)
## Correlations of numeric  columns
# PerformanceAnalytics::chart.Correlation(base_llm[, sapply(base_llm, is.numeric)], histogram = TRUE, method = "pearson")
## You can check help for more options, but basically first argument = data (probably should manipulate this to zoom in to specific relationships)

## Just removing rows with NAs for now:
# Remove rows with missing values in the relevant columns
base_llm_clean <- base_llm[complete.cases(base_llm[c("arithmetic_3ds_2_acc", "FLOPs..1E21.", "Model.Family", "GSM8K")]), ]

## Get n-1 score
## Sort the data frame by Model.Family and FLOPs
base_llm_clean <- base_llm_clean %>%
  arrange(Model.Family, FLOPs..1E21.)
## Function to find n-1 benchmark score of the smaller model
get_minus_1_normalized <- function(model_index, data, benchmark_column) {
  current_family <- data$Model.Family[model_index]
  current_flops <- data$FLOPs..1E21.[model_index]
  current_score <- data[[benchmark_column]][model_index]
  # Filter to smaller than current model_index flops
  smaller_models <- data %>%
    filter(FLOPs..1E21. < current_flops) %>%
    arrange(desc(FLOPs..1E21.))
  # First, try to locate the closest smaller model in the same family
  closest_within_family <- smaller_models %>%
    filter(Model.Family == current_family) %>%
    slice(1)  # Uses slice to get the first model, which has the max FLOPs after arranging in descending order
  # Second, for use if no model in family is found, use the closest global model
  closest_global <- smaller_models %>%
    slice(1)
  # Determine the correct model source (within family or global) or fallback to current if no smaller exists
  if (nrow(closest_within_family) > 0) {
    target_model <- closest_within_family
  } else if (nrow(closest_global) > 0) {
    target_model <- closest_global
  } else {
    # Fallback to current model's values if no smaller models exist
    target_model <- tibble(`FLOPs..1E21.` = current_flops, !!benchmark_column := current_score)
  }
  # Calculate and return the normalized score directly
  minus_1_flops <- target_model$FLOPs..1E21.[1]
  minus_1_score <- target_model[[benchmark_column]][1]
  # Final normalized score calculation
  return(minus_1_score * current_flops / minus_1_flops)
}
## Apply the function to create 'GSM8K_minus_1', specifying 'GSM8K' as the benchmark_column to use
# base_llm_clean$GSM8K_minus_1 <- sapply(1:nrow(base_llm_clean), function(idx) {
#  get_minus_1(idx, base_llm_clean, "GSM8K")
# })

# Define non-benchmark columns
non_benchmark_columns <- c("Model", "Model.Family", "Model.Size..B.", "Pretraining.Data.Size..T.", "FLOPs..1E21.", "Repo_benchmark", "Repo_emergent")
# Identify benchmark columns by exclusion
benchmark_columns <- setdiff(names(base_llm_clean), non_benchmark_columns)
# Apply the 'get_minus_1_normalized' function across all benchmarks dynamically
for (benchmark_column in benchmark_columns) {
  new_column_name <- paste(benchmark_column, "minus_1", sep = "_")
  base_llm_clean[[new_column_name]] <- sapply(1:nrow(base_llm_clean), function(idx) {
    get_minus_1_normalized(idx, base_llm_clean, benchmark_column)
  })
}
# 63 rows, 39 cols 06/07

## Relationship Check
plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$GSM8K + 0.01)) # just eyeballing making it more linear for now
plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$FLOPs..1E21.))
plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$GSM8K_minus_1 + 0.01))

### MODELING
## Example of how to fit the mixed effects model
# y ~ x
# y ~ (x1/x2)
# x1 * x2
mixed_model <- lmer(arithmetic_3ds_2_acc ~ log(FLOPs..1E21.) + (1|Model.Family) + log(GSM8K_minus_1 + 0.01), data = base_llm_clean, REML = FALSE)
# Logistic regression 
# slope as well as intercept
# grid search of all possible predictive models
## Summarize the model
summary(mixed_model)
mixed_model
