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
base_llm_emergent_capability_eval$Model.Family <- toupper(base_llm_emergent_capability_eval$Model.Family)

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

## Get n-1 score (we can talk about we'd want to define this)
## Sort the data frame by Model.Family and FLOPs
base_llm_clean <- base_llm_clean %>%
  arrange(Model.Family, FLOPs..1E21.)
## Function to find GSM8K score of the smaller model # TODO: generalize to other benchmarks
get_smaller_gsm8k <- function(model_index, data) {
  current_family <- data$Model.Family[model_index]
  current_flops <- data$FLOPs..1E21.[model_index]
  
  # Get models in the same family with smaller FLOPs
  family_subset <- data %>%
    filter(Model.Family == current_family & FLOPs..1E21. < current_flops)
  
  if (nrow(family_subset) > 0) {
    return(family_subset$GSM8K[which.max(family_subset$FLOPs..1E21.)])
  } else {
    # Get the overall smaller model
    smaller_subset <- data %>%
      filter(FLOPs..1E21. < current_flops)
    
    if (nrow(smaller_subset) > 0) {
      return(smaller_subset$GSM8K[which.max(smaller_subset$FLOPs..1E21.)])
    } else {
      return(data$GSM8K[model_index])
    }
  }
}
## Apply the function to create GSM8K-1
base_llm_clean$GSM8K_minus_1 <- sapply(1:nrow(base_llm_clean), get_smaller_gsm8k, data = base_llm_clean)

## Relationship Check
plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$GSM8K + 0.01)) # just eyeballing making it more linear for now
plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$FLOPs..1E21.))
plot(base_llm_clean$arithmetic_3ds_2_acc, log10(base_llm_clean$GSM8K_minus_1 + 0.01))

### MODELING
## Example of how to fit the mixed effects model
mixed_model <- lmer(arithmetic_3ds_2_acc ~ log10(FLOPs..1E21.) + (1|Model.Family) + log10(GSM8K + 0.01) + log10(GSM8K_minus_1 + 0.01), data = base_llm_clean, REML = FALSE)

## Summarize the model
summary(mixed_model)
