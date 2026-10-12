#!/usr/bin/env Rscript

# Recompute the observed Tc.ae / Ki67 / tumor-distance patient feature at
# 20 and 30 um from the frozen cell-level prediction table, then compare the
# resulting survival analyses with the frozen story_cd8 30-um analysis.

suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
here <- dirname(script_path)
story_out <- normalizePath(
  file.path(here, "../../../../../story_cd8/survival_analysis/output"),
  mustWork = TRUE
)

cell_path <- file.path(story_out, "full_cell_predictions.csv.gz")
frozen_patient_path <- file.path(story_out, "patient_level_features.csv")
frozen_result_path <- file.path(story_out, "survival_and_response_results.csv")

cells <- read.csv(gzfile(cell_path), check.names = FALSE, stringsAsFactors = FALSE)
cells <- cells[cells$dataset == "Moldoveanu IMC", , drop = FALSE]
cells$prob.Ki67 <- as.numeric(cells$prob.Ki67)
cells$distance_to_nearest_tumor <- as.numeric(cells$distance_to_nearest_tumor)
cells$Known_Deceased <- as.integer(cells$Known_Deceased)
cells$Survival_from_Rx_Start <- as.numeric(cells$Survival_from_Rx_Start)

first_unique <- function(x, field, patient) {
  values <- unique(x[!is.na(x)])
  if (length(values) != 1L) {
    stop(sprintf("%s: expected one nonmissing %s value, found %d", patient, field, length(values)))
  }
  values[[1]]
}

summarize_patient <- function(patient) {
  group <- cells[cells$donor == patient, , drop = FALSE]
  tcae <- group$Cluster.v2 == "Tc.ae"
  ki67 <- !is.na(group$prob.Ki67) & group$prob.Ki67 >= 0.5
  target_base <- tcae & ki67
  d20 <- !is.na(group$distance_to_nearest_tumor) & group$distance_to_nearest_tumor <= 20
  d30 <- !is.na(group$distance_to_nearest_tumor) & group$distance_to_nearest_tumor <= 30
  n_all <- nrow(group)
  n_tcae <- sum(tcae)
  n20 <- sum(target_base & d20)
  n30 <- sum(target_base & d30)
  data.frame(
    patient = patient,
    n_all_cells = n_all,
    n_tcae = n_tcae,
    n_observed_ki67_tcae = sum(target_base),
    n_observed_20um = n20,
    n_observed_30um = n30,
    observed_20um_per_all = n20 / n_all,
    observed_30um_per_all = n30 / n_all,
    observed_20um_fraction_within_tcae = if (n_tcae > 0) n20 / n_tcae else NA_real_,
    observed_30um_fraction_within_tcae = if (n_tcae > 0) n30 / n_tcae else NA_real_,
    Known_Deceased = first_unique(group$Known_Deceased, "Known_Deceased", patient),
    Survival_from_Rx_Start = first_unique(
      group$Survival_from_Rx_Start, "Survival_from_Rx_Start", patient
    ),
    Treatment = first_unique(group$Treatment, "Treatment", patient),
    stringsAsFactors = FALSE
  )
}

patients <- sort(unique(cells$donor))
features <- do.call(rbind, lapply(patients, summarize_patient))
rownames(features) <- NULL

frozen_patients <- read.csv(
  frozen_patient_path, check.names = FALSE, stringsAsFactors = FALSE
)
frozen_keep <- frozen_patients[, c(
  "patient", "cd8_per_all", "observed_ki67_contact30_tcae_per_all",
  "observed_ki67_contact30_fraction_within_tcae"
)]
names(frozen_keep)[names(frozen_keep) == "observed_ki67_contact30_tcae_per_all"] <-
  "frozen_observed_30um_per_all"
names(frozen_keep)[names(frozen_keep) == "observed_ki67_contact30_fraction_within_tcae"] <-
  "frozen_observed_30um_fraction_within_tcae"
features <- merge(features, frozen_keep, by = "patient", all.x = TRUE, sort = TRUE)

if (nrow(features) != 30L) stop("Expected 30 outcome-bearing patients")
if (any(is.na(features$frozen_observed_30um_per_all))) {
  stop("Frozen 30-um feature is missing after patient join")
}

safe_cox <- function(formula, frame) {
  fit <- tryCatch(
    coxph(formula, data = frame, ties = "efron"),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    return(list(hr = NA_real_, lower95 = NA_real_, upper95 = NA_real_, p = NA_real_))
  }
  s <- summary(fit)
  list(
    hr = unname(s$conf.int[1, "exp(coef)"]),
    lower95 = unname(s$conf.int[1, "lower .95"]),
    upper95 = unname(s$conf.int[1, "upper .95"]),
    p = unname(s$coefficients[1, "Pr(>|z|)"])
  )
}

analyze_feature <- function(score_col, threshold_um) {
  frame <- features[
    is.finite(features[[score_col]]) &
      is.finite(features$Survival_from_Rx_Start) &
      is.finite(features$Known_Deceased),
    , drop = FALSE
  ]
  frame$score <- frame[[score_col]]
  frame$x <- as.numeric(scale(frame$score))
  frame$cd8_z <- as.numeric(scale(frame$cd8_per_all))
  cutoff <- median(frame$score)
  frame$group <- factor(
    ifelse(frame$score >= cutoff, "High", "Low"),
    levels = c("Low", "High")
  )
  survival_formula <- Surv(Survival_from_Rx_Start, Known_Deceased) ~ group
  lr <- survdiff(survival_formula, data = frame)
  fit <- survfit(survival_formula, data = frame)
  fit_table <- summary(fit)$table
  median_by_group <- setNames(
    as.numeric(fit_table[, "median"]),
    sub("^group=", "", rownames(fit_table))
  )
  continuous <- safe_cox(
    Surv(Survival_from_Rx_Start, Known_Deceased) ~ x, frame
  )
  adjusted_cd8 <- safe_cox(
    Surv(Survival_from_Rx_Start, Known_Deceased) ~ x + cd8_z, frame
  )
  adjusted_cd8_treatment <- safe_cox(
    Surv(Survival_from_Rx_Start, Known_Deceased) ~ x + cd8_z + strata(Treatment),
    frame
  )
  high <- frame$group == "High"
  low <- frame$group == "Low"
  list(
    threshold_um = threshold_um,
    definition = sprintf(
      "Tc.ae AND prob.Ki67 >= 0.5 AND nearest melanoma distance <= %d um",
      threshold_um
    ),
    patient_score = "observed target cells divided by all segmented cells",
    n = nrow(frame),
    events = sum(frame$Known_Deceased),
    target_cells = sum(
      if (threshold_um == 20) frame$n_observed_20um else frame$n_observed_30um
    ),
    patients_with_target = sum(
      if (threshold_um == 20) frame$n_observed_20um > 0 else frame$n_observed_30um > 0
    ),
    median_cutoff = cutoff,
    groups = list(
      Low = list(
        n = sum(low),
        events = sum(frame$Known_Deceased[low]),
        median_os_days = unname(median_by_group[["Low"]]),
        patients = sort(frame$patient[low])
      ),
      High = list(
        n = sum(high),
        events = sum(frame$Known_Deceased[high]),
        median_os_days = unname(median_by_group[["High"]]),
        patients = sort(frame$patient[high])
      )
    ),
    logrank_p = unname(1 - pchisq(lr$chisq, df = 1)),
    cox_continuous_per_sd = continuous,
    cox_adjusted_for_total_cd8 = adjusted_cd8,
    cox_adjusted_for_total_cd8_and_stratified_by_treatment = adjusted_cd8_treatment
  )
}

result20 <- analyze_feature("observed_20um_per_all", 20)
result30 <- analyze_feature("observed_30um_per_all", 30)

frozen_results <- read.csv(
  frozen_result_path, check.names = FALSE, stringsAsFactors = FALSE
)
frozen30 <- frozen_results[
  frozen_results$feature == "observed_ki67_contact30_tcae_per_all",
  , drop = FALSE
]
if (nrow(frozen30) != 1L) stop("Expected one frozen observed 30-um result row")

features$group_20um <- ifelse(
  features$observed_20um_per_all >= result20$median_cutoff, "High", "Low"
)
features$group_30um <- ifelse(
  features$observed_30um_per_all >= result30$median_cutoff, "High", "Low"
)
features$score_delta_20_minus_30 <-
  features$observed_20um_per_all - features$observed_30um_per_all

max_feature_diff <- max(abs(
  features$observed_30um_per_all - features$frozen_observed_30um_per_all
))
fraction_diff <- abs(
  features$observed_30um_fraction_within_tcae -
    features$frozen_observed_30um_fraction_within_tcae
)
max_fraction_diff <- if (all(is.na(fraction_diff))) NA_real_ else max(fraction_diff, na.rm = TRUE)

audit <- list(
  sources = list(
    cell_level = normalizePath(cell_path),
    frozen_patient_features = normalizePath(frozen_patient_path),
    frozen_survival_results = normalizePath(frozen_result_path)
  ),
  definition_common = "Observed provider markers only; no projected PD-1/CD69 feature is used",
  patient_score = "count / all segmented cells in patient sample",
  thresholds_um = c(20, 30),
  n_moldoveanu_cells = nrow(cells),
  n_patients = nrow(features),
  events = sum(features$Known_Deceased),
  feature_comparison = list(
    total_target_cells_20um = sum(features$n_observed_20um),
    total_target_cells_30um = sum(features$n_observed_30um),
    retained_fraction_20_vs_30 = sum(features$n_observed_20um) / sum(features$n_observed_30um),
    patients_with_target_20um = sum(features$n_observed_20um > 0),
    patients_with_target_30um = sum(features$n_observed_30um > 0),
    pearson_score_correlation = cor(
      features$observed_20um_per_all, features$observed_30um_per_all,
      method = "pearson"
    ),
    spearman_score_correlation = cor(
      features$observed_20um_per_all, features$observed_30um_per_all,
      method = "spearman"
    ),
    median_group_disagreements = sum(features$group_20um != features$group_30um)
  ),
  frozen_30um_reproduction = list(
    max_abs_patient_score_difference = max_feature_diff,
    max_abs_fraction_within_tcae_difference = max_fraction_diff,
    recomputed_logrank_p = result30$logrank_p,
    frozen_logrank_p = frozen30$logrank_p[[1]],
    abs_logrank_p_difference = abs(result30$logrank_p - frozen30$logrank_p[[1]]),
    recomputed_cox_hr_per_sd = result30$cox_continuous_per_sd$hr,
    frozen_cox_hr_per_sd = frozen30$cox_hr_per_sd[[1]],
    abs_cox_hr_difference = abs(
      result30$cox_continuous_per_sd$hr - frozen30$cox_hr_per_sd[[1]]
    ),
    recomputed_cox_p = result30$cox_continuous_per_sd$p,
    frozen_cox_p = frozen30$cox_p[[1]],
    abs_cox_p_difference = abs(
      result30$cox_continuous_per_sd$p - frozen30$cox_p[[1]]
    )
  )
)

if (max_feature_diff > 1e-14) {
  stop(sprintf("Recomputed 30-um patient feature differs from frozen table: %.3g", max_feature_diff))
}
if (audit$feature_comparison$median_group_disagreements != 0L) {
  stop("20- and 30-um median groups unexpectedly differ")
}
if (audit$frozen_30um_reproduction$abs_logrank_p_difference > 1e-12) {
  stop("Recomputed 30-um log-rank result differs from frozen result")
}

write_json(
  features,
  file.path(here, "patient_features_20_vs_30.json"),
  dataframe = "rows", pretty = TRUE, digits = 16, na = "null"
)
write_json(
  list(results = list(`20um` = result20, `30um` = result30), audit = audit),
  file.path(here, "survival_results_20_vs_30.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 16, na = "null"
)

fmt <- function(x, digits = 4) formatC(x, digits = digits, format = "g")
percent <- function(x, digits = 1) paste0(formatC(100 * x, digits = digits, format = "f"), "%")

report <- c(
  "# Observed 20 µm versus 30 µm survival sensitivity",
  "",
  "## Bottom line",
  "",
  sprintf(
    paste0(
      "The 20 µm definition retains %d/%d observed composite cells (%s) ",
      "but produces exactly the same 15-high/15-low patient split as 30 µm. "
    ),
    audit$feature_comparison$total_target_cells_20um,
    audit$feature_comparison$total_target_cells_30um,
    percent(audit$feature_comparison$retained_fraction_20_vs_30)
  ),
  sprintf(
    paste0(
      "Therefore its median-split Kaplan–Meier curves are identical: median OS ",
      "%s versus %s days (high versus low), log-rank P = %s for both thresholds."
    ),
    fmt(result20$groups$High$median_os_days),
    fmt(result20$groups$Low$median_os_days),
    fmt(result20$logrank_p, 6)
  ),
  "",
  "The continuous patient score changes, so the continuous Cox estimates are not identical.",
  "",
  "## Survival comparison",
  "",
  "| Distance | Composite cells | Patients with ≥1 | Median cutoff (per all cells) | High/low events | Median OS high/low (days) | Log-rank P | Cox HR per SD (95% CI) | Cox P |",
  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  sprintf(
    "| 20 µm | %d | %d | %s | %d/%d | %s/%s | %s | %s (%s–%s) | %s |",
    result20$target_cells, result20$patients_with_target,
    fmt(result20$median_cutoff, 7),
    result20$groups$High$events, result20$groups$Low$events,
    fmt(result20$groups$High$median_os_days), fmt(result20$groups$Low$median_os_days),
    fmt(result20$logrank_p, 6),
    fmt(result20$cox_continuous_per_sd$hr),
    fmt(result20$cox_continuous_per_sd$lower95),
    fmt(result20$cox_continuous_per_sd$upper95),
    fmt(result20$cox_continuous_per_sd$p, 5)
  ),
  sprintf(
    "| 30 µm | %d | %d | %s | %d/%d | %s/%s | %s | %s (%s–%s) | %s |",
    result30$target_cells, result30$patients_with_target,
    fmt(result30$median_cutoff, 7),
    result30$groups$High$events, result30$groups$Low$events,
    fmt(result30$groups$High$median_os_days), fmt(result30$groups$Low$median_os_days),
    fmt(result30$logrank_p, 6),
    fmt(result30$cox_continuous_per_sd$hr),
    fmt(result30$cox_continuous_per_sd$lower95),
    fmt(result30$cox_continuous_per_sd$upper95),
    fmt(result30$cox_continuous_per_sd$p, 5)
  ),
  "",
  "## Patient-feature agreement",
  "",
  sprintf(
    "- Pearson correlation of 20 versus 30 µm patient scores: %s; Spearman rho: %s.",
    fmt(audit$feature_comparison$pearson_score_correlation, 5),
    fmt(audit$feature_comparison$spearman_score_correlation, 5)
  ),
  sprintf(
    "- Median-group disagreements: %d/30.",
    audit$feature_comparison$median_group_disagreements
  ),
  sprintf(
    "- Patients with a nonzero score: %d at 20 µm and %d at 30 µm.",
    audit$feature_comparison$patients_with_target_20um,
    audit$feature_comparison$patients_with_target_30um
  ),
  "",
  "## Frozen 30 µm reproduction audit",
  "",
  sprintf(
    "- Maximum absolute patient-score difference versus the frozen story table: %.3g.",
    audit$frozen_30um_reproduction$max_abs_patient_score_difference
  ),
  sprintf(
    "- Recomputed versus frozen log-rank P: %.16g versus %.16g (absolute difference %.3g).",
    audit$frozen_30um_reproduction$recomputed_logrank_p,
    audit$frozen_30um_reproduction$frozen_logrank_p,
    audit$frozen_30um_reproduction$abs_logrank_p_difference
  ),
  sprintf(
    "- Recomputed versus frozen continuous Cox HR per SD: %.16g versus %.16g; P: %.16g versus %.16g.",
    audit$frozen_30um_reproduction$recomputed_cox_hr_per_sd,
    audit$frozen_30um_reproduction$frozen_cox_hr_per_sd,
    audit$frozen_30um_reproduction$recomputed_cox_p,
    audit$frozen_30um_reproduction$frozen_cox_p
  ),
  "",
  "## Interpretation",
  "",
  paste0(
    "For the prespecified median-split KM/log-rank analysis, 20 µm reproduces the 30 µm result ",
    "exactly because no patient crosses the median-group boundary. This is a threshold-sensitivity ",
    "result, not an independent survival replication: both definitions use the same 30 patients and outcomes."
  ),
  paste0(
    "The continuous Cox model should be reported separately because tightening the threshold changes ",
    "the score magnitudes and its HR/P value; it cannot be described as numerically identical to 30 µm."
  ),
  "",
  "## Reproducibility",
  "",
  "Run `Rscript run_observed20_sensitivity.R` from any working directory. The script reads the frozen story outputs but writes only in this sensitivity directory."
)
writeLines(report, file.path(here, "README.md"), useBytes = TRUE)

cat(paste(report, collapse = "\n"), "\n")
