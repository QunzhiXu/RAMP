#!/usr/bin/env Rscript

# Pre-specified joint-burden survival test for the locked TIGIT/CD39 state.
#
# This script deliberately evaluates only two continuous patient-level scores:
#   1. mean(within-patient state rank * positive spatial exposure)
#   2. niche fraction * mean(within-patient state rank among exposed CD8 cells)
#
# Both are computed before opening survival and are tested (i) alone and
# (ii) together with the original signed spatial niche score.  No thresholds,
# cutpoints, patient subsets, or marker combinations are searched.

suppressPackageStartupMessages({
  library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve script path")
script_path <- gsub("~\\+~", " ", sub("^--file=", "", script_arg))
here <- dirname(normalizePath(script_path))
input_path <- file.path(here, "output", "haist70_pt_cd8_tigit_cd39_return.csv.gz")
survival_path <- file.path(here, "output", "survival_analysis_input.csv")
out <- file.path(here, "joint_burden")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

zscore <- function(x) {
  sigma <- sd(x)
  if (!is.finite(sigma) || sigma <= 0) stop("Cannot standardize constant feature")
  as.numeric((x - mean(x)) / sigma)
}

cells <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
surv <- read.csv(survival_path, stringsAsFactors = FALSE, check.names = FALSE)
cells$patient <- as.character(cells$patient)
surv$patient <- as.character(surv$patient)

required_cells <- c(
  "patient", "spatial_y", "predicted_pair_rank_within_patient"
)
if (!all(required_cells %in% colnames(cells))) stop("Cell input lacks required fields")
if (nrow(surv) != 70L || anyDuplicated(surv$patient) || sum(surv$os_event) != 23L) {
  stop("Expected the locked all-70 survival cohort with 23 deaths")
}

# Feature construction is completed without consulting OS or event status.
split_cells <- split(cells, cells$patient)
feature_rows <- lapply(names(split_cells), function(patient) {
  d <- split_cells[[patient]]
  state_rank <- as.numeric(d$predicted_pair_rank_within_patient)
  positive_exposure <- pmax(as.numeric(d$spatial_y), 0)
  exposed <- positive_exposure > 0
  if (!any(exposed)) stop(paste("No exposed PT-CD8 cells for patient", patient))
  data.frame(
    patient = patient,
    pt_cd8_cells = nrow(d),
    exposed_cells = sum(exposed),
    positive_exposure_mean = mean(positive_exposure),
    niche_fraction = mean(exposed),
    exposed_state_rank_mean = mean(state_rank[exposed]),
    # Intensity-aware joint burden: high-state cells contribute in proportion
    # to their positive, ROI-adjusted CAF/macrophage exposure.
    joint_intensity_burden = mean(state_rank * positive_exposure),
    # Extent-aware joint burden: average state among exposed cells multiplied
    # by the fraction of all PT-CD8 cells that are exposed.
    joint_exposed_state_burden = mean(exposed) * mean(state_rank[exposed]),
    stringsAsFactors = FALSE
  )
})
features <- do.call(rbind, feature_rows)
row.names(features) <- NULL

if (nrow(features) != 70L || anyDuplicated(features$patient)) {
  stop("Expected one joint-burden row for each of 70 patients")
}
if (!setequal(features$patient, surv$patient)) {
  stop("Joint-burden patients differ from the locked survival cohort")
}

# Merge only after outcome-blind patient features have been finalized.
dat <- merge(
  features,
  surv[, c("patient", "OS", "os_event", "patient_spatial_niche_score")],
  by = "patient", all.x = TRUE, sort = TRUE
)
dat$joint_intensity_z <- zscore(dat$joint_intensity_burden)
dat$joint_exposed_state_z <- zscore(dat$joint_exposed_state_burden)
dat$direct_spatial_niche_z <- zscore(dat$patient_spatial_niche_score)
write.csv(dat, file.path(out, "patient_joint_burden_features.csv"), row.names = FALSE)

fit_direct <- coxph(
  Surv(OS, os_event) ~ direct_spatial_niche_z,
  data = dat, ties = "efron", x = TRUE
)
fit_intensity <- coxph(
  Surv(OS, os_event) ~ joint_intensity_z,
  data = dat, ties = "efron", x = TRUE
)
fit_intensity_adjusted <- coxph(
  Surv(OS, os_event) ~ direct_spatial_niche_z + joint_intensity_z,
  data = dat, ties = "efron", x = TRUE
)
fit_exposed <- coxph(
  Surv(OS, os_event) ~ joint_exposed_state_z,
  data = dat, ties = "efron", x = TRUE
)
fit_exposed_adjusted <- coxph(
  Surv(OS, os_event) ~ direct_spatial_niche_z + joint_exposed_state_z,
  data = dat, ties = "efron", x = TRUE
)

extract_term <- function(fit, term, model, definition) {
  sm <- summary(fit)
  data.frame(
    model = model,
    term = term,
    definition = definition,
    patients = nrow(dat),
    deaths = sum(dat$os_event),
    hazard_ratio_per_sd = unname(sm$conf.int[term, "exp(coef)"]),
    ci95_low = unname(sm$conf.int[term, "lower .95"]),
    ci95_high = unname(sm$conf.int[term, "upper .95"]),
    wald_p = unname(sm$coefficients[term, "Pr(>|z|)"]),
    concordance = unname(sm$concordance[1]),
    stringsAsFactors = FALSE
  )
}

definition_direct <- "Mean signed ROI-adjusted CAF/macrophage exposure across all PT-CD8 cells"
definition_intensity <- "Mean(within-patient TIGIT/CD39 state rank * max(ROI-adjusted CAF/macrophage exposure, 0))"
definition_exposed <- "Fraction spatially exposed * mean within-patient TIGIT/CD39 state rank among exposed PT-CD8 cells"

results <- rbind(
  extract_term(fit_direct, "direct_spatial_niche_z", "Direct spatial niche only", definition_direct),
  extract_term(fit_intensity, "joint_intensity_z", "Joint intensity burden only", definition_intensity),
  extract_term(fit_intensity_adjusted, "direct_spatial_niche_z", "Direct spatial + joint intensity burden", definition_direct),
  extract_term(fit_intensity_adjusted, "joint_intensity_z", "Direct spatial + joint intensity burden", definition_intensity),
  extract_term(fit_exposed, "joint_exposed_state_z", "Joint exposed-state burden only", definition_exposed),
  extract_term(fit_exposed_adjusted, "direct_spatial_niche_z", "Direct spatial + joint exposed-state burden", definition_direct),
  extract_term(fit_exposed_adjusted, "joint_exposed_state_z", "Direct spatial + joint exposed-state burden", definition_exposed)
)
write.csv(results, file.path(out, "cox_joint_burden_models.csv"), row.names = FALSE)

lrt_intensity <- anova(fit_direct, fit_intensity_adjusted, test = "LRT")
lrt_exposed <- anova(fit_direct, fit_exposed_adjusted, test = "LRT")
incremental <- data.frame(
  added_term = c("joint_intensity_z", "joint_exposed_state_z"),
  base_model = "Direct spatial niche only",
  likelihood_ratio_chisq = c(lrt_intensity[2, "Chisq"], lrt_exposed[2, "Chisq"]),
  df = c(lrt_intensity[2, "Df"], lrt_exposed[2, "Df"]),
  likelihood_ratio_p = c(lrt_intensity[2, "Pr(>|Chi|)"], lrt_exposed[2, "Pr(>|Chi|)"]),
  stringsAsFactors = FALSE
)
write.csv(incremental, file.path(out, "incremental_likelihood_ratio_tests.csv"), row.names = FALSE)

get_row <- function(model, term) {
  results[results$model == model & results$term == term, , drop = FALSE]
}
direct <- get_row("Direct spatial niche only", "direct_spatial_niche_z")
intensity <- get_row("Joint intensity burden only", "joint_intensity_z")
intensity_adj <- get_row("Direct spatial + joint intensity burden", "joint_intensity_z")
exposed <- get_row("Joint exposed-state burden only", "joint_exposed_state_z")
exposed_adj <- get_row("Direct spatial + joint exposed-state burden", "joint_exposed_state_z")

line <- function(label, row) {
  sprintf(
    "%s: HR=%.6f per SD (95%% CI %.6f-%.6f), Wald P=%.8g",
    label, row$hazard_ratio_per_sd, row$ci95_low, row$ci95_high, row$wald_p
  )
}
summary_lines <- c(
  "PRE-SPECIFIED TIGIT/CD39 x SPATIAL-EXPOSURE JOINT-BURDEN TEST",
  "Cohort: all 70 PT70 patients; 23 deaths; continuous Cox models; no cutpoint or subset search.",
  line("Direct signed spatial niche score", direct),
  line("Joint intensity burden, univariate", intensity),
  line("Joint intensity burden, adjusted for direct spatial score", intensity_adj),
  sprintf("Increment beyond direct spatial score, intensity-burden LRT P=%.8g", incremental$likelihood_ratio_p[1]),
  line("Joint exposed-state burden, univariate", exposed),
  line("Joint exposed-state burden, adjusted for direct spatial score", exposed_adj),
  sprintf("Increment beyond direct spatial score, exposed-state-burden LRT P=%.8g", incremental$likelihood_ratio_p[2])
)
writeLines(summary_lines, file.path(out, "SUMMARY.txt"))
cat(paste(summary_lines, collapse = "\n"), "\n")

