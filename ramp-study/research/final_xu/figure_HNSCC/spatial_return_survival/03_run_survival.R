#!/usr/bin/env Rscript

# Survival analysis for the fixed TIGIT/CD39 return in the complete Haist
# primary-tumour cohort.
#
# The molecular/spatial patient features are constructed upstream without
# opening survival outcomes.  This script then joins those fixed features to
# the provider clinical table and fits the pre-specified survival models.
#
# Primary score:
#   mean signed ROI-adjusted CAF/macrophage exposure among each patient's
#   within-patient top-40% returned TIGIT/CD39-state PT-CD8 cells.
#
# The continuous per-SD Cox model in all 74 complete-OS patients is primary.
# The median split is used only to visualize the same fixed continuous score.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve this script's path")
script_path <- gsub("~\\+~", " ", sub("^--file=", "", script_arg))
here <- dirname(normalizePath(script_path))

feature_path <- file.path(
  here, "output", "haist74_tigit_cd39_top40_patient_features.csv"
)
clinical_path <- file.path(
  here, "..", "sampled_data", "haist_codex", "complete",
  "HNSCCdiscovery_clinmetadata.csv"
)
out <- file.path(here, "output")
figdir <- file.path(here, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

required_files <- c(feature_path, clinical_path)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(paste("Missing required input:", paste(missing_files, collapse = ", ")))
}

# Guard the formal analysis against accidentally acquiring a dependency on an
# exploratory directory through a symlink or a future path edit.
normalized_inputs <- normalizePath(required_files, mustWork = TRUE)
if (any(grepl("(^|/)(test|test0827)(/|$)", normalized_inputs))) {
  stop("Formal survival analysis cannot read an input below a test directory")
}

zscore <- function(x) {
  x <- as.numeric(x)
  if (any(!is.finite(x))) stop("Cannot standardize non-finite values")
  sigma <- sd(x)
  if (!is.finite(sigma) || sigma <= 0) {
    stop("Cannot standardize a constant feature")
  }
  as.numeric((x - mean(x)) / sigma)
}

format_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return(format(p, scientific = TRUE, digits = 2))
  sprintf("%.3f", p)
}

fit_model <- function(
  frame, formula, term, event_term, label, role, definition, cohort
) {
  variables <- all.vars(formula)
  missing_variables <- setdiff(variables, colnames(frame))
  if (length(missing_variables) > 0L) {
    stop(paste(
      "Model", label, "is missing variables:",
      paste(missing_variables, collapse = ", ")
    ))
  }

  d <- frame[complete.cases(frame[, variables, drop = FALSE]), , drop = FALSE]
  fit <- coxph(formula, data = d, ties = "efron", x = TRUE)
  sm <- summary(fit)
  if (!(term %in% rownames(sm$coefficients))) {
    stop(paste("Requested term", term, "is absent from model", label))
  }

  zph <- try(cox.zph(fit), silent = TRUE)
  feature_ph_p <- global_ph_p <- NA_real_
  if (!inherits(zph, "try-error")) {
    if (term %in% rownames(zph$table)) {
      feature_ph_p <- zph$table[term, "p"]
    }
    if ("GLOBAL" %in% rownames(zph$table)) {
      global_ph_p <- zph$table["GLOBAL", "p"]
    }
  }

  data.frame(
    model = label,
    role = role,
    cohort = cohort,
    definition = definition,
    patients = nrow(d),
    deaths = sum(d[[event_term]]),
    beta = unname(sm$coefficients[term, "coef"]),
    se = unname(sm$coefficients[term, "se(coef)"]),
    hazard_ratio_per_sd = unname(sm$conf.int[term, "exp(coef)"]),
    ci95_low = unname(sm$conf.int[term, "lower .95"]),
    ci95_high = unname(sm$conf.int[term, "upper .95"]),
    wald_p = unname(sm$coefficients[term, "Pr(>|z|)"]),
    feature_ph_p = feature_ph_p,
    global_ph_p = global_ph_p,
    concordance = unname(sm$concordance[1]),
    stringsAsFactors = FALSE
  )
}

features <- read.csv(
  feature_path, stringsAsFactors = FALSE, check.names = FALSE
)
clinical_roi <- read.csv(
  clinical_path, stringsAsFactors = FALSE, check.names = FALSE
)

required_feature_columns <- c(
  "patient", "pt_cd8_cells", "high_pair_mean_spatial_y",
  "patient_spatial_niche_score"
)
required_clinical_columns <- c(
  "CaseID", "unique_region", "OS", "OS_status", "Age", "HPV_status",
  "AJCC_stage"
)
missing_feature_columns <- setdiff(required_feature_columns, colnames(features))
missing_clinical_columns <- setdiff(
  required_clinical_columns, colnames(clinical_roi)
)
if (length(missing_feature_columns) > 0L) {
  stop(paste(
    "Patient feature table lacks required columns:",
    paste(missing_feature_columns, collapse = ", ")
  ))
}
if (length(missing_clinical_columns) > 0L) {
  stop(paste(
    "Clinical table lacks required columns:",
    paste(missing_clinical_columns, collapse = ", ")
  ))
}

features$patient <- as.character(features$patient)
clinical_roi$CaseID <- as.character(clinical_roi$CaseID)
if (nrow(features) != 74L || anyDuplicated(features$patient)) {
  stop("Expected one unique patient-feature row for each of 74 patients")
}
if (anyDuplicated(clinical_roi$unique_region)) {
  stop("Provider clinical table contains duplicated unique_region values")
}

# The provider clinical table is ROI-level.  Verify patient-level invariance
# before taking one row per patient, preventing ROI pseudoreplication.
clinical_fields <- c("OS", "OS_status", "Age", "HPV_status", "AJCC_stage")
for (column in clinical_fields) {
  inconsistent <- tapply(
    clinical_roi[[column]], clinical_roi$CaseID,
    function(x) length(unique(x[!is.na(x)])) > 1L
  )
  if (any(inconsistent)) {
    stop(paste("Clinical field varies within patient:", column))
  }
}
clinical <- clinical_roi[!duplicated(clinical_roi$CaseID), , drop = FALSE]
clinical$os_event <- unname(c("OS+" = 1L, "OS-" = 0L)[clinical$OS_status])
complete_os <- clinical[
  is.finite(clinical$OS) & !is.na(clinical$os_event), , drop = FALSE
]
if (
  nrow(complete_os) != 74L || anyDuplicated(complete_os$CaseID) ||
  sum(complete_os$os_event) != 24L
) {
  stop(sprintf(
    paste(
      "Provider clinical table should contain 74 unique complete-OS patients",
      "and 24 deaths; observed %d patients and %d deaths"
    ),
    nrow(complete_os), sum(complete_os$os_event)
  ))
}
if (!setequal(features$patient, complete_os$CaseID)) {
  missing_features <- setdiff(complete_os$CaseID, features$patient)
  extra_features <- setdiff(features$patient, complete_os$CaseID)
  stop(paste0(
    "Patient features do not equal the complete-OS clinical cohort. ",
    "Missing feature patients: ", paste(missing_features, collapse = ";"),
    ". Extra feature patients: ", paste(extra_features, collapse = ";")
  ))
}

dat <- merge(
  features,
  complete_os[, c(
    "CaseID", "OS", "OS_status", "os_event", "Age", "HPV_status",
    "AJCC_stage"
  )],
  by.x = "patient", by.y = "CaseID", all.x = TRUE, sort = TRUE
)
if (
  nrow(dat) != 74L || anyDuplicated(dat$patient) ||
  any(!is.finite(dat$OS)) || any(is.na(dat$os_event)) ||
  sum(dat$os_event) != 24L
) {
  stop(sprintf(
    "Expected 74 complete-OS patients and 24 deaths; observed %d and %d",
    nrow(dat), sum(dat$os_event, na.rm = TRUE)
  ))
}
for (column in c(
  "high_pair_mean_spatial_y", "patient_spatial_niche_score", "Age",
  "HPV_status", "AJCC_stage"
)) {
  if (any(!is.finite(as.numeric(dat[[column]])))) {
    stop(paste("Analysis variable contains non-finite values:", column))
  }
}

joint_definition <- paste(
  "Mean signed ROI-adjusted CAF/macrophage exposure among each patient's",
  "within-patient top-40% returned TIGIT/CD39-state PT-CD8 cells"
)
direct_definition <- paste(
  "Mean signed ROI-adjusted CAF/macrophage exposure across all PT-CD8 cells"
)

dat$high_pair_spatial_z <- zscore(dat$high_pair_mean_spatial_y)
dat$direct_spatial_niche_z <- zscore(dat$patient_spatial_niche_score)
dat$Age_z <- zscore(dat$Age)
dat$HPV_status_z <- zscore(dat$HPV_status)
dat$AJCC_stage_z <- zscore(dat$AJCC_stage)
dat$os_time_60 <- pmin(dat$OS, 60)
dat$os_event_60 <- as.integer(dat$os_event == 1L & dat$OS <= 60)

without_41403 <- dat[dat$patient != "41403", , drop = FALSE]
if (
  nrow(without_41403) != 73L ||
  sum(without_41403$os_event) != 23L
) {
  stop("Excluding Case 41403 should yield 73 patients and 23 deaths")
}
# Re-standardize in the deletion cohort so this sensitivity HR is per SD among
# the 73 patients actually entering that model.
without_41403$high_pair_spatial_z <- zscore(
  without_41403$high_pair_mean_spatial_y
)

primary <- fit_model(
  dat,
  Surv(OS, os_event) ~ high_pair_spatial_z,
  "high_pair_spatial_z", "os_event",
  "Primary: top-40% CD8 spatial exposure",
  "primary", joint_definition, "Complete OS cohort"
)
timing <- fit_model(
  dat,
  Surv(os_time_60, os_event_60) ~ high_pair_spatial_z,
  "high_pair_spatial_z", "os_event_60",
  "Timing sensitivity: 0-60 months",
  "timing sensitivity", joint_definition, "Complete OS cohort"
)
clinical_adjusted <- fit_model(
  dat,
  Surv(OS, os_event) ~ high_pair_spatial_z + Age_z + HPV_status_z + AJCC_stage_z,
  "high_pair_spatial_z", "os_event",
  "Fixed clinical adjustment",
  "clinical sensitivity", joint_definition, "Complete OS cohort"
)
spatial_adjusted <- fit_model(
  dat,
  Surv(OS, os_event) ~ high_pair_spatial_z + direct_spatial_niche_z,
  "high_pair_spatial_z", "os_event",
  "Joint score + direct spatial score",
  "spatial-burden sensitivity", joint_definition, "Complete OS cohort"
)
direct_reference <- fit_model(
  dat,
  Surv(OS, os_event) ~ direct_spatial_niche_z,
  "direct_spatial_niche_z", "os_event",
  "Direct spatial niche score",
  "direct spatial reference", direct_definition, "Complete OS cohort"
)
delete_41403 <- fit_model(
  without_41403,
  Surv(OS, os_event) ~ high_pair_spatial_z,
  "high_pair_spatial_z", "os_event",
  "Sensitivity: exclude Case 41403",
  "low-cell-count sensitivity", joint_definition,
  "Complete OS cohort excluding Case 41403"
)

models <- rbind(
  primary, timing, clinical_adjusted, spatial_adjusted, direct_reference,
  delete_41403
)
write.csv(
  models, file.path(out, "top40_survival_models.csv"), row.names = FALSE
)

# Median split is for visualization only; it is not used by any Cox model and
# is distinct from the within-patient top-40% cell-state definition.
cutpoint <- median(dat$high_pair_mean_spatial_y)
dat$joint_group <- factor(
  ifelse(dat$high_pair_mean_spatial_y >= cutpoint, "Higher", "Lower"),
  levels = c("Lower", "Higher")
)
if (any(table(dat$joint_group) == 0L)) stop("Median split produced an empty group")
write.csv(
  dat, file.path(out, "top40_survival_analysis_input.csv"), row.names = FALSE
)

risk_times <- c(0, 24, 48, 72, 96, 120)
km_rows <- list()
risk_rows <- list()
for (group_name in levels(dat$joint_group)) {
  local <- dat[dat$joint_group == group_name, , drop = FALSE]
  fit <- survfit(Surv(OS, os_event) ~ 1, data = local)
  km_rows[[length(km_rows) + 1L]] <- data.frame(
    group = group_name,
    time = c(0, fit$time),
    survival = c(1, fit$surv),
    stringsAsFactors = FALSE
  )
  risk <- summary(fit, times = risk_times, extend = TRUE)
  risk_rows[[length(risk_rows) + 1L]] <- data.frame(
    group = group_name,
    time = risk_times,
    n_risk = unname(risk$n.risk),
    stringsAsFactors = FALSE
  )
}
km <- do.call(rbind, km_rows)
risk <- do.call(rbind, risk_rows)
write.csv(
  km, file.path(out, "top40_median_split_km_curve.csv"), row.names = FALSE
)
write.csv(
  risk,
  file.path(out, "top40_median_split_number_at_risk.csv"),
  row.names = FALSE
)

model_line <- function(label, row) {
  sprintf(
    "%s: HR=%.6f per SD (95%% CI %.6f-%.6f), Wald P=%.8g; n=%d, deaths=%d",
    label, row$hazard_ratio_per_sd, row$ci95_low, row$ci95_high,
    row$wald_p, row$patients, row$deaths
  )
}
summary_lines <- c(
  "TOP-40% TIGIT/CD39 HIGH-STATE SPATIAL-EXPOSURE SURVIVAL ANALYSIS",
  paste(
    "Cohort: all 74 complete-OS Haist patients; 24 deaths; no minimum PT-CD8",
    "threshold. The continuous per-SD Cox model is primary; the median split",
    "is used only for Kaplan-Meier visualization."
  ),
  paste("Patient score:", joint_definition),
  sprintf("Visualization-only median cutpoint=%.10g", cutpoint),
  model_line("Primary complete-74 model", primary),
  model_line("0-60-month timing sensitivity", timing),
  model_line("Age/HPV/AJCC-adjusted sensitivity", clinical_adjusted),
  model_line("Adjusted for direct spatial score", spatial_adjusted),
  model_line("Direct spatial score reference", direct_reference),
  model_line("Exclude Case 41403 sensitivity", delete_41403),
  sprintf(
    paste(
      "Case 41403 contributes %d PT-CD8 cells and is retained in the primary",
      "complete-cohort model."
    ),
    dat$pt_cd8_cells[dat$patient == "41403"]
  )
)
writeLines(summary_lines, file.path(out, "TOP40_SURVIVAL_SUMMARY.txt"))

colors <- c("Lower" = "#2B6CB0", "Higher" = "#C43C39")
counts <- table(dat$joint_group)
p_km <- ggplot(km[km$time <= 120, ], aes(time, survival, color = group)) +
  geom_step(linewidth = 1.15, direction = "hv") +
  annotate(
    "label", x = 116, y = 0.98, hjust = 1, vjust = 1,
    label = sprintf(
      "Continuous Cox\nHR %.2f per SD\n95%% CI %.2f-%.2f; P=%s",
      primary$hazard_ratio_per_sd, primary$ci95_low, primary$ci95_high,
      format_p(primary$wald_p)
    ),
    size = 3.5, label.size = 0.2, fill = "white"
  ) +
  scale_color_manual(
    values = colors,
    breaks = c("Lower", "Higher"),
    labels = c(
      "Lower" = sprintf("Lower top-40%% score (n=%d)", counts[["Lower"]]),
      "Higher" = sprintf("Higher top-40%% score (n=%d)", counts[["Higher"]])
    )
  ) +
  scale_x_continuous(
    breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)
  ) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "Spatial exposure of returned TIGIT/CD39-high CD8 cells",
    subtitle = paste(
      "Complete OS cohort; median split shown only for visualization"
    ),
    x = NULL, y = "Overall survival probability", color = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "inside",
    legend.position.inside = c(0.30, 0.18),
    legend.background = element_rect(fill = "white", color = NA)
  )

p_risk <- ggplot(risk, aes(time, group, label = n_risk, color = group)) +
  geom_text(size = 3.4, fontface = "bold") +
  scale_color_manual(values = colors, guide = "none") +
  scale_x_continuous(
    breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)
  ) +
  labs(x = "Overall survival (months)", y = "At risk") +
  theme_classic(base_size = 9) +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(face = "bold")
  )

forest <- models
forest$model <- factor(forest$model, levels = rev(models$model))
forest$label <- sprintf(
  "HR %.2f (%.2f-%.2f), P=%s",
  forest$hazard_ratio_per_sd, forest$ci95_low, forest$ci95_high,
  vapply(forest$wald_p, format_p, character(1))
)
p_forest <- ggplot(forest, aes(hazard_ratio_per_sd, model)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#718096") +
  geom_errorbarh(
    aes(xmin = ci95_low, xmax = ci95_high), height = 0.18, linewidth = 0.8
  ) +
  geom_point(aes(color = role == "primary"), size = 2.7) +
  geom_text(aes(x = ci95_high * 1.04, label = label), hjust = 0, size = 2.9) +
  scale_color_manual(
    values = c("TRUE" = "#C43C39", "FALSE" = "#4A5568"),
    guide = "none"
  ) +
  scale_x_log10() +
  coord_cartesian(clip = "off") +
  labs(title = "Continuous OS models", x = "Hazard ratio per SD", y = NULL) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.margin = margin(5.5, 125, 5.5, 5.5)
  )

joint_figure <- (((p_km / p_risk) + plot_layout(heights = c(4.2, 1))) |
  p_forest) + plot_layout(widths = c(1.05, 1.35))
ggsave(
  file.path(figdir, "tigit_cd39_complete74_top40_survival.png"),
  joint_figure, width = 14.2, height = 6.6, dpi = 300, bg = "white"
)
ggsave(
  file.path(figdir, "tigit_cd39_complete74_top40_survival.pdf"),
  joint_figure, width = 14.2, height = 6.6, device = grDevices::pdf,
  useDingbats = FALSE, bg = "white"
)

cat(paste(summary_lines, collapse = "\n"), "\n")
