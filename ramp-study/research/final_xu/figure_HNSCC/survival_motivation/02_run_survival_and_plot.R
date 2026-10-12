#!/usr/bin/env Rscript

"%||%" <- function(x, y) if (is.null(x)) y else x

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve this script's path")
script_path <- gsub("~\\+~", " ", sub("^--file=", "", script_arg))
here <- dirname(normalizePath(script_path))

score_path <- normalizePath(file.path(here, "output", "patient_spatial_niche_scores.csv"))
clinical_path <- normalizePath(file.path(
  here, "..", "sampled_data", "haist_codex", "complete",
  "HNSCCdiscovery_clinmetadata.csv"
))
plan_path <- normalizePath(file.path(here, "analysis_plan.json"))
out <- file.path(here, "output")
figdir <- file.path(here, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

plan <- fromJSON(plan_path, simplifyVector = FALSE)
if (!is.null(plan$survival$minimum_pt_cd8_cell_threshold)) {
  stop("This analysis must not impose a minimum PT-CD8 cell threshold")
}

zscore <- function(x) {
  if (any(!is.finite(x))) stop("zscore received non-finite values")
  sigma <- sd(x)
  if (!is.finite(sigma) || sigma <= 0) stop("Cannot standardize a constant score")
  (x - mean(x)) / sigma
}

rank_normal <- function(x) {
  zscore(qnorm((rank(x, ties.method = "average") - 0.5) / length(x)))
}

format_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return(format(p, scientific = TRUE, digits = 2))
  sprintf("%.3f", p)
}

fit_cox <- function(frame, formula, score_term, event_term, model, scale) {
  variables <- all.vars(formula)
  d <- frame[complete.cases(frame[, variables, drop = FALSE]), , drop = FALSE]
  fit <- coxph(formula, data = d, ties = "efron", x = TRUE)
  s <- summary(fit)
  if (!(score_term %in% rownames(s$coefficients))) {
    stop(sprintf("Score term %s is absent from model %s", score_term, model))
  }
  zph <- try(cox.zph(fit), silent = TRUE)
  feature_ph_p <- global_ph_p <- NA_real_
  if (!inherits(zph, "try-error")) {
    if (score_term %in% rownames(zph$table)) feature_ph_p <- zph$table[score_term, "p"]
    if ("GLOBAL" %in% rownames(zph$table)) global_ph_p <- zph$table["GLOBAL", "p"]
  }
  data.frame(
    model = model,
    score_scale = scale,
    patients = nrow(d),
    deaths = sum(d[[event_term]]),
    beta = unname(s$coefficients[score_term, "coef"]),
    se = unname(s$coefficients[score_term, "se(coef)"]),
    hazard_ratio = unname(s$conf.int[score_term, "exp(coef)"]),
    ci95_low = unname(s$conf.int[score_term, "lower .95"]),
    ci95_high = unname(s$conf.int[score_term, "upper .95"]),
    wald_p = unname(s$coefficients[score_term, "Pr(>|z|)"]),
    feature_ph_p = feature_ph_p,
    global_ph_p = global_ph_p,
    concordance = unname(s$concordance[1]),
    stringsAsFactors = FALSE
  )
}

rmst_to_tau <- function(time, event, tau) {
  fit <- survfit(Surv(time, event) ~ 1)
  before <- fit$time < tau
  ends <- c(fit$time[before], tau)
  starts <- c(0, fit$time[before])
  survival_left <- c(1, fit$surv[before])
  sum((ends - starts) * survival_left)
}

scores <- read.csv(score_path, stringsAsFactors = FALSE, check.names = FALSE)
clinical_roi <- read.csv(clinical_path, stringsAsFactors = FALSE, check.names = FALSE)

if (anyDuplicated(scores$CaseID)) stop("Spatial patient scores contain duplicate CaseID")
if (anyDuplicated(clinical_roi$unique_region)) stop("Clinical table contains duplicate ROI rows")

# The provider clinical table is ROI-level. Check that patient-level values are
# invariant before reducing it to one row per patient, preventing ROI-level
# pseudoreplication in survival analysis.
clinical_columns <- c("OS", "OS_status", "Age", "HPV_status", "AJCC_stage")
for (column in clinical_columns) {
  inconsistent <- tapply(
    clinical_roi[[column]], clinical_roi$CaseID,
    function(x) length(unique(x[!is.na(x)])) > 1L
  )
  if (any(inconsistent)) stop(sprintf("Clinical column %s varies within patient", column))
}
clinical <- clinical_roi[!duplicated(clinical_roi$CaseID), , drop = FALSE]
clinical$os_event <- unname(c("OS+" = 1L, "OS-" = 0L)[clinical$OS_status])

analysis <- merge(scores, clinical, by = "CaseID", all = FALSE, sort = TRUE)
analysis <- analysis[
  is.finite(analysis$OS) & !is.na(analysis$os_event) &
    is.finite(analysis$patient_spatial_niche_score),
  , drop = FALSE
]
if (nrow(analysis) != 74L || sum(analysis$os_event) != 24L) {
  stop(sprintf(
    "Expected 74 patients and 24 deaths; found %d patients and %d deaths",
    nrow(analysis), sum(analysis$os_event)
  ))
}

analysis$spatial_score_raw <- analysis$patient_spatial_niche_score
analysis$spatial_score_z <- zscore(analysis$spatial_score_raw)
analysis$spatial_score_rank_z <- rank_normal(analysis$spatial_score_raw)
analysis$os_time_60 <- pmin(analysis$OS, 60)
analysis$os_event_60 <- as.integer(analysis$os_event == 1 & analysis$OS <= 60)
analysis$Age_z <- rep(NA_real_, nrow(analysis))
analysis$HPV_status_z <- rep(NA_real_, nrow(analysis))
analysis$AJCC_stage_z <- rep(NA_real_, nrow(analysis))
for (column in c("Age", "HPV_status", "AJCC_stage")) {
  ok <- is.finite(analysis[[column]])
  analysis[[paste0(column, "_z")]][ok] <- zscore(analysis[[column]][ok])
}

median_cutpoint <- median(analysis$spatial_score_raw)
analysis$score_group <- factor(
  ifelse(analysis$spatial_score_raw >= median_cutpoint, "Higher", "Lower"),
  levels = c("Lower", "Higher")
)
write.csv(analysis, file.path(out, "survival_analysis_input.csv"), row.names = FALSE)

# The continuous raw score is the only primary test. The rank-normalized and
# fixed clinical-adjustment models are scale/confounding checks, not alternate
# cohort definitions.
model_rows <- list(
  fit_cox(
    analysis,
    Surv(OS, os_event) ~ spatial_score_z,
    "spatial_score_z",
    "os_event",
    "Primary: continuous spatial score",
    "Raw score per analysis-cohort SD"
  ),
  fit_cox(
    analysis,
    Surv(OS, os_event) ~ spatial_score_rank_z,
    "spatial_score_rank_z",
    "os_event",
    "Scale check: inverse-normal rank",
    "Inverse-normal rank per SD"
  ),
  fit_cox(
    analysis,
    Surv(os_time_60, os_event_60) ~ spatial_score_z,
    "spatial_score_z",
    "os_event_60",
    "Timing description: 0-60 months",
    "Raw score per analysis-cohort SD"
  ),
  fit_cox(
    analysis,
    Surv(OS, os_event) ~ spatial_score_z + Age_z + HPV_status_z + AJCC_stage_z,
    "spatial_score_z",
    "os_event",
    "Fixed clinical adjustment",
    "Raw score per analysis-cohort SD"
  )
)
models <- do.call(rbind, model_rows)
write.csv(models, file.path(out, "survival_models.csv"), row.names = FALSE)

# Median split is created after fixing the continuous model and is used only to
# provide an interpretable Kaplan-Meier visualization.
risk_times <- c(0, 24, 48, 72, 96, 120)
km_rows <- list()
risk_rows <- list()
absolute_rows <- list()
for (group_name in levels(analysis$score_group)) {
  d <- analysis[analysis$score_group == group_name, , drop = FALSE]
  fit <- survfit(Surv(OS, os_event) ~ 1, data = d, conf.type = "log")
  km_rows[[length(km_rows) + 1L]] <- data.frame(
    group = group_name,
    time = c(0, fit$time),
    survival = c(1, fit$surv),
    stringsAsFactors = FALSE
  )
  risk_summary <- summary(fit, times = risk_times, extend = TRUE)
  risk_rows[[length(risk_rows) + 1L]] <- data.frame(
    group = group_name,
    time = risk_times,
    n_risk = unname(risk_summary$n.risk),
    stringsAsFactors = FALSE
  )
  at_60 <- summary(fit, times = 60, extend = TRUE)
  absolute_rows[[length(absolute_rows) + 1L]] <- data.frame(
    group = group_name,
    patients = nrow(d),
    deaths = sum(d$os_event),
    survival_60_months = unname(at_60$surv[1]),
    survival_60_months_ci_low = unname(at_60$lower[1]),
    survival_60_months_ci_high = unname(at_60$upper[1]),
    rmst_60_months = rmst_to_tau(d$OS, d$os_event, 60),
    stringsAsFactors = FALSE
  )
}
km <- do.call(rbind, km_rows)
risk <- do.call(rbind, risk_rows)
absolute <- do.call(rbind, absolute_rows)
write.csv(km, file.path(out, "median_split_km_curve.csv"), row.names = FALSE)
write.csv(risk, file.path(out, "median_split_number_at_risk.csv"), row.names = FALSE)
write.csv(absolute, file.path(out, "median_split_60month_summary.csv"), row.names = FALSE)

primary <- models[models$model == "Primary: continuous spatial score", , drop = FALSE]
group_counts <- table(analysis$score_group)
colors <- c("Lower" = "#2B6CB0", "Higher" = "#C43C39")

p_km <- ggplot(km[km$time <= 120, ], aes(x = time, y = survival, color = group)) +
  geom_step(linewidth = 1.15, direction = "hv") +
  geom_vline(xintercept = 60, linetype = "dotted", color = "#718096") +
  annotate(
    "label", x = 116, y = 0.98, hjust = 1, vjust = 1,
    label = sprintf(
      "Continuous Cox: HR %.2f per SD\n95%% CI %.2f-%.2f; P=%s",
      primary$hazard_ratio, primary$ci95_low, primary$ci95_high,
      format_p(primary$wald_p)
    ),
    size = 3.7, label.size = 0.2, fill = "white"
  ) +
  scale_color_manual(
    values = colors,
    breaks = c("Lower", "Higher"),
    labels = c(
      "Lower" = sprintf("Lower score (n=%d)", group_counts[["Lower"]]),
      "Higher" = sprintf("Higher score (n=%d)", group_counts[["Higher"]])
    )
  ) +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "CAF/macrophage enrichment around primary-tumor CD8 T cells",
    subtitle = "Median split is shown for visualization; the continuous Cox model is primary",
    x = NULL,
    y = "Overall survival probability",
    color = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "inside",
    legend.position.inside = c(0.25, 0.18),
    legend.background = element_rect(fill = "white", color = NA)
  )

p_risk <- ggplot(risk, aes(x = time, y = group, label = n_risk, color = group)) +
  geom_text(size = 3.6, fontface = "bold") +
  scale_color_manual(values = colors, guide = "none") +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  labs(x = "Overall survival (months)", y = "Number at risk") +
  theme_classic(base_size = 10) +
  theme(
    axis.line.y = element_blank(), axis.ticks.y = element_blank(),
    axis.text.y = element_text(face = "bold")
  )

forest_order <- c(
  "Primary: continuous spatial score",
  "Scale check: inverse-normal rank",
  "Timing description: 0-60 months",
  "Fixed clinical adjustment"
)
forest <- models[match(forest_order, models$model), , drop = FALSE]
forest$model <- factor(forest$model, levels = rev(forest_order))
forest$label <- sprintf(
  "HR %.2f (%.2f-%.2f), P=%s",
  forest$hazard_ratio, forest$ci95_low, forest$ci95_high,
  vapply(forest$wald_p, format_p, character(1))
)
p_forest <- ggplot(forest, aes(x = hazard_ratio, y = model)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#718096") +
  geom_errorbarh(aes(xmin = ci95_low, xmax = ci95_high), height = 0.18, linewidth = 0.8) +
  geom_point(size = 2.7, color = "#C43C39") +
  geom_text(aes(x = ci95_high * 1.06, label = label), hjust = 0, size = 3.15) +
  scale_x_log10(limits = c(0.65, 11), breaks = c(0.75, 1, 1.5, 2, 3, 4, 6, 10)) +
  labs(
    title = "Continuous-score survival models",
    subtitle = "All 74 patients with complete overall-survival data",
    x = "Overall-survival hazard ratio",
    y = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.margin = margin(5.5, 85, 5.5, 5.5)
  )

combined <- ((p_km / p_risk) + plot_layout(heights = c(4.2, 1))) | p_forest
combined <- combined + plot_layout(widths = c(1.05, 1.15))
ggsave(
  file.path(figdir, "hnscc_pt_cd8_niche_survival.png"), combined,
  width = 14, height = 6.7, dpi = 300, bg = "white"
)
ggsave(
  file.path(figdir, "hnscc_pt_cd8_niche_survival.pdf"), combined,
  width = 14, height = 6.7, device = grDevices::pdf, bg = "white",
  useDingbats = FALSE
)

manifest <- list(
  stage = "patient-level overall-survival analysis",
  input_spatial_score = score_path,
  input_clinical_table = clinical_path,
  previous_result_files_read = list(),
  patients_with_spatial_score = nrow(scores),
  patients_with_complete_os_and_spatial_score = nrow(analysis),
  deaths = sum(analysis$os_event),
  minimum_pt_cd8_cell_threshold = NULL,
  primary_test = "continuous raw patient spatial niche score, standardized once across all 74 patients",
  median_split_role = "descriptive visualization only",
  median_cutpoint = median_cutpoint,
  primary_result = list(
    hazard_ratio_per_sd = primary$hazard_ratio,
    ci95 = c(primary$ci95_low, primary$ci95_high),
    wald_p = primary$wald_p,
    feature_ph_p = primary$feature_ph_p
  ),
  figures = list(
    png = file.path(figdir, "hnscc_pt_cd8_niche_survival.png"),
    pdf = file.path(figdir, "hnscc_pt_cd8_niche_survival.pdf")
  )
)
write_json(manifest, file.path(out, "survival_manifest.json"), pretty = TRUE, auto_unbox = TRUE)

summary_lines <- c(
  sprintf("Cohort: %d patients / %d deaths", nrow(analysis), sum(analysis$os_event)),
  "No minimum PT-CD8 cell-count threshold was applied.",
  sprintf(
    "Primary continuous Cox: HR %.3f per SD (95%% CI %.3f-%.3f), P=%.5g",
    primary$hazard_ratio, primary$ci95_low, primary$ci95_high, primary$wald_p
  ),
  sprintf("Median split: Lower n=%d; Higher n=%d", group_counts[["Lower"]], group_counts[["Higher"]]),
  "The median split is descriptive; inference is based on the continuous Cox model."
)
writeLines(summary_lines, file.path(out, "SURVIVAL_SUMMARY.txt"))
cat(paste(summary_lines, collapse = "\n"), "\n")
