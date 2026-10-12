#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve script path")
script_path <- gsub("~\\+~", " ", sub("^--file=", "", script_arg))
here <- dirname(normalizePath(script_path))
out <- file.path(here, "output")
figdir <- file.path(here, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

feature_path <- file.path(out, "haist70_tigit_cd39_patient_features.csv")
lock_path <- file.path(out, "PAIR_LOCK_AND_RETURN_MANIFEST.json")
clinical_path <- file.path(
  here, "..", "..", "sampled_data", "haist_codex", "complete",
  "HNSCCdiscovery_clinmetadata.csv"
)
selected_path <- file.path(
  here, "..", "pt70_full_90488_umap", "input", "selected_70_patients.csv"
)

lock <- fromJSON(lock_path, simplifyVector = FALSE)
if (!identical(lock$survival_outcome_read, FALSE)) {
  stop("The upstream pair lock must be outcome blind")
}
if (!identical(unlist(lock$program$markers_raw), c("TIGIT--VSTM3", "CD39"))) {
  stop("The locked program is not exactly TIGIT/CD39")
}

zscore <- function(x) {
  sigma <- sd(x)
  if (!is.finite(sigma) || sigma <= 0) stop("Cannot standardize a constant feature")
  (x - mean(x)) / sigma
}

format_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return(format(p, scientific = TRUE, digits = 2))
  sprintf("%.3f", p)
}

fit_model <- function(frame, formula, term, label, role) {
  variables <- all.vars(formula)
  d <- frame[complete.cases(frame[, variables, drop = FALSE]), , drop = FALSE]
  fit <- coxph(formula, data = d, ties = "efron", x = TRUE)
  sm <- summary(fit)
  if (!(term %in% rownames(sm$coefficients))) stop(paste("Missing term", term))
  zph <- try(cox.zph(fit), silent = TRUE)
  feature_ph_p <- global_ph_p <- NA_real_
  if (!inherits(zph, "try-error")) {
    if (term %in% rownames(zph$table)) feature_ph_p <- zph$table[term, "p"]
    if ("GLOBAL" %in% rownames(zph$table)) global_ph_p <- zph$table["GLOBAL", "p"]
  }
  data.frame(
    model = label,
    role = role,
    patients = nrow(d),
    deaths = sum(d$os_event),
    beta = unname(sm$coefficients[term, "coef"]),
    se = unname(sm$coefficients[term, "se(coef)"]),
    hazard_ratio = unname(sm$conf.int[term, "exp(coef)"]),
    ci95_low = unname(sm$conf.int[term, "lower .95"]),
    ci95_high = unname(sm$conf.int[term, "upper .95"]),
    wald_p = unname(sm$coefficients[term, "Pr(>|z|)"]),
    feature_ph_p = feature_ph_p,
    global_ph_p = global_ph_p,
    concordance = unname(sm$concordance[1]),
    stringsAsFactors = FALSE
  )
}

features <- read.csv(feature_path, stringsAsFactors = FALSE, check.names = FALSE)
selected <- read.csv(selected_path, stringsAsFactors = FALSE, check.names = FALSE)
clinical_roi <- read.csv(clinical_path, stringsAsFactors = FALSE, check.names = FALSE)
features$patient <- as.character(features$patient)
selected$CaseID <- as.character(selected$CaseID)
clinical_roi$CaseID <- as.character(clinical_roi$CaseID)

if (nrow(features) != 70L || anyDuplicated(features$patient)) {
  stop("Expected one outcome-blind feature row for each of 70 patients")
}
if (!setequal(features$patient, selected$CaseID)) {
  stop("Feature patients do not equal the locked PT70 selection")
}
for (column in c("OS", "OS_status", "Age", "HPV_status", "AJCC_stage")) {
  inconsistent <- tapply(
    clinical_roi[[column]], clinical_roi$CaseID,
    function(x) length(unique(x[!is.na(x)])) > 1L
  )
  if (any(inconsistent)) stop(paste("Clinical field varies within patient:", column))
}
clinical <- clinical_roi[!duplicated(clinical_roi$CaseID), , drop = FALSE]
clinical$os_event <- unname(c("OS+" = 1L, "OS-" = 0L)[clinical$OS_status])

dat <- merge(
  features,
  clinical[, c("CaseID", "OS", "OS_status", "os_event", "Age", "HPV_status", "AJCC_stage")],
  by.x = "patient", by.y = "CaseID", all.x = TRUE, sort = TRUE
)
if (nrow(dat) != 70L || sum(dat$os_event) != 23L || any(!is.finite(dat$OS))) {
  stop(sprintf("Expected 70 patients and 23 deaths; observed %d and %d", nrow(dat), sum(dat$os_event)))
}

dat$niche_enrichment_z <- zscore(dat$niche_minus_outside_mean_pair_rank)
dat$niche_raw_enrichment_z <- zscore(dat$niche_minus_outside_mean_pair_raw)
dat$pair_spatial_spearman_z <- zscore(dat$pair_spatial_spearman)
dat$overall_pair_z <- zscore(dat$overall_mean_predicted_pair)
dat$within_niche_pair_z <- zscore(dat$niche_mean_pair_rank)
dat$original_spatial_niche_z <- zscore(dat$patient_spatial_niche_score)
dat$high_pair_spatial_z <- zscore(dat$high_pair_mean_spatial_y)
dat$Age_z <- NA_real_
dat$HPV_status_z <- NA_real_
dat$AJCC_stage_z <- NA_real_
for (column in c("Age", "HPV_status", "AJCC_stage")) {
  ok <- is.finite(dat[[column]])
  dat[[paste0(column, "_z")]][ok] <- zscore(dat[[column]][ok])
}
write.csv(dat, file.path(out, "survival_analysis_input.csv"), row.names = FALSE)

models <- rbind(
  fit_model(
    dat, Surv(OS, os_event) ~ niche_raw_enrichment_z, "niche_raw_enrichment_z",
    "Primary: niche-minus-outside pair score", "niche-specific enrichment"
  ),
  fit_model(
    dat, Surv(OS, os_event) ~ niche_enrichment_z, "niche_enrichment_z",
    "Within-patient rank sensitivity", "niche-specific enrichment sensitivity"
  ),
  fit_model(
    dat, Surv(OS, os_event) ~ pair_spatial_spearman_z, "pair_spatial_spearman_z",
    "Within-patient pair/spatial coupling", "niche coupling"
  ),
  fit_model(
    dat, Surv(OS, os_event) ~ within_niche_pair_z, "within_niche_pair_z",
    "Pair state within CAF/macrophage niche", "within-niche intensity"
  ),
  fit_model(
    dat, Surv(OS, os_event) ~ overall_pair_z, "overall_pair_z",
    "Overall pair-state abundance", "nonspatial abundance control"
  ),
  fit_model(
    dat,
    Surv(OS, os_event) ~ niche_raw_enrichment_z + original_spatial_niche_z,
    "niche_raw_enrichment_z",
    "Niche enrichment + original spatial score", "spatial-score adjusted"
  ),
  fit_model(
    dat,
    Surv(OS, os_event) ~ niche_raw_enrichment_z + Age_z + HPV_status_z + AJCC_stage_z,
    "niche_raw_enrichment_z",
    "Fixed clinical adjustment", "clinically adjusted"
  )
)
models$fdr_bh_all_models <- p.adjust(models$wald_p, method = "BH")
write.csv(models, file.path(out, "survival_models.csv"), row.names = FALSE)

# A separate, biologically joint score that was already part of the historical
# return analysis: mean continuous spatial exposure among each patient's
# top-quartile predicted-state CD8 cells.  It is a joint phenotype and is not
# intended to establish independence from the direct spatial niche burden.
dat$os_time_60 <- pmin(dat$OS, 60)
dat$os_event_60 <- as.integer(dat$os_event == 1 & dat$OS <= 60)
joint_models <- rbind(
  fit_model(
    dat, Surv(OS, os_event) ~ high_pair_spatial_z, "high_pair_spatial_z",
    "High-state CD8 spatial exposure", "joint spatial-molecular score"
  ),
  fit_model(
    dat, Surv(os_time_60, os_event_60) ~ high_pair_spatial_z,
    "high_pair_spatial_z",
    "High-state exposure: 0-60 months", "timing sensitivity"
  ),
  fit_model(
    dat,
    Surv(OS, os_event) ~ high_pair_spatial_z + Age_z + HPV_status_z + AJCC_stage_z,
    "high_pair_spatial_z",
    "High-state exposure: clinical adjustment", "clinically adjusted"
  ),
  fit_model(
    dat,
    Surv(OS, os_event) ~ high_pair_spatial_z + original_spatial_niche_z,
    "high_pair_spatial_z",
    "High-state exposure + direct niche score", "spatial-burden adjusted"
  ),
  fit_model(
    dat, Surv(OS, os_event) ~ original_spatial_niche_z,
    "original_spatial_niche_z",
    "Direct spatial niche score", "direct spatial reference"
  )
)
write.csv(
  joint_models, file.path(out, "joint_spatial_molecular_survival_models.csv"),
  row.names = FALSE
)

# Diagnostic only: explain why the earlier 30-patient result differs from the
# full PT70 result.  This historical partition is not used for marker or model
# selection.
historical_30_path <- file.path(
  here, "..", "..", "..", "test0827", "hnscc", "pt_survival_first",
  "biomarker_combination_search", "output_haist40_selected",
  "haist30_cells_endpoint.csv.gz"
)
historical_30 <- read.csv(
  historical_30_path, stringsAsFactors = FALSE,
  colClasses = c(patient = "character")
)
heldout_patients <- unique(as.character(historical_30$patient))
dat$historical_partition <- ifelse(
  as.character(dat$patient) %in% heldout_patients,
  "historical_Haist30", "complementary_Haist40"
)
partition_rows <- list()
for (partition in c("historical_Haist30", "complementary_Haist40")) {
  local <- dat[dat$historical_partition == partition, , drop = FALSE]
  for (feature in c(
    "niche_minus_outside_mean_pair_raw",
    "niche_minus_outside_mean_pair_rank"
  )) {
    frame <- data.frame(
      OS = local$OS, os_event = local$os_event,
      feature_z = zscore(local[[feature]])
    )
    fit <- coxph(Surv(OS, os_event) ~ feature_z, data = frame, ties = "efron")
    sm <- summary(fit)
    partition_rows[[length(partition_rows) + 1L]] <- data.frame(
      partition = partition,
      feature = feature,
      patients = nrow(frame),
      deaths = sum(frame$os_event),
      hazard_ratio = unname(sm$conf.int[1, "exp(coef)"]),
      ci95_low = unname(sm$conf.int[1, "lower .95"]),
      ci95_high = unname(sm$conf.int[1, "upper .95"]),
      wald_p = unname(sm$coefficients[1, "Pr(>|z|)"]),
      stringsAsFactors = FALSE
    )
  }
}
partition_sensitivity <- do.call(rbind, partition_rows)
write.csv(
  partition_sensitivity,
  file.path(out, "historical_partition_sensitivity.csv"),
  row.names = FALSE
)

# Median split is visualization only; the continuous Cox model above is primary.
cutpoint <- median(dat$niche_minus_outside_mean_pair_rank)
dat$group <- factor(
  ifelse(dat$niche_minus_outside_mean_pair_rank >= cutpoint, "Higher", "Lower"),
  levels = c("Lower", "Higher")
)
km_rows <- list()
risk_rows <- list()
risk_times <- c(0, 24, 48, 72, 96, 120)
for (group_name in levels(dat$group)) {
  local <- dat[dat$group == group_name, , drop = FALSE]
  fit <- survfit(Surv(OS, os_event) ~ 1, data = local)
  km_rows[[length(km_rows) + 1L]] <- data.frame(
    group = group_name, time = c(0, fit$time), survival = c(1, fit$surv)
  )
  risk <- summary(fit, times = risk_times, extend = TRUE)
  risk_rows[[length(risk_rows) + 1L]] <- data.frame(
    group = group_name, time = risk_times, n_risk = unname(risk$n.risk)
  )
}
km <- do.call(rbind, km_rows)
risk <- do.call(rbind, risk_rows)
write.csv(km, file.path(out, "median_split_km_curve.csv"), row.names = FALSE)
write.csv(risk, file.path(out, "median_split_number_at_risk.csv"), row.names = FALSE)

enrichment <- fromJSON(file.path(out, "NICHE_ENRICHMENT_SUMMARY.json"))
primary <- models[models$model == "Primary: niche-minus-outside pair score", , drop = FALSE]
joint_primary <- joint_models[
  joint_models$model == "High-state CD8 spatial exposure", , drop = FALSE
]
summary_lines <- c(
  "LOCKED TIGIT/CD39 RETURN TO HAIST70",
  sprintf(
    "Niche enrichment: median score difference %.4f; %d/%d patients positive; Wilcoxon one-sided P=%s",
    enrichment$median_niche_minus_outside_raw, enrichment$positive_patients,
    enrichment$patients, format_p(enrichment$one_sided_wilcoxon_p)
  ),
  sprintf(
    "Primary continuous OS: HR %.3f per SD (95%% CI %.3f-%.3f), P=%s; n=%d, deaths=%d",
    primary$hazard_ratio, primary$ci95_low, primary$ci95_high,
    format_p(primary$wald_p), primary$patients, primary$deaths
  ),
  sprintf(
    "Joint high-state spatial exposure OS: HR %.3f per SD (95%% CI %.3f-%.3f), P=%s",
    joint_primary$hazard_ratio, joint_primary$ci95_low, joint_primary$ci95_high,
    format_p(joint_primary$wald_p)
  ),
  "The pair and patient features were locked before survival was opened.",
  "This is an internal closed-loop test because PT70 contributed to upstream state discovery."
)
writeLines(summary_lines, file.path(out, "SURVIVAL_SUMMARY.txt"))

colors <- c("Lower" = "#2B6CB0", "Higher" = "#C43C39")
group_counts <- table(dat$group)
p_enrichment <- ggplot(dat, aes(x = "", y = niche_minus_outside_mean_pair_raw)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#718096") +
  geom_violin(fill = "#E8EDF2", color = "#4A5568", width = 0.75) +
  geom_boxplot(width = 0.2, outlier.shape = NA, fill = "white") +
  geom_jitter(width = 0.11, height = 0, size = 1.6, alpha = 0.7, color = "#7B2CBF") +
  annotate(
    "label", x = 1, y = max(dat$niche_minus_outside_mean_pair_raw) * 0.98,
    label = sprintf(
      "Median = %.3f\n%d/%d positive\nWilcoxon P=%s",
      enrichment$median_niche_minus_outside_raw, enrichment$positive_patients,
      enrichment$patients, format_p(enrichment$one_sided_wilcoxon_p)
    ), vjust = 1, size = 3.5, label.size = 0.2
  ) +
  labs(
    title = "TIGIT/CD39 enrichment\nin CAF/macrophage-rich niches",
    subtitle = "Niche minus outside; n=70 patients",
    x = NULL, y = "Niche-minus-outside predicted state"
  ) +
  theme_classic(base_size = 11) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        plot.title = element_text(face = "bold"))

p_km <- ggplot(km[km$time <= 120, ], aes(time, survival, color = group)) +
  geom_step(linewidth = 1.1, direction = "hv") +
  annotate(
    "label", x = 116, y = 0.98, hjust = 1, vjust = 1,
    label = sprintf(
      "Continuous Cox\nHR %.2f per SD\n95%% CI %.2f-%.2f; P=%s",
      primary$hazard_ratio, primary$ci95_low, primary$ci95_high,
      format_p(primary$wald_p)
    ), size = 3.4, label.size = 0.2
  ) +
  scale_color_manual(
    values = colors,
    labels = c(
      "Lower" = sprintf("Lower enrichment (n=%d)", group_counts[["Lower"]]),
      "Higher" = sprintf("Higher enrichment (n=%d)", group_counts[["Higher"]])
    )
  ) +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "Niche-specific TIGIT/CD39 enrichment and survival",
    subtitle = "Median split shown only for visualization",
    x = NULL, y = "Overall survival probability", color = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(plot.title = element_text(face = "bold"), legend.position = "inside",
        legend.position.inside = c(0.31, 0.18),
        legend.background = element_rect(fill = "white", color = NA))

p_risk <- ggplot(risk, aes(time, group, label = n_risk, color = group)) +
  geom_text(size = 3.3, fontface = "bold") +
  scale_color_manual(values = colors, guide = "none") +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  labs(x = "Overall survival (months)", y = "At risk") +
  theme_classic(base_size = 9) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

forest <- models
forest$model <- factor(forest$model, levels = rev(models$model))
forest$label <- sprintf(
  "HR %.2f (%.2f-%.2f), P=%s",
  forest$hazard_ratio, forest$ci95_low, forest$ci95_high,
  vapply(forest$wald_p, format_p, character(1))
)
p_forest <- ggplot(forest, aes(hazard_ratio, model)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#718096") +
  geom_errorbarh(aes(xmin = ci95_low, xmax = ci95_high), height = 0.18) +
  geom_point(aes(color = role == "niche-specific enrichment"), size = 2.5) +
  geom_text(aes(x = ci95_high * 1.04, label = label), hjust = 0, size = 2.9) +
  scale_color_manual(values = c("TRUE" = "#C43C39", "FALSE" = "#4A5568"), guide = "none") +
  scale_x_log10() +
  coord_cartesian(clip = "off") +
  labs(title = "Continuous OS models", x = "Hazard ratio per SD", y = NULL) +
  theme_classic(base_size = 10) +
  theme(plot.title = element_text(face = "bold"), plot.margin = margin(5.5, 120, 5.5, 5.5))

combined <- p_enrichment | ((p_km / p_risk) + plot_layout(heights = c(4.2, 1))) | p_forest
combined <- combined + plot_layout(widths = c(0.72, 1.18, 1.35))
ggsave(
  file.path(figdir, "tigit_cd39_return_niche_survival.png"), combined,
  width = 17, height = 6.6, dpi = 300, bg = "white"
)
ggsave(
  file.path(figdir, "tigit_cd39_return_niche_survival.pdf"), combined,
  width = 17, height = 6.6, device = grDevices::pdf,
  useDingbats = FALSE, bg = "white"
)

# Dedicated joint-score survival figure.
joint_cutpoint <- median(dat$high_pair_mean_spatial_y)
dat$joint_group <- factor(
  ifelse(dat$high_pair_mean_spatial_y >= joint_cutpoint, "Higher", "Lower"),
  levels = c("Lower", "Higher")
)
joint_km_rows <- list()
joint_risk_rows <- list()
for (group_name in levels(dat$joint_group)) {
  local <- dat[dat$joint_group == group_name, , drop = FALSE]
  fit <- survfit(Surv(OS, os_event) ~ 1, data = local)
  joint_km_rows[[length(joint_km_rows) + 1L]] <- data.frame(
    group = group_name, time = c(0, fit$time), survival = c(1, fit$surv)
  )
  risk_summary <- summary(fit, times = risk_times, extend = TRUE)
  joint_risk_rows[[length(joint_risk_rows) + 1L]] <- data.frame(
    group = group_name, time = risk_times, n_risk = unname(risk_summary$n.risk)
  )
}
joint_km <- do.call(rbind, joint_km_rows)
joint_risk <- do.call(rbind, joint_risk_rows)
write.csv(joint_km, file.path(out, "joint_median_split_km_curve.csv"), row.names = FALSE)
write.csv(joint_risk, file.path(out, "joint_median_split_number_at_risk.csv"), row.names = FALSE)

joint_counts <- table(dat$joint_group)
p_joint_km <- ggplot(
  joint_km[joint_km$time <= 120, ], aes(time, survival, color = group)
) +
  geom_step(linewidth = 1.15, direction = "hv") +
  annotate(
    "label", x = 116, y = 0.98, hjust = 1, vjust = 1,
    label = sprintf(
      "Continuous Cox\nHR %.2f per SD\n95%% CI %.2f-%.2f; P=%s",
      joint_primary$hazard_ratio, joint_primary$ci95_low, joint_primary$ci95_high,
      format_p(joint_primary$wald_p)
    ), size = 3.5, label.size = 0.2
  ) +
  scale_color_manual(
    values = colors,
    labels = c(
      "Lower" = sprintf("Lower joint score (n=%d)", joint_counts[["Lower"]]),
      "Higher" = sprintf("Higher joint score (n=%d)", joint_counts[["Higher"]])
    )
  ) +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "CAF/macrophage exposure of predicted TIGIT/CD39-high CD8 cells",
    subtitle = "Top within-patient state quartile; median split shown for visualization",
    x = NULL, y = "Overall survival probability", color = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(plot.title = element_text(face = "bold"), legend.position = "inside",
        legend.position.inside = c(0.30, 0.18),
        legend.background = element_rect(fill = "white", color = NA))

p_joint_risk <- ggplot(joint_risk, aes(time, group, label = n_risk, color = group)) +
  geom_text(size = 3.4, fontface = "bold") +
  scale_color_manual(values = colors, guide = "none") +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  labs(x = "Overall survival (months)", y = "At risk") +
  theme_classic(base_size = 9) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

joint_forest <- joint_models
joint_forest$model <- factor(joint_forest$model, levels = rev(joint_models$model))
joint_forest$label <- sprintf(
  "HR %.2f (%.2f-%.2f), P=%s",
  joint_forest$hazard_ratio, joint_forest$ci95_low, joint_forest$ci95_high,
  vapply(joint_forest$wald_p, format_p, character(1))
)
p_joint_forest <- ggplot(joint_forest, aes(hazard_ratio, model)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#718096") +
  geom_errorbarh(aes(xmin = ci95_low, xmax = ci95_high), height = 0.18) +
  geom_point(aes(color = role == "joint spatial-molecular score"), size = 2.7) +
  geom_text(aes(x = ci95_high * 1.04, label = label), hjust = 0, size = 3.0) +
  scale_color_manual(values = c("TRUE" = "#C43C39", "FALSE" = "#4A5568"), guide = "none") +
  scale_x_log10() +
  coord_cartesian(clip = "off") +
  labs(title = "Joint-score survival models", x = "Hazard ratio per SD", y = NULL) +
  theme_classic(base_size = 10) +
  theme(plot.title = element_text(face = "bold"), plot.margin = margin(5.5, 120, 5.5, 5.5))

joint_figure <- ((p_joint_km / p_joint_risk) + plot_layout(heights = c(4.2, 1))) |
  p_joint_forest
joint_figure <- joint_figure + plot_layout(widths = c(1.05, 1.25))
ggsave(
  file.path(figdir, "tigit_cd39_high_state_spatial_survival.png"), joint_figure,
  width = 13.5, height = 6.4, dpi = 300, bg = "white"
)
ggsave(
  file.path(figdir, "tigit_cd39_high_state_spatial_survival.pdf"), joint_figure,
  width = 13.5, height = 6.4, device = grDevices::pdf,
  useDingbats = FALSE, bg = "white"
)

cat(paste(summary_lines, collapse = "\n"), "\n")
