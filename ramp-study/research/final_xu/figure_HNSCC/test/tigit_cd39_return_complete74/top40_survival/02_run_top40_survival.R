#!/usr/bin/env Rscript

# Refit the complete-74 HNSCC survival models after changing only the
# within-patient returned-state threshold from top 25% to top 40%.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve script path")
script_path <- gsub("~\\+~", " ", sub("^--file=", "", script_arg))
here <- dirname(normalizePath(script_path))

feature_path <- file.path(here, "output", "haist74_tigit_cd39_top40_patient_features.csv")
base_path <- file.path(here, "..", "output", "survival_analysis_input.csv")
top25_models_path <- file.path(
  here, "..", "output", "joint_spatial_molecular_survival_models.csv"
)
out <- file.path(here, "output")
figdir <- file.path(here, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

zscore <- function(x) as.numeric((x - mean(x)) / sd(x))
format_p <- function(p) {
  if (p < 0.001) return(format(p, scientific = TRUE, digits = 2))
  sprintf("%.3f", p)
}

fit_model <- function(frame, formula, term, event_term, label, role) {
  variables <- all.vars(formula)
  d <- frame[complete.cases(frame[, variables, drop = FALSE]), , drop = FALSE]
  fit <- coxph(formula, data = d, ties = "efron", x = TRUE)
  sm <- summary(fit)
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

features <- read.csv(feature_path, stringsAsFactors = FALSE)
base <- read.csv(base_path, stringsAsFactors = FALSE)
features$patient <- as.character(features$patient)
base$patient <- as.character(base$patient)

clinical_columns <- c(
  "patient", "OS", "os_event", "Age_z", "HPV_status_z", "AJCC_stage_z",
  "direct_spatial_niche_z"
)
if (!all(clinical_columns %in% colnames(base))) {
  stop("The frozen top-25 survival input lacks required clinical columns")
}
dat <- merge(features, base[, clinical_columns], by = "patient", all.x = TRUE, sort = TRUE)
if (nrow(dat) != 74L || sum(dat$os_event) != 24L || anyDuplicated(dat$patient)) {
  stop("Expected the same 74 patients and 24 deaths")
}

dat$high_pair_spatial_z <- zscore(dat$high_pair_mean_spatial_y)
dat$os_time_60 <- pmin(dat$OS, 60)
dat$os_event_60 <- as.integer(dat$os_event == 1L & dat$OS <= 60)

without_41403 <- dat[dat$patient != "41403", , drop = FALSE]
without_41403$high_pair_spatial_z <- zscore(without_41403$high_pair_mean_spatial_y)

primary <- fit_model(
  dat, Surv(OS, os_event) ~ high_pair_spatial_z,
  "high_pair_spatial_z", "os_event",
  "Primary: top-40% CD8 spatial exposure", "primary"
)
timing <- fit_model(
  dat, Surv(os_time_60, os_event_60) ~ high_pair_spatial_z,
  "high_pair_spatial_z", "os_event_60",
  "Timing sensitivity: 0-60 months", "timing sensitivity"
)
clinical_adjusted <- fit_model(
  dat,
  Surv(OS, os_event) ~ high_pair_spatial_z + Age_z + HPV_status_z + AJCC_stage_z,
  "high_pair_spatial_z", "os_event",
  "Fixed clinical adjustment", "clinical sensitivity"
)
spatial_adjusted <- fit_model(
  dat, Surv(OS, os_event) ~ high_pair_spatial_z + direct_spatial_niche_z,
  "high_pair_spatial_z", "os_event",
  "Joint score + direct spatial score", "spatial-burden sensitivity"
)
direct_reference <- fit_model(
  dat, Surv(OS, os_event) ~ direct_spatial_niche_z,
  "direct_spatial_niche_z", "os_event",
  "Direct spatial niche score", "direct spatial reference"
)
delete_41403 <- fit_model(
  without_41403, Surv(OS, os_event) ~ high_pair_spatial_z,
  "high_pair_spatial_z", "os_event",
  "Sensitivity: exclude Case 41403", "low-cell-count sensitivity"
)

models <- rbind(
  primary, timing, clinical_adjusted, spatial_adjusted,
  direct_reference, delete_41403
)
write.csv(models, file.path(out, "top40_survival_models.csv"), row.names = FALSE)
write.csv(dat, file.path(out, "top40_survival_analysis_input.csv"), row.names = FALSE)

top25_models <- read.csv(top25_models_path, stringsAsFactors = FALSE)
top25_models$threshold <- "Top 25%"
models$threshold <- "Top 40%"
shared <- intersect(colnames(top25_models), colnames(models))
comparison <- rbind(top25_models[, shared], models[, shared])
write.csv(comparison, file.path(out, "top25_vs_top40_models.csv"), row.names = FALSE)

# Median split is for visualization only, exactly as in the top-25 analysis.
cutpoint <- median(dat$high_pair_mean_spatial_y)
dat$joint_group <- factor(
  ifelse(dat$high_pair_mean_spatial_y >= cutpoint, "Higher", "Lower"),
  levels = c("Lower", "Higher")
)
risk_times <- c(0, 24, 48, 72, 96, 120)
km_rows <- list()
risk_rows <- list()
for (group_name in levels(dat$joint_group)) {
  local <- dat[dat$joint_group == group_name, , drop = FALSE]
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
write.csv(km, file.path(out, "top40_median_split_km_curve.csv"), row.names = FALSE)
write.csv(risk, file.path(out, "top40_median_split_number_at_risk.csv"), row.names = FALSE)

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
    ), size = 3.5, label.size = 0.2
  ) +
  scale_color_manual(
    values = colors,
    labels = c(
      "Lower" = sprintf("Lower top-40%% joint score (n=%d)", counts[["Lower"]]),
      "Higher" = sprintf("Higher top-40%% joint score (n=%d)", counts[["Higher"]])
    )
  ) +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "CAF/macrophage exposure of returned TIGIT/CD39 top-40% CD8",
    subtitle = "Complete OS cohort; median split shown only for visualization",
    x = NULL, y = "Overall survival probability", color = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "inside", legend.position.inside = c(0.31, 0.18),
    legend.background = element_rect(fill = "white", color = NA)
  )

p_risk <- ggplot(risk, aes(time, group, label = n_risk, color = group)) +
  geom_text(size = 3.4, fontface = "bold") +
  scale_color_manual(values = colors, guide = "none") +
  scale_x_continuous(breaks = risk_times, limits = c(0, 120), expand = c(0.01, 0.01)) +
  labs(x = "Overall survival (months)", y = "At risk") +
  theme_classic(base_size = 9) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

forest <- models
forest$model <- factor(forest$model, levels = rev(models$model))
forest$label <- sprintf(
  "HR %.2f (%.2f-%.2f), P=%s",
  forest$hazard_ratio_per_sd, forest$ci95_low, forest$ci95_high,
  vapply(forest$wald_p, format_p, character(1))
)
p_forest <- ggplot(forest, aes(hazard_ratio_per_sd, model)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#718096") +
  geom_errorbarh(aes(xmin = ci95_low, xmax = ci95_high), height = 0.18) +
  geom_point(aes(color = role == "primary"), size = 2.7) +
  geom_text(aes(x = ci95_high * 1.04, label = label), hjust = 0, size = 2.9) +
  scale_color_manual(values = c("TRUE" = "#C43C39", "FALSE" = "#4A5568"), guide = "none") +
  scale_x_log10() + coord_cartesian(clip = "off") +
  labs(title = "Continuous OS models", x = "Hazard ratio per SD", y = NULL) +
  theme_classic(base_size = 10) +
  theme(plot.title = element_text(face = "bold"), plot.margin = margin(5.5, 125, 5.5, 5.5))

joint <- (((p_km / p_risk) + plot_layout(heights = c(4.2, 1))) | p_forest) +
  plot_layout(widths = c(1.05, 1.35))
ggsave(
  file.path(figdir, "tigit_cd39_complete74_top40_survival.png"), joint,
  width = 14.2, height = 6.6, dpi = 300, bg = "white"
)
ggsave(
  file.path(figdir, "tigit_cd39_complete74_top40_survival.pdf"), joint,
  width = 14.2, height = 6.6, device = grDevices::pdf,
  useDingbats = FALSE, bg = "white"
)

top25_primary <- top25_models[top25_models$role == "primary continuous joint score", ]
summary_lines <- c(
  "TOP-40% TIGIT/CD39 HIGH-STATE SPATIAL-EXPOSURE SURVIVAL TEST",
  "Only the within-patient high-state threshold changed: top 25% -> top 40%.",
  sprintf(
    "Top 40%% primary: HR=%.6f per SD (95%% CI %.6f-%.6f), Wald P=%.8g; n=%d, deaths=%d",
    primary$hazard_ratio_per_sd, primary$ci95_low, primary$ci95_high,
    primary$wald_p, primary$patients, primary$deaths
  ),
  sprintf(
    "Top 25%% reference: HR=%.6f per SD (95%% CI %.6f-%.6f), Wald P=%.8g",
    top25_primary$hazard_ratio_per_sd, top25_primary$ci95_low,
    top25_primary$ci95_high, top25_primary$wald_p
  ),
  sprintf(
    "Top25-vs-top40 patient-score Spearman rho=%.6f",
    cor(
      base$high_pair_mean_spatial_y,
      dat$high_pair_mean_spatial_y[match(base$patient, dat$patient)],
      method = "spearman"
    )
  )
)
writeLines(summary_lines, file.path(out, "TOP40_SURVIVAL_SUMMARY.txt"))
cat(paste(summary_lines, collapse = "\n"), "\n")
