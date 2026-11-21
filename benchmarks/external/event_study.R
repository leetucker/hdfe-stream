# An event study and its sensitivity analysis in R, for
# benchmarks/event_study_benchmark.py: read the panel into memory with arrow, fit
# y ~ i(rel_time, ref = -1) | unit + year with fixest, clustered by unit, then run
# HonestDiD on the event-time coefficients and their covariance, for relative
# magnitudes (C-LF) and smoothness (FLCI) at the values of M given. Prints one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "t_relative_magnitudes": ..., "t_smoothness": ..., ...}
#
#   Rscript event_study.R '<spec as JSON>'
#
# spec: path, n_pre, n_post, mbar, m (lists), threads (fixest's), parallel (HonestDiD's
# parallel = TRUE, on doParallel's forked workers, one per logical CPU).
spec <- jsonlite::fromJSON(commandArgs(TRUE)[1])
hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]
suppressPackageStartupMessages({
  library(fixest)
  library(arrow)
  library(HonestDiD)
})
setFixest_nthreads(spec$threads)
parallel <- isTRUE(spec$parallel)
if (parallel) {
  suppressPackageStartupMessages(library(doParallel))
  registerDoParallel(cores = parallel::detectCores())
}
baseline <- hwm_mb()

t0 <- now()
df <- as.data.frame(read_parquet(spec$path))
t_read <- now() - t0

t0 <- now()
fit <- feols(y ~ i(rel_time, ref = -1) | unit + year, data = df, cluster = ~unit)
V <- vcov(fit)
t_fit <- now() - t0

# the event-time coefficients, earliest first: rel_time::-9 ... rel_time::3
times <- as.integer(sub("^rel_time::", "", names(coef(fit))))
o <- order(times)
b <- unname(coef(fit)[o])
V <- unname(V[o, o])
np <- spec$n_pre
nq <- spec$n_post

t0 <- now()
rm <- suppressWarnings(createSensitivityResults_relativeMagnitudes(
  b, V, np, nq, Mbarvec = as.numeric(spec$mbar), parallel = parallel))
t_rm <- now() - t0
t0 <- now()
sd <- suppressWarnings(createSensitivityResults(
  b, V, np, nq, Mvec = as.numeric(spec$m), parallel = parallel))
t_sd <- now() - t0
rm <- rm[order(rm$Mbar), ]
sd <- sd[order(sd$M), ]

cat("RESULT ", jsonlite::toJSON(list(
  t_read = t_read, t_fit = t_fit, t_relative_magnitudes = t_rm, t_smoothness = t_sd,
  n_obs = nobs(fit), peak_rss_mb = hwm_mb(), baseline_rss_mb = baseline, peak_disk_mb = 0,
  beta = b, relative_magnitudes = list(lower = rm$lb, upper = rm$ub),
  smoothness = list(lower = sd$lb, upper = sd$ub),
  threads = getFixest_nthreads(),
  workers = if (parallel) foreach::getDoParWorkers() else 1L
), auto_unbox = TRUE, digits = NA), "\n", sep = "")
