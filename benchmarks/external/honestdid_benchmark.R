# One sensitivity analysis with R's HonestDiD, for benchmarks/sensitivity_benchmark.py:
# read the event study, run it once for the values of M given, and print one line,
#
#   RESULT {"seconds": ..., "peak_rss_mb": ..., "baseline_rss_mb": ..., "lower": [...], "upper": [...]}
#
# Peak memory is the process's VmHWM (with parallel workers, the caller also samples
# the whole process tree); the baseline is VmHWM after loading the packages.
#
#   Rscript honestdid_benchmark.R '<spec as JSON>'
#
# spec: path (the event study: beta, sigma, n_pre), job, M (a list), parallel. job is
# "relative_magnitudes" (createSensitivityResults_relativeMagnitudes, C-LF),
# "smoothness_relative_magnitudes" (the same with bound = "deviation from linear
# trend"), "smoothness" (createSensitivityResults, FLCI) or "smoothness_hybrid"
# (createSensitivityResults, C-F). With parallel, HonestDiD's parallel = TRUE runs the
# values of M on doParallel's forked workers, one per logical CPU.
spec <- jsonlite::fromJSON(commandArgs(TRUE)[1])
hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
suppressPackageStartupMessages(library(HonestDiD))
parallel <- isTRUE(spec$parallel)
if (parallel) {
  suppressPackageStartupMessages(library(doParallel))
  registerDoParallel(cores = parallel::detectCores())
}
es <- jsonlite::fromJSON(spec$path)
b <- as.numeric(es$beta)
V <- as.matrix(es$sigma)
np <- es$n_pre
nq <- length(b) - np
M <- as.numeric(spec$M)
baseline <- hwm_mb()
t0 <- proc.time()[["elapsed"]]
r <- suppressWarnings(switch(spec$job,
  relative_magnitudes = createSensitivityResults_relativeMagnitudes(
    b, V, np, nq, Mbarvec = M, parallel = parallel),
  smoothness_relative_magnitudes = createSensitivityResults_relativeMagnitudes(
    b, V, np, nq, Mbarvec = M, bound = "deviation from linear trend", parallel = parallel),
  smoothness = createSensitivityResults(b, V, np, nq, Mvec = M, parallel = parallel),
  smoothness_hybrid = createSensitivityResults(b, V, np, nq, Mvec = M, method = "C-F",
                                               parallel = parallel)))
seconds <- proc.time()[["elapsed"]] - t0
r <- r[order(if ("Mbar" %in% names(r)) r$Mbar else r$M), ]
cat("RESULT ", jsonlite::toJSON(list(
  seconds = seconds, peak_rss_mb = hwm_mb(), baseline_rss_mb = baseline,
  lower = r$lb, upper = r$ub, workers = if (parallel) foreach::getDoParWorkers() else 1L
), auto_unbox = TRUE, digits = NA), "\n", sep = "")
