# One sensitivity analysis with R's HonestDiD, for benchmarks/sensitivity_benchmark.py:
# read the event study, run it once, and print one line,
#
#   RESULT {"seconds": ..., "peak_rss_mb": ..., "baseline_rss_mb": ..., "lb": ..., "ub": ...}
#
# Peak memory is the process's VmHWM; the baseline is VmHWM after loading the package.
#
#   Rscript honestdid_benchmark.R <event study JSON> <job> <M>
#
# job: "relative_magnitudes" (createSensitivityResults_relativeMagnitudes, C-LF),
# "smoothness_relative_magnitudes" (the same with bound = "deviation from linear
# trend"), "smoothness" (createSensitivityResults, FLCI) or "smoothness_hybrid"
# (createSensitivityResults, C-F).
args <- commandArgs(TRUE)
hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
suppressPackageStartupMessages(library(HonestDiD))
es <- jsonlite::fromJSON(args[1])
job <- args[2]
M <- as.numeric(args[3])
b <- as.numeric(es$beta)
V <- as.matrix(es$sigma)
np <- es$n_pre
nq <- length(b) - np
baseline <- hwm_mb()
t0 <- proc.time()[["elapsed"]]
r <- suppressWarnings(switch(job,
  relative_magnitudes = createSensitivityResults_relativeMagnitudes(b, V, np, nq, Mbarvec = M),
  smoothness_relative_magnitudes = createSensitivityResults_relativeMagnitudes(
    b, V, np, nq, Mbarvec = M, bound = "deviation from linear trend"),
  smoothness = createSensitivityResults(b, V, np, nq, Mvec = M),
  smoothness_hybrid = createSensitivityResults(b, V, np, nq, Mvec = M, method = "C-F")))
seconds <- proc.time()[["elapsed"]] - t0
cat("RESULT", jsonlite::toJSON(list(seconds = seconds, peak_rss_mb = hwm_mb(),
                                    baseline_rss_mb = baseline, lb = r$lb[1], ub = r$ub[1]),
                               auto_unbox = TRUE, digits = NA), "\n")
