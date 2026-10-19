# One regression with R's fixest, for benchmarks/_harness.py: read the Parquet
# file into memory (counted, as pandas is for pyfixest), fit with standard
# errors clustered as asked, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_rss_mb": ..., "coefs": {...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations.
#
#   Rscript fixest.R '<spec as JSON>'      (see external_spec in _harness.py)

spec <- jsonlite::fromJSON(commandArgs(TRUE)[1])

hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]

suppressPackageStartupMessages({
  library(fixest)
  library(arrow)
})
setFixest_nthreads(spec$threads)
baseline <- hwm_mb()

terms <- spec$covariates
if (!is.null(spec$categorical)) {
  terms <- c(terms, sprintf("i(%s, ref = %s)", spec$categorical$column,
                            spec$categorical$ref))
}
fml <- as.formula(sprintf("%s ~ %s | %s", spec$outcome, paste(terms, collapse = " + "),
                          paste(spec$fixed_effects, collapse = " + ")))

t0 <- now()
if (is.null(spec$columns)) {
  df <- as.data.frame(read_parquet(spec$path))
} else {
  df <- as.data.frame(read_parquet(spec$path, col_select = spec$columns))
}
t_read <- now() - t0

t0 <- now()
fit <- feols(fml, data = df, cluster = as.formula(paste0("~", spec$cluster)),
             fixef.rm = if (isTRUE(spec$drop_singletons)) "singleton" else "none")
invisible(se(fit))
t_fit <- now() - t0

cat("RESULT ", jsonlite::toJSON(list(
  t_read = t_read, t_fit = t_fit, n_obs = nobs(fit),
  peak_rss_mb = hwm_mb(), baseline_rss_mb = baseline, peak_disk_mb = 0,
  coefs = as.list(coef(fit)),
  version = as.character(packageVersion("fixest")), threads = getFixest_nthreads()
), auto_unbox = TRUE, digits = NA), "\n", sep = "")
