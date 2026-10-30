# One OLS, IV or binary-outcome regression in R, for benchmarks/_regression.py:
# read the Parquet file into memory (counted, as pandas is for the Python
# libraries), fit with iid standard errors, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_rss_mb": ..., "coefs": {...}, "ses": {...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations.
#
#   Rscript regression.R '<spec as JSON>'
#
# The spec: path, task ("ols", "iv", "logit", "probit"), library ("fixest" for
# feols and feglm, "base" for lm and glm), fe (0 or 1; one fixed effect, the
# column g), outcome, covariates, threads.

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

fe <- isTRUE(spec$fe == 1)
iv <- spec$task == "iv"
columns <- c(spec$outcome, spec$covariates, if (fe) "g", if (iv) c("d", "z1", "z2"))

t0 <- now()
df <- as.data.frame(read_parquet(spec$path, col_select = columns))
t_read <- now() - t0

rhs <- paste(spec$covariates, collapse = " + ")
family <- if (spec$task %in% c("logit", "probit")) binomial(link = spec$task)

t0 <- now()
if (spec$library == "fixest") {
  # fixest's formula: outcome ~ covariates | fixed effects | endogenous ~ instruments
  parts <- c(rhs, if (fe) "g" else if (iv) "0", if (iv) "d ~ z1 + z2")
  fml <- as.formula(sprintf("%s ~ %s", spec$outcome, paste(parts, collapse = " | ")))
  fit <- if (is.null(family)) feols(fml, data = df, vcov = "iid")
         else feglm(fml, data = df, family = family, vcov = "iid")
  coefs <- coef(fit)
  ses <- se(fit)
  n_obs <- nobs(fit)
} else {
  fml <- as.formula(sprintf("%s ~ %s", spec$outcome, rhs))
  fit <- if (is.null(family)) lm(fml, data = df) else glm(fml, data = df, family = family)
  table <- summary(fit)$coefficients
  coefs <- table[, "Estimate"]
  ses <- table[, "Std. Error"]
  n_obs <- nrow(df)
}
t_fit <- now() - t0

# the names the Python libraries use
rename <- function(x) {
  names(x) <- sub("^fit_", "", sub("^\\(Intercept\\)$", "Intercept", names(x)))
  as.list(x)
}

cat("RESULT ", jsonlite::toJSON(list(
  t_read = t_read, t_fit = t_fit, n_obs = n_obs,
  peak_rss_mb = hwm_mb(), baseline_rss_mb = baseline, peak_disk_mb = 0,
  coefs = rename(coefs), ses = rename(ses),
  version = as.character(packageVersion(if (spec$library == "fixest") "fixest" else "stats")),
  threads = getFixest_nthreads()
), auto_unbox = TRUE, digits = NA), "\n", sep = "")
