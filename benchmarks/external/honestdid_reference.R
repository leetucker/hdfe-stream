# Reference values from the R package HonestDiD 0.2.8 for
# tests/test_core_sensitivity.py, written to tests/data/honestdid_reference.json.gz.
#
#   Rscript honestdid_reference.R <output.json.gz> [cores]
#
# The cases are the package's two example event studies (BCdata_EventStudy,
# LWdata_EventStudy) and simulated event studies of several shapes. For each
# case and each restriction, sign or monotonicity variant and confidence
# interval, it records R's grid of target values and which of them the test
# accepts (computeConditionalCS_*), or the FLCI (findOptimalFLCI).
#
# Two parts of the R code are simulated, and are made comparable here:
#
# - .qfoldednormal (the folded-normal quantile of the FLCI) estimates the
#   quantile from 10^6 draws. Section "exact" replaces it with the exact
#   quantile (uniroot), which is what hdfe_stream computes; section "stock"
#   leaves it alone, to check that the difference is within the simulation
#   error.
# - .compute_least_favorable_cv (the least-favorable hybrid's first-stage
#   critical value) uses 1000 draws of R's generator. Every call is recorded
#   with a fingerprint of its covariance, so that the Python test can use R's
#   critical value and compare every grid point exactly.
#
# Needs HonestDiD 0.2.8 and its dependencies (CVXR needs clarabel, which
# builds with Rust; Rglpk needs GLPK): see benchmarks/README.md.

suppressPackageStartupMessages({
  library(HonestDiD)
  library(parallel)
})
args <- commandArgs(TRUE)
out_path <- args[1]
cores <- if (length(args) > 1) as.integer(args[2]) else 1L
stopifnot(as.character(packageVersion("HonestDiD")) == "0.2.8")

qfold_exact <- function(p, mu = 0, sd = 1, numSims = 10^6, seed = 0) {
  vapply(mu, function(m) {
    m <- abs(m)
    f <- function(c) pnorm(c - m, sd = sd) - pnorm(-c - m, sd = sd) - p
    uniroot(f, c(0, m + 12 * sd), tol = 1e-15)$root
  }, numeric(1))
}
stock_qfold <- HonestDiD:::.qfoldednormal

orig_lf <- HonestDiD:::.compute_least_favorable_cv
rec <- new.env()
rec$items <- list()
lf_recorder <- function(X_T, sigma, hybrid_kappa, sims = 1000, rowsForARP = NULL, seed = 0) {
  v <- orig_lf(X_T, sigma, hybrid_kappa, sims, rowsForARP, seed)
  S <- sigma
  X <- X_T
  if (!is.null(rowsForARP)) {
    if (!is.null(X)) X <- if (is.vector(X)) X[rowsForARP] else X[rowsForARP, , drop = FALSE]
    S <- S[rowsForARP, rowsForARP]
  }
  rec$items[[length(rec$items) + 1]] <- list(
    sum = sum(S), trace = sum(diag(S)), xsum = if (is.null(X)) NA else sum(X),
    m = NROW(S), cv = v)
  v
}
assignInNamespace(".compute_least_favorable_cv", lf_recorder, "HonestDiD")

# ---- cases ---------------------------------------------------------------

data(BCdata_EventStudy)
data(LWdata_EventStudy)
cases <- list(
  bc = list(beta = as.numeric(BCdata_EventStudy$betahat), sigma = BCdata_EventStudy$sigma,
            n_pre = length(BCdata_EventStudy$prePeriodIndices)),
  lw = list(beta = as.numeric(LWdata_EventStudy$betahat), sigma = LWdata_EventStudy$sigma,
            n_pre = length(LWdata_EventStudy$prePeriodIndices))
)
simulate_case <- function(n_pre, n_post, seed) {
  set.seed(seed)
  n <- n_pre + n_post
  F <- matrix(rnorm(n * 2), n, 2)
  C <- cov2cor(F %*% t(F) + diag(n))
  se <- 0.05 * exp(rnorm(n, sd = 0.3))
  sigma <- diag(se) %*% C %*% diag(se)
  times <- c(-(n_pre:1), 1:n_post)
  trend <- 0.01 * times
  effect <- c(rep(0, n_pre), 0.15 + 0.02 * (1:n_post))
  beta <- as.numeric(trend + effect + t(chol(sigma)) %*% rnorm(n))
  list(beta = beta, sigma = (sigma + t(sigma)) / 2, n_pre = n_pre)
}
shapes <- list(c(1, 1), c(1, 3), c(2, 1), c(2, 2), c(3, 3), c(4, 2), c(5, 4))
for (i in seq_along(shapes)) {
  s <- shapes[[i]]
  cases[[sprintf("sim_%d_%d", s[1], s[2])]] <- simulate_case(s[1], s[2], 100 + i)
}

# ---- configurations ------------------------------------------------------

compute_fn <- list(
  smoothness = list(none = computeConditionalCS_DeltaSD, bias = computeConditionalCS_DeltaSDB,
                    monotonicity = computeConditionalCS_DeltaSDM),
  relative_magnitudes = list(none = computeConditionalCS_DeltaRM, bias = computeConditionalCS_DeltaRMB,
                             monotonicity = computeConditionalCS_DeltaRMM),
  smoothness_relative_magnitudes = list(none = computeConditionalCS_DeltaSDRM,
                                        bias = computeConditionalCS_DeltaSDRMB,
                                        monotonicity = computeConditionalCS_DeltaSDRMM)
)
flag <- c(conditional = "ARP", hybrid_fixed_length = "FLCI", hybrid_least_favorable = "LF")

jobs <- list()
add <- function(...) jobs[[length(jobs) + 1]] <<- list(...)
for (name in names(cases)) {
  cs <- cases[[name]]
  n_pre <- cs$n_pre
  n_post <- length(cs$beta) - n_pre
  targets <- list(first = c(1, rep(0, n_post - 1)))
  if (n_post > 1 && name %in% c("bc", "sim_3_3", "sim_5_4")) targets$average <- rep(1 / n_post, n_post)
  Mub <- if (n_pre > 1) DeltaSD_upperBound_Mpre(cs$beta, cs$sigma, n_pre) else sqrt(cs$sigma[1, 1])
  M_sd <- c(0, Mub / 3, Mub)
  M_rm <- c(0, 0.5, 1.5)
  for (tname in names(targets)) {
    l <- targets[[tname]]
    add(case = name, target = tname, l = l, restriction = "smoothness", method = "fixed_length",
        variant = "none", direction = NA, M = M_sd)
    variants <- list(c("none", NA), c("bias", "positive"), c("bias", "negative"))
    if (n_pre >= 2) variants <- c(variants, list(c("monotonicity", "increasing"),
                                                 c("monotonicity", "decreasing")))
    for (v in variants) {
      for (m in c("conditional", "hybrid_fixed_length", "hybrid_least_favorable")) {
        add(case = name, target = tname, l = l, restriction = "smoothness", method = m,
            variant = v[1], direction = v[2], M = M_sd)
      }
    }
    for (r in c("relative_magnitudes", "smoothness_relative_magnitudes")) {
      if (r == "smoothness_relative_magnitudes" && n_pre < 2) next
      for (v in variants) {
        methods <- if (v[1] == "none") c("conditional", "hybrid_least_favorable") else "hybrid_least_favorable"
        for (m in methods) {
          add(case = name, target = tname, l = l, restriction = r, method = m,
              variant = v[1], direction = v[2], M = M_rm)
        }
      }
    }
  }
}

run_job <- function(job) {
  cs <- cases[[job$case]]
  n_pre <- cs$n_pre
  n_post <- length(cs$beta) - n_pre
  out <- list()
  for (M in job$M) {
    rec$items <- list()
    t0 <- proc.time()[["elapsed"]]
    if (job$method == "fixed_length") {
      f <- findOptimalFLCI(betahat = cs$beta, sigma = cs$sigma, M = M, numPrePeriods = n_pre,
                           numPostPeriods = n_post, l_vec = job$l, alpha = 0.05)
      res <- list(M = M, lb = f$FLCI[1], ub = f$FLCI[2], vec = as.numeric(f$optimalVec),
                  half = f$optimalHalfLength)
    } else {
      fn <- compute_fn[[job$restriction]][[job$variant]]
      a <- list(betahat = cs$beta, sigma = cs$sigma, numPrePeriods = n_pre, numPostPeriods = n_post,
                l_vec = job$l, alpha = 0.05, hybrid_flag = flag[[job$method]])
      if (job$restriction == "smoothness") a$M <- M else a$Mbar <- M
      if (job$variant == "bias") a$biasDirection <- job$direction
      if (job$variant == "monotonicity") a$monotonicityDirection <- job$direction
      g <- tryCatch(suppressWarnings(do.call(fn, a)), error = function(e) e)
      if (inherits(g, "error")) {
        out[[length(out) + 1]] <- list(M = M, error = conditionMessage(g))
        next
      }
      acc <- g$accept
      res <- list(M = M, grid_lb = g$grid[1], grid_ub = g$grid[length(g$grid)], points = length(g$grid),
                  accept = paste(as.integer(acc), collapse = ""),
                  lb = if (any(acc == 1)) min(g$grid[acc == 1]) else NA,
                  ub = if (any(acc == 1)) max(g$grid[acc == 1]) else NA,
                  lf = rec$items)
    }
    res$seconds <- proc.time()[["elapsed"]] - t0
    out[[length(out) + 1]] <- res
  }
  job$results <- out
  job
}

# ---- exact section: exact folded-normal quantile, recorded LF critical values
assignInNamespace(".qfoldednormal", qfold_exact, "HonestDiD")
exact <- mclapply(jobs, run_job, mc.cores = cores, mc.preschedule = FALSE)

# ---- defaults: createSensitivityResults' default M grids and methods (exact quantile)
defaults <- list()
for (name in c("bc", "sim_3_3", "sim_1_3")) {
  cs <- cases[[name]]
  n_pre <- cs$n_pre
  n_post <- length(cs$beta) - n_pre
  sd <- suppressWarnings(createSensitivityResults(cs$beta, cs$sigma, n_pre, n_post))
  defaults[[length(defaults) + 1]] <- list(case = name, restriction = "smoothness",
                                           M = sd$M, lb = sd$lb, ub = sd$ub, method = sd$method)
}

# ---- stock section: R as distributed (simulated folded-normal quantile)
assignInNamespace(".qfoldednormal", stock_qfold, "HonestDiD")
stock_jobs <- Filter(function(j) j$method %in% c("fixed_length", "hybrid_fixed_length") &&
                       j$variant == "none" && j$target == "first" &&
                       j$case %in% c("bc", "lw", "sim_3_3", "sim_5_4"), jobs)
stock <- mclapply(stock_jobs, run_job, mc.cores = cores, mc.preschedule = FALSE)

payload <- list(
  generated_by = "benchmarks/external/honestdid_reference.R",
  honestdid = as.character(packageVersion("HonestDiD")),
  r = R.version.string,
  cases = cases, exact = exact, stock = stock, defaults = defaults
)
con <- gzfile(out_path, "w")
writeLines(jsonlite::toJSON(payload, digits = NA, auto_unbox = TRUE, matrix = "rowmajor",
                            na = "null"), con)
close(con)
cat("wrote", length(exact), "exact and", length(stock), "stock configurations to", out_path, "\n")
