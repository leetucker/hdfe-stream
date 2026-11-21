# HonestDiD 0.2.8 and TruncatedNormal 2.3, which are not on conda-forge, from the CRAN
# snapshot of 2026-10-15, into the R environment of environment.yml (whose packages
# are their other dependencies, so nothing else is downloaded):
#
#   benchmarks/external/.r/bin/Rscript benchmarks/external/install_honestdid.R
# the environment's compilers (bin/ beside lib/R), which R's Makeconf names
Sys.setenv(PATH = paste(normalizePath(file.path(R.home(), "..", "..", "bin")),
                        Sys.getenv("PATH"), sep = .Platform$path.sep))
snapshot <- "https://packagemanager.posit.co/cran/2026-10-15"
options(Ncpus = parallel::detectCores())
install.packages(c("TruncatedNormal", "HonestDiD"), repos = snapshot,
                 dependencies = c("Depends", "Imports", "LinkingTo"))
stopifnot(as.character(packageVersion("HonestDiD")) == "0.2.8",
          as.character(packageVersion("TruncatedNormal")) == "2.3")
cat("HonestDiD", as.character(packageVersion("HonestDiD")), "\n")
