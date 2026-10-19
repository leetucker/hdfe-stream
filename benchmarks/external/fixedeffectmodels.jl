# One regression with Julia's FixedEffectModels.jl, for benchmarks/_harness.py:
# read the Parquet file into memory (counted, as pandas is for pyfixest), fit
# with standard errors clustered as asked, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_rss_mb": ..., "coefs": {...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations. Julia
# compiles each method the first time it runs, so the same read and fit run
# first on a small panel (spec "warmup_path"): the times are for compiled
# code, as they are for numba in hdfe_stream, and the warm-up is reported on
# its own as t_warmup. Its memory, and the compiled code's, stay in the peak.
#
#   julia --threads=N --project=benchmarks/external/julia \
#       fixedeffectmodels.jl '<spec as JSON>'      (see external_spec in _harness.py)

using DataFrames, FixedEffectModels, JSON, Parquet2, StatsModels, Vcov
import Pkg

hwm_mb() = parse(Float64, split(only(filter(startswith("VmHWM:"),
    readlines("/proc/self/status"))))[2]) / 1024

const SPEC = JSON.parse(ARGS[1])

function read_panel(path)
    table = DataFrame(Parquet2.Dataset(path); copycols=false)
    columns = SPEC["columns"]
    columns === nothing ? copy(table) : table[:, Symbol.(columns)]
end

function fit(df; ref=nothing)
    rhs = Any[term(Symbol(c)) for c in SPEC["covariates"]]
    contrasts = Dict{Symbol, Any}()
    if SPEC["categorical"] !== nothing
        column = Symbol(SPEC["categorical"]["column"])
        push!(rhs, term(column))
        # the warm-up panel need not contain the full panel's reference level
        base = ref === nothing ? minimum(df[!, column]) : ref
        contrasts[column] = DummyCoding(base=base)
    end
    append!(rhs, [fe(Symbol(c)) for c in SPEC["fixed_effects"]])
    options = get(SPEC, "tol", nothing) === nothing ? (;) : (; tol=SPEC["tol"])
    model = reg(df, term(Symbol(SPEC["outcome"])) ~ foldl(+, rhs),
                Vcov.cluster(Symbol(SPEC["cluster"])); contrasts=contrasts,
                drop_singletons=SPEC["drop_singletons"], progress_bar=false,
                options...)
    stderror(model)
    model
end

baseline = hwm_mb()
t0 = time()
fit(read_panel(SPEC["warmup_path"]))
t_warmup = time() - t0
GC.gc()

t0 = time()
df = read_panel(SPEC["path"])
t_read = time() - t0

ref = SPEC["categorical"] === nothing ? nothing : SPEC["categorical"]["ref"]
t0 = time()
model = fit(df; ref=ref)
t_fit = time() - t0

version = Pkg.dependencies()[Base.UUID("9d5cd8c9-2029-5cab-9928-427838db53e3")].version
println("RESULT ", JSON.json(Dict(
    "t_read" => t_read, "t_fit" => t_fit, "n_obs" => nobs(model),
    "peak_rss_mb" => hwm_mb(), "baseline_rss_mb" => baseline, "peak_disk_mb" => 0,
    "t_warmup" => t_warmup, "coefs" => Dict(zip(coefnames(model), coef(model))),
    "version" => string(version), "threads" => Threads.nthreads())))
