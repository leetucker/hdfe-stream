"""Draw the benchmark figures from the CSVs the benchmarks write.

    python benchmarks/plot_benchmarks.py

Reads benchmarks/results/akm.csv, kss.csv, covariates.csv, glm.csv and
glm_covariates.csv and writes, for each, three figures -- wall time, peak
memory, peak disk -- against the number of workers (the number of covariates
for the covariates benchmarks), one line per configuration, in a light and a
dark version:

    docs/figures/<benchmark>_{time,memory,disk}.{light,dark}.svg

The GPU benchmark (gpu.csv, measured on its own machine) has wall time, peak
memory and peak GPU memory instead of disk: gpu_{time,memory,gpu_memory}.

The GLM figures have two panels, Poisson and logit, with one legend: a setting
has the same color in both.

Only this file needs changing to restyle them; the numbers stay in the CSVs.

Colors follow the library, not the rank of the line, so a library looks the
same in every figure: xhdfe is aqua in both benchmarks, hdfe_stream's default
magenta, its low-memory setting violet, fixest red, FixedEffectModels.jl green.
In the GPU benchmark a library's GPU configurations are dashed, in its color.
Every series also has its own marker, so no line is identified by color alone.
Points that failed or timed out are left out, and listed in the note under the
figure.

Only the configurations listed in SERIES are drawn. The CSVs keep others: the
AKM benchmark's "hdfe_stream (within)" is measured and kept in akm.csv, but not
drawn, as the default solver does better at every size but the smallest.
"""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter,  # noqa: E402
                               LogLocator, NullFormatter, NullLocator)

from _settings import LOW_MEMORY_NOTE, SIZED_NOTE, limit_note  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "docs" / "figures"

# the reference categorical palette, in its fixed slot order, light and dark
SLOTS = {
    "blue": ("#2a78d6", "#3987e5"), "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"), "yellow": ("#eda100", "#c98500"),
    "magenta": ("#e87ba4", "#d55181"), "green": ("#008300", "#008300"),
    "violet": ("#4a3aa7", "#9085e9"), "red": ("#e34948", "#e66767"),
}
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
              "grid": "#e4e3de", "axis": "#b9b8b1", "slot": 0},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
             "grid": "#34342f", "axis": "#5b5a55", "slot": 1},
}

# configuration -> (color slot, marker, line style[, name in the legend])
SERIES = {
    "akm": {
        "pyfixest (MAP)": ("blue", "o", "-"),
        "pyfixest (LSMR)": ("orange", "s", "-"),
        "xhdfe": ("aqua", "D", "-"),
        "hdfe_stream (explicit)": ("yellow", "^", "-"),
        "hdfe_stream (stream_cg)": ("magenta", "v", "-"),
        "hdfe_stream (stream_cg, low memory)": ("violet", "X", "-"),
        # told how much memory it may use: violet like the other memory-limited
        # setting, dashed to tell them apart
        "hdfe_stream (max_memory_gb=2)": ("violet", "P", "--"),
        "fixest (R)": ("red", "p", "-"),
        "FixedEffectModels.jl": ("green", "h", "-"),
    },
    # standard errors are the same library doing more work: same color, dashed
    "kss": {
        "xhdfe": ("aqua", "D", "-"),
        "xhdfe (standard errors)": ("aqua", "d", "--"),
        "hdfe_stream": ("magenta", "v", "-"),
        "hdfe_stream (standard errors)": ("magenta", "^", "--"),
        "hdfe_stream (low memory)": ("violet", "X", "-"),
    },
    "covariates": {
        "pyfixest (MAP)": ("blue", "o", "-"),
        "pyfixest (LSMR)": ("orange", "s", "-"),
        "xhdfe": ("aqua", "D", "-"),
        "hdfe_stream": ("magenta", "v", "-"),
        "hdfe_stream (sized to the design)": ("violet", "X", "-"),
        "fixest (R)": ("red", "p", "-"),
        "FixedEffectModels.jl": ("green", "h", "-"),
    },
    # each library on the CPU and, dashed, on the GPU
    "gpu": {
        "pyfixest (LSMR, CPU)": ("orange", "s", "-"),
        "pyfixest (LSMR, GPU)": ("orange", "P", "--"),
        "xhdfe (CPU)": ("aqua", "D", "-"),
        "xhdfe (GPU)": ("aqua", "d", "--"),
        "hdfe_stream (CPU)": ("magenta", "v", "-"),
        "hdfe_stream (GPU solve)": ("magenta", "^", "--"),
        "hdfe_stream (GPU solve + GPU Polars)": ("violet", "X", "--"),
    },
    "glm": {
        "pyfixest fepois (LSMR)": ("orange", "s", "-", "pyfixest (LSMR)"),
        "fepois_stream": ("magenta", "v", "-", "hdfe_stream"),
        "fepois_stream (explicit)": ("yellow", "^", "-", "hdfe_stream (explicit)"),
        "fepois_stream (low memory)": ("violet", "X", "-", "hdfe_stream (low memory)"),
        "pyfixest feglm logit (LSMR)": ("orange", "s", "-", "pyfixest (LSMR)"),
        "feglm_stream logit": ("magenta", "v", "-", "hdfe_stream"),
        "feglm_stream logit (explicit)": ("yellow", "^", "-", "hdfe_stream (explicit)"),
        "feglm_stream logit (low memory)": ("violet", "X", "-", "hdfe_stream (low memory)"),
    },
    "glm_covariates": {
        "pyfixest fepois (LSMR)": ("orange", "s", "-", "pyfixest (LSMR)"),
        "fepois_stream": ("magenta", "v", "-", "hdfe_stream"),
        "fepois_stream (sized to the design)": ("violet", "X", "-",
                                                "hdfe_stream (sized to the design)"),
        "pyfixest feglm logit (LSMR)": ("orange", "s", "-", "pyfixest (LSMR)"),
        "feglm_stream logit": ("magenta", "v", "-", "hdfe_stream"),
        "feglm_stream logit (sized to the design)": ("violet", "X", "-",
                                                     "hdfe_stream (sized to the design)"),
    },
    # general regression: a library has one color; the second panel's series
    # share the first's legend entry, and what does less work than the rest
    # (a floor of the normal equations, a by-hand fit) is dashed
    "ols": {
        "hdfe_stream": ("magenta", "v", "-"),
        "scikit-learn": ("green", "h", "-"),
        "polars-ols": ("aqua", "D", "-"),
        "polars-ds": ("yellow", "^", "-"),
        "statsmodels": ("blue", "o", "-"),
        "pyfixest": ("orange", "s", "-"),
        "fixest (R)": ("red", "p", "-"),
        "lm (R)": ("red", "h", "--"),
        "NumPy normal equations": ("violet", "X", "--"),
        "hdfe_stream (1 FE)": ("magenta", "v", "-", "hdfe_stream"),
        "pyfixest (1 FE)": ("orange", "s", "-", "pyfixest"),
        "fixest (R, 1 FE)": ("red", "p", "-", "fixest (R)"),
        "linearmodels AbsorbingLS (1 FE)": ("violet", "P", "-", "linearmodels"),
        "polars-ols (demeaned, 1 FE)": ("aqua", "D", "-", "polars-ols"),
    },
    "iv": {
        "hdfe_stream": ("magenta", "v", "-"),
        "linearmodels IV2SLS": ("violet", "P", "-", "linearmodels"),
        "pyfixest": ("orange", "s", "-"),
        "fixest (R)": ("red", "p", "-"),
        "polars-ols (two stages)": ("aqua", "D", "--", "polars-ols (two stages)"),
        "NumPy normal equations": ("violet", "X", "--"),
        "hdfe_stream (1 FE)": ("magenta", "v", "-", "hdfe_stream"),
        "pyfixest (1 FE)": ("orange", "s", "-", "pyfixest"),
        "fixest (R, 1 FE)": ("red", "p", "-", "fixest (R)"),
        "linearmodels IV2SLS (demeaned, 1 FE)": ("violet", "P", "-", "linearmodels"),
        "polars-ols (two stages, demeaned, 1 FE)": ("aqua", "D", "--",
                                                    "polars-ols (two stages)"),
    },
    "binary": {
        "hdfe_stream logit": ("magenta", "v", "-", "hdfe_stream"),
        "statsmodels logit": ("blue", "o", "-", "statsmodels"),
        "pyfixest logit": ("orange", "s", "-", "pyfixest"),
        "fixest logit (R)": ("red", "p", "-", "fixest (R)"),
        "glm logit (R)": ("red", "h", "--", "glm (R)"),
        "scikit-learn logit": ("green", "h", "-", "scikit-learn"),
        "glum logit": ("aqua", "D", "-", "glum"),
        "polars-ds logit": ("yellow", "^", "-", "polars-ds"),
        "hdfe_stream probit": ("magenta", "v", "-", "hdfe_stream"),
        "statsmodels probit": ("blue", "o", "-", "statsmodels"),
        "pyfixest probit": ("orange", "s", "-", "pyfixest"),
        "fixest probit (R)": ("red", "p", "-", "fixest (R)"),
        "glm probit (R)": ("red", "h", "--", "glm (R)"),
    },
    # each model on the CPU and, dashed, on the GPU; what the other libraries
    # have for it, on the same machine, share a legend entry
    "general_gpu": {
        "hdfe_stream OLS (CPU)": ("magenta", "v", "-", "hdfe_stream (CPU)"),
        "hdfe_stream IV (CPU)": ("magenta", "v", "-", "hdfe_stream (CPU)"),
        "hdfe_stream logit (CPU)": ("magenta", "v", "-", "hdfe_stream (CPU)"),
        "hdfe_stream OLS (GPU Polars)": ("violet", "P", "--", "hdfe_stream (GPU Polars)"),
        "hdfe_stream IV (GPU Polars)": ("violet", "P", "--", "hdfe_stream (GPU Polars)"),
        "hdfe_stream logit (GPU Polars)": ("violet", "P", "--", "hdfe_stream (GPU Polars)"),
        "hdfe_stream logit (GPU IRLS)": ("magenta", "^", "--", "hdfe_stream (GPU IRLS)"),
        "hdfe_stream logit (GPU IRLS + GPU Polars)": ("violet", "X", "--",
                                                      "hdfe_stream (GPU IRLS + GPU Polars)"),
        "polars-ols (CPU)": ("aqua", "D", "-", "polars-ols, polars-ds (CPU)"),
        "polars-ols two stages (CPU)": ("aqua", "D", "-", "polars-ols, polars-ds (CPU)"),
        "polars-ds logit (CPU)": ("aqua", "D", "-", "polars-ols, polars-ds (CPU)"),
        "cuML OLS": ("green", "h", "--", "cuML"),
        "cuML logit": ("green", "h", "--", "cuML"),
        "CuPy normal equations (OLS)": ("blue", "o", "--", "CuPy, by hand"),
        "CuPy 2SLS": ("blue", "o", "--", "CuPy, by hand"),
        "CuPy Newton logit": ("blue", "o", "--", "CuPy, by hand"),
    },
}

# benchmarks drawn as side-by-side panels: (panel title, value of `family`)
PANELS = {
    "glm": [("Poisson", "poisson"), ("Logit", "logit")],
    "glm_covariates": [("Poisson", "poisson"), ("Logit", "logit")],
    "ols": [("No fixed effects", "0"), ("One fixed effect", "1")],
    "iv": [("No fixed effects", "0"), ("One fixed effect", "1")],
    "binary": [("Logit", "logit"), ("Probit", "probit")],
    "general_gpu": [("OLS", "ols"), ("Two-stage least squares", "iv"), ("Logit", "logit")],
}
# the column that tells the panels apart, where it is not `family`
PANEL_COLUMN = {"ols": "fixed_effects", "iv": "fixed_effects", "binary": "task",
                "general_gpu": "task"}

METRICS = {
    "time": ("wall_s", "Wall time (seconds)", 1.0),
    "memory": ("peak_memory_mb", "Peak memory (GB)", 1e-3),
    "disk": ("peak_disk_mb", "Peak disk (GB)", 1e-3),
    "gpu_memory": ("peak_gpu_memory_mb", "Peak GPU memory (GB)", 1e-3),
}
# the figures of each benchmark; the others have time, memory and disk
FIGURES_OF = {"gpu": ("time", "memory", "gpu_memory"),
              "ols": ("time", "memory"), "iv": ("time", "memory"),
              "binary": ("time", "memory"),
              "general_gpu": ("time", "memory", "gpu_memory")}

TITLES = {
    "akm": "AKM regression",
    "kss": "KSS leave-out variance decomposition",
    "covariates": "Many covariates (8.5 million rows)",
    "glm": "Poisson and logit regression",
    "glm_covariates": "Poisson and logit, many covariates (8.5 million rows)",
    "gpu": "AKM regression, CPU and GPU",
    "ols": "OLS regression",
    "iv": "Two-stage least squares",
    "binary": "Logit and probit regression",
    "general_gpu": "OLS, IV and logit, CPU and GPU",
}

# the x axis: column, tick label, axis label (given the rows), point name
X_AXES = {
    "workers": ("n_workers",
                lambda n: f"{n / 1_000_000:g}M" if n >= 1_000_000 else f"{n / 1_000:g}k",
                lambda rows, sizes: "Workers (firms = workers / 15; about "
                f"{sum(rows[n] / n for n in sizes) / len(sizes):.1f} rows per worker)",
                "workers"),
    "rows": ("n_rows",
             lambda n: f"{n / 1_000_000:g}M" if n >= 1_000_000 else f"{n / 1_000:g}k",
             lambda rows, sizes: "Rows (10 covariates)", "rows"),
    "covariates": ("n_covariates", lambda n: f"{n:,}",
                   lambda rows, sizes: "Covariates (age indicators, from 5-year bins "
                   "to 1-month bins)", "covariates"),
}
AXIS = {"akm": "workers", "kss": "workers", "covariates": "covariates",
        "glm": "workers", "glm_covariates": "covariates", "gpu": "workers",
        "ols": "rows", "iv": "rows", "binary": "rows", "general_gpu": "rows"}


def read(name):
    with open(RESULTS / f"{name}.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def _in_memory(label):
    return label.startswith(("pyfixest", "xhdfe", "fixest", "FixedEffectModels"))


def figure(bench, metric, theme_name, rows):
    column, ylabel, scale = METRICS[metric]
    theme = THEMES[theme_name]
    series = SERIES[bench]
    x_col, tick_label, axis_label, point_name = X_AXES[AXIS[bench]]
    sizes = sorted({int(r[x_col]) for r in rows})
    n_rows = {int(r[x_col]): int(r.get("n_rows") or 0) for r in rows}
    panels = PANELS.get(bench, [(None, None)])

    plt.rcParams.update({
        "svg.fonttype": "none",
        "svg.hashsalt": "hdfe-stream",     # stable element ids: a replot of the
                                           # same numbers is the same file
        "font.family": "sans-serif",
        "font.size": 10,
    })
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=(7.4, 4.2) if len(panels) == 1 else (10.4, 4.4))
    fig.patch.set_facecolor(theme["surface"])

    skipped = {}                    # reason -> configuration -> x labels
    drawn = 0
    legend_lines = {}               # legend name -> line, first panel it appears in
    for ax, (panel, family) in zip(axes[0], panels):
        ax.set_facecolor(theme["surface"])
        for label, (slot, marker, style, *name) in series.items():
            name = name[0] if name else label
            where = f"{name}, {panel}" if panel else label
            points = []
            for r in rows:
                if r["configuration"] != label or (
                        family and r[PANEL_COLUMN.get(bench, "family")] != family):
                    continue
                if r["status"] != "ok":
                    why = r["status"].split(":")[0]
                    if why.startswith("timeout after"):
                        hours = float(why.split()[2]) / 3600
                        why = f"did not finish within {hours:g} hours"
                    if why.startswith("killed"):
                        why = "out of memory"
                    skipped.setdefault(why, {}).setdefault(where, []).append(
                        tick_label(int(r[x_col])))
                    continue
                value = float(r[column] or 0) * scale
                if value <= 0:          # uses no disk: nothing to draw on a log axis
                    continue
                points.append((int(r[x_col]), value))
            if not points:
                continue
            points.sort()
            color = SLOTS[slot][theme["slot"]]
            line, = ax.plot([p[0] for p in points], [p[1] for p in points], style,
                            color=color, linewidth=2, marker=marker, markersize=7,
                            markeredgecolor=theme["surface"], markeredgewidth=1.2,
                            label=name, zorder=3)
            legend_lines.setdefault(name, line)
            drawn += 1

        if metric == "memory" and (limit := _memory_limit(bench)):
            # what the in-memory libraries run into; a recessive rule, not a series
            what, limit = limit
            ax.axhline(limit, color=theme["axis"], linewidth=1, linestyle=(0, (4, 3)),
                       zorder=1)
            ax.annotate(f"{what} ({limit:g} GB)", xy=(0.01, limit),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", fontsize=8, color=theme["muted"],
                        va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([tick_label(n) for n in sizes])
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        if not panel:
            ax.set_xlabel(axis_label(n_rows, sizes), color=theme["muted"])
        if ax is axes[0][0]:
            ax.set_ylabel(ylabel, color=theme["muted"])
        if panel:
            ax.set_title(panel, color=theme["text"], loc="left", fontsize=10, pad=6)
        else:
            ax.set_title(f"{_title(bench)}: {_lower(ylabel.split(' (')[0])}",
                         color=theme["text"], loc="left", fontsize=11, pad=10)
        # label 1-2-5 steps in plain decimals; leave the other minor ticks bare
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
        ax.grid(True, which="minor", axis="y", color=theme["grid"], linewidth=0.4,
                zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["axis"])
        ax.tick_params(colors=theme["muted"], which="both")

    legend = axes[0][-1].legend(legend_lines.values(), legend_lines.keys(),
                                loc="upper left", bbox_to_anchor=(1.01, 1.0),
                                frameon=False, fontsize=9, handlelength=2.6)
    for text in legend.get_texts():
        text.set_color(theme["text"])

    notes = []
    names = list(dict.fromkeys(entry[3] if len(entry) > 3 else label
                               for label, entry in series.items()))
    if metric == "disk":
        absent = list(dict.fromkeys(name.split(" (")[0] for name in names
                                    if _in_memory(name)))
        if absent:
            notes.append("Not shown, as they hold the data in memory and write "
                         "nothing to disk: " + ", ".join(absent) + ".")
        drawn_disk = [name for name in names if not _in_memory(name)]
        if len(drawn_disk) > 1:
            notes.append("The hdfe_stream settings write almost the same "
                         "amount, so their lines overlap.")
    if metric == "gpu_memory":
        notes.append("The CPU configurations use no GPU memory and are not shown.")
    # what the memory settings in the legend mean, so the figure carries its own definitions:
    # only for the series drawn (a configuration with no results is not in the legend)
    shown = list(legend_lines)
    if any("low memory" in name for name in shown):
        notes.append(LOW_MEMORY_NOTE)
    if any("sized to the design" in name for name in shown):
        notes.append(SIZED_NOTE)
    for name in shown:
        limit = re.search(r"max_memory_gb=([0-9.]+)", name)
        if limit:
            notes.append(limit_note(limit.group(1)))
    for why, where in skipped.items():
        notes.append(f"Missing ({why}): " + "; ".join(
            f"{label} at {_and(xs)}" for label, xs in where.items())
            + f" {point_name}.")
    wide = len(panels) > 1
    lines = [line for note in notes for line in textwrap.wrap(note, 175 if wide else 115)]
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), color=theme["muted"],
                 fontsize=8, ha="left", va="bottom")
    bottom, top = 0.035 * len(lines) + (0.01 if lines else 0), 1
    if wide:
        fig.suptitle(f"{_title(bench)}: {_lower(ylabel.split(' (')[0])}",
                     color=theme["text"], x=0.01, ha="left", fontsize=11)
        # one axis label for both panels, above the notes
        fig.supxlabel(axis_label(n_rows, sizes), color=theme["muted"], fontsize=10,
                      y=bottom + 0.01, va="bottom")
        bottom, top = bottom + 0.06, 0.97
    fig.tight_layout(rect=(0, bottom, 1, top))
    return fig, drawn


def _and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _lower(text):
    """'Peak GPU memory' -> 'peak GPU memory'."""
    return text[:1].lower() + text[1:]


def _machine(bench):
    name = {"gpu": "gpu_machine.json", "general_gpu": "general_gpu_machine.json",
            "ols": "general_machine.json", "iv": "general_machine.json",
            "binary": "general_machine.json"}.get(bench, "machine.json")
    try:
        return json.loads((RESULTS / name).read_text())
    except (OSError, ValueError):
        return {}


def _title(bench):
    """The GPU benchmark's title names its GPU, as gpu_machine.json has it."""
    if bench not in ("gpu", "general_gpu"):
        return TITLES[bench]
    gpus = _machine(bench).get("gpus") or [{}]
    model = (gpus[0].get("name") or "").removeprefix("NVIDIA ").split(" ")[0]
    return f"{TITLES[bench]} (NVIDIA {model})" if model else TITLES[bench]


def _memory_limit(bench):
    """(what, GB) for the rule on a memory figure: the machine's memory, or,
    on the GPU machine, which has far more, the limit its benchmarks ran
    under (the CPU machine's, so that its configurations had the same room)."""
    info = _machine(bench)
    if bench not in ("gpu", "general_gpu"):
        return ("this machine's memory", info["memory_gb"]) if info.get("memory_gb") else None
    cap = (info.get("ec2") or {}).get("memory_max", "").split(" ")[0]
    if cap.endswith("M") and cap[:-1].isdigit():
        return "the benchmarks' memory limit", int(cap[:-1]) / 1000
    return None


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for bench in ("akm", "kss", "covariates", "glm", "glm_covariates", "gpu",
                  "ols", "iv", "binary", "general_gpu"):
        if not (RESULTS / f"{bench}.csv").exists():
            print(f"no results for {bench}; run benchmarks/{bench}_benchmark.py")
            continue
        rows = read(bench)
        for metric in FIGURES_OF.get(bench, ("time", "memory", "disk")):
            for theme in THEMES:
                fig, drawn = figure(bench, metric, theme, rows)
                path = FIGURES / f"{bench}_{metric}.{theme}.svg"
                fig.savefig(path, facecolor=fig.get_facecolor(),
                            metadata={"Date": None})
                plt.close(fig)
                print(f"{path.relative_to(HERE.parent)} ({drawn} series)")


if __name__ == "__main__":
    main()
