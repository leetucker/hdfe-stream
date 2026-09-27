"""Draw the benchmark figures from the CSVs the benchmarks write.

    python benchmarks/plot_benchmarks.py

Reads benchmarks/results/akm.csv and kss.csv and writes, for each, three
figures -- wall time, peak memory, peak disk -- against the number of workers,
one line per configuration, in a light and a dark version:

    docs/figures/{akm,kss}_{time,memory,disk}.{light,dark}.svg

Only this file needs changing to restyle them; the numbers stay in the CSVs.

Colors follow the library, not the rank of the line, so a library looks the
same in every figure: xhdfe is aqua in both benchmarks, hdfe_stream's default
magenta, its low-memory setting violet. Every series also has its own marker,
so no line is identified by color alone. Points that failed or timed out are
left out, and listed in the note under the figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter,  # noqa: E402
                               LogLocator, NullFormatter, NullLocator)

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

# configuration -> (color slot, marker, line style)
SERIES = {
    "akm": {
        "pyfixest (MAP)": ("blue", "o", "-"),
        "pyfixest (LSMR)": ("orange", "s", "-"),
        "xhdfe": ("aqua", "D", "-"),
        "hdfe_stream (explicit)": ("yellow", "^", "-"),
        "hdfe_stream (stream_cg)": ("magenta", "v", "-"),
        "hdfe_stream (within)": ("green", "P", "-"),
        "hdfe_stream (stream_cg, low memory)": ("violet", "X", "-"),
    },
    # standard errors are the same library doing more work: same color, dashed
    "kss": {
        "xhdfe": ("aqua", "D", "-"),
        "xhdfe (standard errors)": ("aqua", "d", "--"),
        "hdfe_stream": ("magenta", "v", "-"),
        "hdfe_stream (standard errors)": ("magenta", "^", "--"),
        "hdfe_stream (low memory)": ("violet", "X", "-"),
    },
}

METRICS = {
    "time": ("wall_s", "Wall time (seconds)", 1.0),
    "memory": ("peak_memory_mb", "Peak memory (GB)", 1e-3),
    "disk": ("peak_disk_mb", "Peak disk (GB)", 1e-3),
}

TITLES = {
    "akm": "AKM regression",
    "kss": "KSS leave-out variance decomposition",
}


def read(name):
    with open(RESULTS / f"{name}.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def worker_label(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    return f"{n / 1_000:g}k"


def figure(bench, metric, theme_name, rows):
    column, ylabel, scale = METRICS[metric]
    theme = THEMES[theme_name]
    series = SERIES[bench]
    sizes = sorted({int(r["n_workers"]) for r in rows})
    n_rows = {int(r["n_workers"]): int(r["n_rows"]) for r in rows}

    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    skipped = []
    drawn = 0
    for label, (slot, marker, style) in series.items():
        points = []
        for r in rows:
            if r["configuration"] != label:
                continue
            if r["status"] != "ok":
                why = r["status"].split(":")[0]
                if why.startswith("timeout after"):
                    hours = float(why.split()[2]) / 3600
                    why = f"did not finish within {hours:g} hours"
                skipped.append(f"{label} at {worker_label(int(r['n_workers']))}"
                               f" workers ({why})")
                continue
            value = float(r[column] or 0) * scale
            if value <= 0:          # uses no disk: nothing to draw on a log axis
                continue
            points.append((int(r["n_workers"]), value))
        if not points:
            continue
        points.sort()
        color = SLOTS[slot][theme["slot"]]
        ax.plot([p[0] for p in points], [p[1] for p in points], style,
                color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=theme["surface"], markeredgewidth=1.2,
                label=label, zorder=3)
        drawn += 1

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([worker_label(n) for n in sizes])
    ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
    per_worker = sum(n_rows[n] / n for n in sizes) / len(sizes)
    ax.set_xlabel(f"Workers (firms = workers / 15; about {per_worker:.1f} rows "
                  "per worker)", color=theme["muted"])
    ax.set_ylabel(ylabel, color=theme["muted"])
    ax.set_title(f"{TITLES[bench]}: {ylabel.split(' (')[0].lower()}",
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

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       frameon=False, fontsize=9, handlelength=2.6)
    for text in legend.get_texts():
        text.set_color(theme["text"])

    notes = []
    if metric == "disk":
        absent = list(dict.fromkeys(label.split(" (")[0] for label in series
                                    if not label.startswith("hdfe_stream")))
        if absent:
            notes.append("Not shown, as they hold the data in memory and write "
                         "nothing to disk: " + ", ".join(absent) + ".")
        drawn_disk = [label for label in series if label.startswith("hdfe_stream")]
        if len(drawn_disk) > 1:
            notes.append("The hdfe_stream settings write almost the same "
                         "amount, so their lines overlap.")
    if skipped:
        notes.append("Missing points: " + "; ".join(skipped) + ".")
    if notes:
        fig.text(0.01, 0.01, "\n".join(notes), color=theme["muted"],
                 fontsize=8, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.05 * len(notes), 1, 1))
    return fig, drawn


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for bench in ("akm", "kss"):
        if not (RESULTS / f"{bench}.csv").exists():
            print(f"no results for {bench}; run benchmarks/{bench}_benchmark.py")
            continue
        rows = read(bench)
        for metric in METRICS:
            for theme in THEMES:
                fig, drawn = figure(bench, metric, theme, rows)
                path = FIGURES / f"{bench}_{metric}.{theme}.svg"
                fig.savefig(path, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"{path.relative_to(HERE.parent)} ({drawn} series)")


if __name__ == "__main__":
    main()
