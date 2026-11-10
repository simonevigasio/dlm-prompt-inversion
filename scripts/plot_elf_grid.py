#!/usr/bin/env python3
"""Plot the results of the ELF experiment grid (scripts/run_elf_grid.py).

    .venv/bin/python scripts/plot_elf_grid.py                       # reads outputs/elf_grid
    .venv/bin/python scripts/plot_elf_grid.py --results outputs/elf_grid --output plots

Needs matplotlib:  .venv/bin/python -m pip install -e '.[plot]'

It reads <results>/<configuration>/results.jsonl and manifest.json of every configuration
(finished or not), and writes:

    <results>/results.jsonl          every row of every configuration, with its settings added
    <output>/summary.csv             one line per configuration: the numbers behind the plots

one set of figures per prompt length <n> and observed_rows setting <rows> (all or generated):

    <output>/overview_exact_recovery_tokens-<n>_rows-<rows>.png, overview_token_accuracy_tokens-<n>_rows-<rows>.png
                                                 heatmap of one score over all configurations
    <output>/exact_recovery_tokens-<n>_rows-<rows>.png      share of runs whose prompt was recovered exactly
    <output>/token_accuracy_tokens-<n>_rows-<rows>.png      share of prompt tokens recovered
    <output>/stage_a_convergence_tokens-<n>_rows-<rows>.png share of runs where Stage A converged
    <output>/prompt_code_error_tokens-<n>_rows-<rows>.png   relative error of the prompt codes after Stage A
    <output>/time_error_tokens-<n>_rows-<rows>.png          error of the recovered time (unknown-time configurations)
    <output>/runtime_tokens-<n>_rows-<rows>.png             seconds per run

and, when there are several prompt lengths, the two headline scores against the prompt length
(one line per time of the observed call, known time):

    <output>/by_prompt_length_exact_recovery_rows-<rows>.png, by_prompt_length_token_accuracy_rows-<rows>.png

The settings of a configuration come from its manifest.json, not from its folder name.
In the other figures the x axis is the time of the observed sampler call (step / 64), with
one panel per victim model and layer and one line per knowledge setting.
Rates and means are over completed runs: first the mean over the repetitions of each prompt,
then the mean over prompts. Vertical lines are 95% bootstrap intervals over prompts (the
repetitions of one prompt are not independent). Errors are shown as the median over prompts
with the 25%-75% range, on a log axis.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Panels: the victim model. Keys are (weights, bottleneck).
BACKBONES = {("checkpoint", True): "Checkpoint, bottleneck",
             ("random", True): "Random weights, bottleneck",
             ("random", False): "Random weights, no bottleneck"}
# Series: what the attacker knows. Keys are (knows_time, knows_noise_scale).
# Each one keeps its color and marker in every figure.
KNOWLEDGE = {(True, True): ("time and noise scale known", "#2a78d6", "o"),
             (True, False): ("time known, scale unknown", "#eb6834", "s"),
             (False, True): ("time unknown, scale known", "#1baf7a", "^"),
             (False, False): ("time and scale unknown", "#eda100", "D")}
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ELF_SAMPLING_STEPS = 64  # ELF's time of sampler call k is k / 64
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlesize": 10.5,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "ytick.major.size": 0, "xtick.major.size": 0,
    "legend.frameon": False, "savefig.dpi": 200,
})


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # the half-written last line of a run that is still going
                pass
    return rows


def load(results_root):
    """One dict per configuration that has results: its settings and its rows."""
    configurations = []
    for results in sorted(results_root.glob("*/results.jsonl")):
        folder = results.parent
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            print(f"skipping {folder.name}: no manifest.json")
            continue
        settings = json.loads(manifest.read_text())["configuration"]
        knowledge, backbone = settings["attacker_knowledge"], settings["backbone"]
        factors = {"configuration": folder.name,
                   "layer": settings["synthetic_observation"]["observation_layer"],
                   "step": settings["synthetic_observation"].get("step", 0),
                   "sampling_steps": settings["synthetic_observation"].get("sampling_steps") or ELF_SAMPLING_STEPS,
                   "observed_rows": knowledge["observed_rows"],
                   "knows_time": knowledge["knows_time"], "knows_noise_scale": knowledge["knows_noise_scale"],
                   "weights": backbone["weights"], "bottleneck": backbone["bottleneck"]}
        if (factors["weights"], factors["bottleneck"]) not in BACKBONES:
            print(f"skipping {folder.name}: unexpected backbone {backbone}")
            continue
        factors["time"] = factors["step"] / factors["sampling_steps"]
        rows = read_jsonl(results)
        # Prompt length: what the pipeline counted, else the "tokens-<n>" of the folder name.
        counts = [row["prompt_token_count"] for row in rows if "prompt_token_count" in row]
        named = re.match(r"tokens-(\d+)_", folder.name)
        factors["prompt_tokens"] = (max(set(counts), key=counts.count) if counts
                                    else int(named.group(1)) if named else None)
        try:
            pending = json.loads((folder / "summary.json").read_text())["pending"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pending = None
        configurations.append({**factors, "rows": rows, "pending": pending})
    return configurations


def write_combined(configurations, path):
    """Every row of every configuration in one file; the settings are added to each row."""
    with open(path, "w", encoding="utf-8") as handle:
        for c in configurations:
            factors = {k: v for k, v in c.items() if k not in ("rows", "pending")}
            for row in c["rows"]:
                handle.write(json.dumps({**factors, **row}, ensure_ascii=False) + "\n")


def per_prompt(rows, value):
    """Mean of ``value(row)`` over the completed repetitions of each prompt; None values are left out."""
    by_prompt = {}
    for row in rows:
        if row["status"] != "done":
            continue
        v = value(row)
        if v is not None:
            by_prompt.setdefault(row["example_id"], []).append(float(v))
    return np.array([np.mean(values) for values in by_prompt.values()])


def mean_interval(values, rng):
    """Mean and its 95% bootstrap interval over prompts."""
    if not len(values):
        return None
    draws = rng.choice(values, size=(2000, len(values))).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def median_quartiles(values, rng):
    """Median and the 25%-75% range over prompts."""
    if not len(values):
        return None
    return float(np.median(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))


# name -> (value of one row, statistic over prompts)
METRICS = {
    "exact_recovery": (lambda r: r["exact_recovery"], mean_interval),
    "token_accuracy": (lambda r: r["token_accuracy"], mean_interval),
    "stage_a_convergence": (lambda r: r["stage_a"]["converged"], mean_interval),
    "prompt_code_error": (lambda r: r.get("prompt_code_relative_error"), median_quartiles),
    "time_error": (lambda r: r.get("time_error"), median_quartiles),
    "runtime": (lambda r: r["runtime_seconds"], mean_interval),
}


def summarize(configurations):
    """Adds c["stats"][metric] = (center, low, high) or None, and the run counts."""
    rng = np.random.default_rng(0)
    for c in configurations:
        c["completed"] = sum(r["status"] == "done" for r in c["rows"])
        c["skipped"] = sum(r["status"] == "skipped" for r in c["rows"])
        c["stats"] = {name: statistic(per_prompt(c["rows"], value), rng)
                      for name, (value, statistic) in METRICS.items()}


def write_csv(configurations, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "prompt_tokens", "layer", "step", "time", "observed_rows", "knows_time", "knows_noise_scale", "weights",
                         "bottleneck", "completed", "skipped", "pending",
                         *(f"{name}{suffix}" for name in METRICS for suffix in ("", "_low", "_high"))])
        for c in configurations:
            writer.writerow([c["configuration"], c["prompt_tokens"], c["layer"], c["step"], f"{c['time']:.4g}", c["observed_rows"], c["knows_time"],
                             c["knows_noise_scale"], c["weights"], c["bottleneck"], c["completed"], c["skipped"],
                             "" if c["pending"] is None else c["pending"],
                             *("" if x is None else f"{x:.6g}"
                               for name in METRICS for x in (c["stats"][name] or (None,) * 3))])


def lookup(configurations):
    return {(c["weights"], c["bottleneck"], c["layer"], c["step"], c["knows_time"], c["knows_noise_scale"]): c
            for c in configurations}


def possible(step, knows_time):
    """The first call is always at time 0, so the grid has no unknown-time configuration there."""
    return step > 0 or knows_time


def knowledge_present(configurations):
    """The knowledge settings that appear in the results, in the order of KNOWLEDGE."""
    seen = {(c["knows_time"], c["knows_noise_scale"]) for c in configurations}
    return [k for k in KNOWLEDGE if k in seen]


def knowledge_label(key, settings):
    """Name of a knowledge setting; only the time is named when the noise scale is not varied."""
    if len({scale for _, scale in settings}) == 1:
        return "time known" if key[0] else "time unknown"
    return KNOWLEDGE[key][0]


def axes_values(configurations):
    """Layers and (step, time) pairs present in the results, sorted."""
    layers = sorted({c["layer"] for c in configurations})
    steps = sorted({(c["step"], c["time"]) for c in configurations})
    return layers, steps


def coverage_note(configurations):
    """Footnote of a figure: says when some configurations are missing or unfinished."""
    layers, steps = axes_values(configurations)
    expected = len(BACKBONES) * len(layers) * sum(possible(step, k[0]) for step, _ in steps
                                                   for k in knowledge_present(configurations))
    unfinished = sum(c["pending"] != 0 for c in configurations)
    notes = []
    if len(configurations) < expected:
        notes.append(f"{expected - len(configurations)} of {expected} configurations have no results yet")
    if unfinished:
        notes.append(f"{unfinished} unfinished (plotted with the runs completed so far)")
    return "Note: " + "; ".join(notes) + "." if notes else ""


def finish(figure, title, subtitle, note, path):
    """Title, subtitle and footnote at fixed distances (in inches) from the figure's edges."""
    height = figure.get_figheight()
    figure.suptitle(title, x=0.02, y=1 - 0.08 / height, ha="left", va="top", fontsize=13, fontweight="bold")
    figure.text(0.02, 1 - 0.38 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)
    if note:
        figure.text(0.02, 0.06 / height, note, ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


def time_label(time):
    return f"{time:.2f}".rstrip("0").rstrip(".") if time else "0"


def plot_metric(configurations, metric, title, subtitle, ylabel, path, *, scale="linear", knowledge=None):
    """Small multiples: rows = victim model, columns = layer, x = time of the observed call,
    one line per knowledge setting, with its interval (see METRICS) as a vertical line.

    scale "share": 0-100%. "log": log axis (errors). "linear": from zero (seconds).
    """
    table = lookup(configurations)
    layers, steps = axes_values(configurations)
    series_all = knowledge_present(configurations)
    series = [k for k in series_all if knowledge is None or k in knowledge]
    height = 1.5 + 2.6 * len(BACKBONES)
    figure, axes = plt.subplots(len(BACKBONES), len(layers), figsize=(12.5, height), sharex=True, sharey=True,
                                squeeze=False)
    for row, ((weights, bottleneck), panel) in zip(axes, BACKBONES.items()):
        for axis, layer in zip(row, layers):
            any_data = False
            for index, key in enumerate(series):
                (_, color, marker), label = KNOWLEDGE[key], knowledge_label(key, series_all)
                xs, centers, lows, highs = [], [], [], []
                for step, time in steps:
                    c = table.get((weights, bottleneck, layer, step, *key))
                    stat = c and c["stats"][metric]
                    if stat:
                        # A small shift keeps the intervals of the series from hiding each other.
                        xs.append(time + (index - (len(series) - 1) / 2) * 0.012)
                        centers.append(stat[0]), lows.append(stat[0] - stat[1]), highs.append(stat[2] - stat[0])
                if not xs:
                    continue
                any_data = True
                axis.errorbar(xs, centers, yerr=[lows, highs], marker=marker, color=color, linewidth=2,
                              markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1, ecolor=color,
                              elinewidth=1.2, capsize=0, label=label)
            if not any_data:
                axis.text(0.5, 0.5, "no results yet", transform=axis.transAxes, ha="center", va="center",
                          color=MUTED)
            axis.set_title(f"{panel} · layer {layer}", loc="left", color=INK, fontsize=9.5)
            axis.set_xticks([time for _, time in steps], [time_label(time) for _, time in steps])
            axis.set_xlim(-0.06, 1.02)
            axis.spines["bottom"].set_color(AXIS)
            axis.tick_params(which="minor", length=0)
        row[0].set_ylabel(ylabel)
    for axis in axes[-1]:
        axis.set_xlabel("time of the observed sampler call")
    if scale == "log":
        axes[0, 0].set_yscale("log")
    elif scale == "share":
        axes[0, 0].set_ylim(-0.04, 1.06)
        axes[0, 0].set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    else:
        axes[0, 0].set_ylim(bottom=0)
    handles = {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(handles.values(), handles.keys(), loc="upper left",
                  bbox_to_anchor=(0.015, 1 - 0.62 / height), ncols=len(series), handlelength=2.2,
                  columnspacing=1.8, fontsize=9.5)
    figure.subplots_adjust(left=0.07, right=0.985, top=1 - 1.3 / height, bottom=0.75 / height,
                           wspace=0.06, hspace=0.32)
    finish(figure, title, subtitle, coverage_note(configurations), path)


# Series of the prompt-length figures: the time of the observed call, light to dark.
TIME_STYLES = [("#86b6ef", "o"), ("#3987e5", "s"), ("#184f95", "^"), ("#0d366b", "D"), ("#6da7ec", "v")]


def plot_by_length(configurations, metric, title, subtitle, ylabel, path):
    """Small multiples: rows = victim model, columns = layer, x = prompt length, one line per
    time of the observed call. Known time only, so that every time has a line."""
    known = [c for c in configurations if c["knows_time"]]
    table = {(c["weights"], c["bottleneck"], c["layer"], c["step"], c["prompt_tokens"]): c for c in known}
    layers, steps = axes_values(known)
    lengths = sorted({c["prompt_tokens"] for c in known})
    height = 1.5 + 2.6 * len(BACKBONES)
    figure, axes = plt.subplots(len(BACKBONES), len(layers), figsize=(12.5, height), sharex=True, sharey=True,
                                squeeze=False)
    shift = 0.012 * (lengths[-1] - lengths[0])
    for row, ((weights, bottleneck), panel) in zip(axes, BACKBONES.items()):
        for axis, layer in zip(row, layers):
            for index, (step, time) in enumerate(steps):
                color, marker = TIME_STYLES[index % len(TIME_STYLES)]
                xs, centers, lows, highs = [], [], [], []
                for length in lengths:
                    c = table.get((weights, bottleneck, layer, step, length))
                    stat = c and c["stats"][metric]
                    if stat:
                        xs.append(length + (index - (len(steps) - 1) / 2) * shift)
                        centers.append(stat[0]), lows.append(stat[0] - stat[1]), highs.append(stat[2] - stat[0])
                if xs:
                    axis.errorbar(xs, centers, yerr=[lows, highs], marker=marker, color=color, linewidth=2,
                                  markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1, ecolor=color,
                                  elinewidth=1.2, capsize=0, label=f"time {time_label(time)}")
            if not axis.has_data():
                axis.text(0.5, 0.5, "no results yet", transform=axis.transAxes, ha="center", va="center",
                          color=MUTED)
            axis.set_title(f"{panel} · layer {layer}", loc="left", color=INK, fontsize=9.5)
            axis.set_xticks(lengths, [str(length) for length in lengths])
            axis.spines["bottom"].set_color(AXIS)
        row[0].set_ylabel(ylabel)
    for axis in axes[-1]:
        axis.set_xlabel("prompt length (tokens)")
    axes[0, 0].set_xlim(lengths[0] - 6 * shift, lengths[-1] + 6 * shift)
    axes[0, 0].set_ylim(-0.04, 1.06)
    axes[0, 0].set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    handles = {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(handles.values(), handles.keys(), loc="upper left", bbox_to_anchor=(0.015, 1 - 0.62 / height),
                  ncols=len(steps), handlelength=2.2, columnspacing=1.8, fontsize=9.5)
    figure.subplots_adjust(left=0.07, right=0.985, top=1 - 1.3 / height, bottom=0.75 / height,
                           wspace=0.06, hspace=0.32)
    finish(figure, title, subtitle, "", path)


def plot_overview(configurations, metric, title, path):
    """Heatmap of one score: rows = victim model x attacker knowledge, columns = layer x time."""
    table = lookup(configurations)
    layers, steps = axes_values(configurations)
    settings = knowledge_present(configurations)
    rows = [(b, k) for b in BACKBONES for k in settings]
    columns = [(layer, step, time) for layer in layers for step, time in steps]
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    colormap.set_bad("#f0efec")
    figure, axis = plt.subplots(figsize=(max(9.5, 4.6 + 0.56 * len(columns)), 2.2 + 0.43 * len(rows)))
    grid = np.full((len(rows), len(columns)), np.nan)
    for i, (backbone, knowledge) in enumerate(rows):
        for j, (layer, step, _) in enumerate(columns):
            c = table.get((*backbone, layer, step, *knowledge))
            if c and c["stats"][metric]:
                grid[i, j] = c["stats"][metric][0]
    image = axis.imshow(np.ma.masked_invalid(grid), cmap=colormap, vmin=0, vmax=1, aspect="auto")
    for i, (_, knowledge) in enumerate(rows):
        for j, (_, step, _) in enumerate(columns):
            value = grid[i, j]
            if not possible(step, knowledge[0]):
                # Not part of the grid: an empty cell instead of a "no results yet" one.
                axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SURFACE, edgecolor="none"))
                continue
            axis.text(j, i, "–" if np.isnan(value) else f"{value:.0%}", ha="center", va="center", fontsize=8.5,
                      color=MUTED if np.isnan(value) else "#ffffff" if value > 0.5 else INK)
    axis.set_xticks(range(len(columns)), [time_label(time) for _, _, time in columns])
    axis.set_yticks(range(len(rows)), [knowledge_label(k, settings) for _, k in rows])
    axis.set_xlabel("time of the observed sampler call")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    # White lines between cells, wider ones between victim models and between layers.
    axis.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="minor", length=0)
    # A little room so the outer cell borders are drawn whole and cover the image edge.
    axis.set_xlim(-0.53, len(columns) - 0.47)
    axis.set_ylim(len(rows) - 0.47, -0.53)
    for boundary in range(len(settings), len(rows), len(settings)):
        axis.axhline(boundary - 0.5, color=SURFACE, linewidth=8)
    for index, layer in enumerate(layers):
        if index:
            axis.axvline(index * len(steps) - 0.5, color=SURFACE, linewidth=8)
        axis.text(index * len(steps) - 0.45, -0.62, f"layer {layer}", ha="left", va="bottom", fontsize=10, color=INK)
    for index, panel in enumerate(BACKBONES.values()):
        axis.annotate(panel.replace(", ", ",\n"), xy=(0.02, (index + 0.5) * len(settings) - 0.5),
                      xycoords=("figure fraction", "data"), ha="left", va="center",
                      fontsize=9.5, fontweight="bold", color=INK, annotation_clip=False)
    width = figure.get_figwidth()
    height = figure.get_figheight()
    figure.subplots_adjust(left=4.0 / width, right=1 - 0.95 / width, top=1 - 1.15 / height, bottom=0.85 / height)
    bar = figure.colorbar(image, cax=figure.add_axes([1 - 0.7 / width, 0.3, 0.14 / width, 0.36]), ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["0%", "50%", "100%"])
    bar.outline.set_visible(False)
    finish(figure, title, "Mean over prompts of the completed runs. "
           "Empty cells at time 0: the first call is always at time 0, so there is no time to recover.",
           coverage_note(configurations), path)


def main():
    parser = argparse.ArgumentParser(description="Plot the results of the ELF experiment grid.")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "elf_grid",
                        help="Folder written by scripts/run_elf_grid.py (default: outputs/elf_grid)")
    parser.add_argument("--output", type=Path, help="Folder for the figures (default: <results>/plots)")
    args = parser.parse_args()
    output = args.output or args.results / "plots"

    configurations = load(args.results)
    if not configurations:
        raise SystemExit(f"No <configuration>/results.jsonl with a manifest.json in {args.results}")
    output.mkdir(parents=True, exist_ok=True)
    write_combined(configurations, args.results / "results.jsonl")
    summarize(configurations)
    write_csv(configurations, output / "summary.csv")
    rows = sum(len(c["rows"]) for c in configurations)
    print(f"{len(configurations)} configurations, {rows} rows -> {args.results / 'results.jsonl'}")

    interval = "Markers: mean over prompts; vertical lines: 95% bootstrap interval."
    quartiles = "Markers: median over prompts; vertical lines: 25%-75% range. Log axis."
    lengths = sorted({c["prompt_tokens"] for c in configurations if c["prompt_tokens"] is not None})
    # One set of figures per prompt length and observed_rows setting.
    for rows in sorted({c["observed_rows"] for c in configurations}):
        same_rows = [c for c in configurations if c["observed_rows"] == rows]
        for tokens in lengths:
            subset = [c for c in same_rows if c["prompt_tokens"] == tokens]
            if not subset:
                continue
            observed = f"{tokens}-token prompts, {'all' if rows == 'all' else 'only generated'} rows observed"
            plot_overview(subset, "exact_recovery", f"ELF exact prompt recovery ({observed})",
                          output / f"overview_exact_recovery_tokens-{tokens}_rows-{rows}.png")
            plot_overview(subset, "token_accuracy", f"ELF token accuracy ({observed})",
                          output / f"overview_token_accuracy_tokens-{tokens}_rows-{rows}.png")
            plot_metric(subset, "exact_recovery", f"Exact prompt recovery ({observed})",
                        f"Share of runs whose recovered tokens equal the prompt. {interval}",
                        "share of runs", output / f"exact_recovery_tokens-{tokens}_rows-{rows}.png", scale="share")
            plot_metric(subset, "token_accuracy", f"Token accuracy ({observed})",
                        f"Share of prompt tokens recovered at the right position. {interval}",
                        "share of tokens", output / f"token_accuracy_tokens-{tokens}_rows-{rows}.png", scale="share")
            plot_metric(subset, "stage_a_convergence", f"Stage A convergence ({observed})",
                        f"Share of runs where Stage A reached its residual tolerance. {interval}",
                        "share of runs", output / f"stage_a_convergence_tokens-{tokens}_rows-{rows}.png", scale="share")
            plot_metric(subset, "prompt_code_error", f"Prompt code error after Stage A ({observed})",
                        f"Distance between recovered and true prompt codes, divided by the norm of the true codes. {quartiles}",
                        "relative error", output / f"prompt_code_error_tokens-{tokens}_rows-{rows}.png", scale="log")
            if any(c["stats"]["time_error"] for c in subset):
                plot_metric(subset, "time_error", f"Error of the recovered time ({observed})",
                            f"Absolute error of the time Stage A recovered (one sampler call = 0.016). {quartiles}",
                            "time error", output / f"time_error_tokens-{tokens}_rows-{rows}.png", scale="log",
                            knowledge=[k for k in KNOWLEDGE if not k[0]])
            plot_metric(subset, "runtime", f"Runtime per run ({observed})",
                        f"Seconds for one (prompt, repetition): observation, Stage A and Stage B. {interval}",
                        "seconds", output / f"runtime_tokens-{tokens}_rows-{rows}.png")
        # Across prompt lengths: one line per time of the observed call (known time).
        if len(lengths) > 1 and any(c["knows_time"] for c in same_rows):
            which = "all rows observed" if rows == "all" else "only generated rows observed"
            plot_by_length(same_rows, "exact_recovery", f"Exact prompt recovery by prompt length ({which})",
                           f"Share of runs whose recovered tokens equal the prompt; the attacker knows the time. {interval}",
                           "share of runs", output / f"by_prompt_length_exact_recovery_rows-{rows}.png")
            plot_by_length(same_rows, "token_accuracy", f"Token accuracy by prompt length ({which})",
                           f"Share of prompt tokens recovered at the right position; the attacker knows the time. {interval}",
                           "share of tokens", output / f"by_prompt_length_token_accuracy_rows-{rows}.png")


if __name__ == "__main__":
    main()
