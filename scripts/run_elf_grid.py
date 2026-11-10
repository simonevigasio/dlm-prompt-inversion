#!/usr/bin/env python3
"""Run the ELF experiment grid, one configuration after the other, skipping the finished ones.

    python3 scripts/run_elf_grid.py                 # run everything that is not done yet
    python3 scripts/run_elf_grid.py --list          # only show which configurations are done
    python3 scripts/run_elf_grid.py --data /path/to/data/dolly --device gpu
    python3 scripts/run_elf_grid.py --shard 3/16    # one of 16 parallel workers, e.g. one per GPU

The grid (90 configurations for each prompt length):

    prompt length    10, 20, 30 tokens                 one Dolly dataset per length, see below
    layer            1, 6 (middle), 12 (last)          ELF-B has 12 blocks
    time of the call 0, 0.5, 0.95                      sampler calls 0, 32, 61 of 64
    time             known, unknown                    attacker_knowledge.knows_time
                                                       (at time 0 only known: the first call is
                                                       always at time 0, the pipeline rejects it)
    observed rows    all, generated                    attacker_knowledge.observed_rows
                                                       (generated: the prompt rows are hidden)
    victim weights   checkpoint                        released weights, with bottleneck
                     random                            random weights, with bottleneck
                     random-no-bottleneck              random weights, no bottleneck

Each configuration inverts PROMPTS prompts REPETITIONS times. The prompts of a length are a
fixed random subset (seed PROMPT_SEED) of that length's dataset in the --data folder,
dolly-elf-<length>-tokens-50.jsonl (or dolly-<length>-tokens-50.jsonl), made with

    .venv/bin/python -m elf_inversion.datasets.dolly --family elf --tokens <length> --count 50

The subset is saved as <output>/prompts-<length>-tokens-<PROMPTS>.jsonl and shared by every
configuration of that length. A pipeline config does not depend on the prompt length: it is
written to configs/pipeline/elf_grid/<name>.yaml, and its results for one length go to
<output>/tokens-<length>_<name>/.

A configuration is done when its summary.json reports every (prompt, repetition) as finished.
Anything else (never started, interrupted, crashed) is run with run_pipeline.py, which resumes
from the rows already in results.jsonl. A configuration that fails does not stop the grid: the
script goes on and lists the failures at the end. Plots: scripts/plot_elf_grid.py.

Several GPUs: start N copies with --shard 0/N ... --shard N-1/N. Worker i takes every N-th
experiment starting at the i-th, so the workers never share an experiment and each gets a mix
of cheap and expensive ones. All of them write below the same --output folder.
"""
import argparse
import itertools
import json
import random
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "pipeline" / "elf_grid"

# Prompt lengths in tokens of ELF's tokenizer (T5); ELF's conditioning prefix holds up to 64.
PROMPT_LENGTHS = (10, 20, 30)
LAYERS = (1, 6, 12)
KNOWS_TIME = (True, False)
# The noise scale is not an axis of the grid: the attacker always knows it.
OBSERVED_ROWS = ("all", "generated")
# name -> backbone section. No checkpoint was trained without the bottleneck, so that case is random.
BACKBONES = {
    "checkpoint": {"weights": "checkpoint", "bottleneck": True, "init": "upstream"},
    "random": {"weights": "random", "bottleneck": True, "init": "normal"},
    "random-no-bottleneck": {"weights": "random", "bottleneck": False, "init": "normal"},
}
REPETITIONS = 1
# Prompts per configuration, drawn once from the dataset with PROMPT_SEED (None: the whole dataset).
PROMPTS = 25
PROMPT_SEED = 0
# Observed sampler calls; ELF's time of call k is k / 64. 0.95 is not a call of the 64-step
# sampler: the nearest one is 61 (time 0.953).
STEPS = (0, 32, 61)
INIT_STD, INIT_SEED = 0.02, 0
STAGE_B = "k100.yaml"

TEMPLATE = """\
# Written by scripts/run_elf_grid.py; edit the grid there, not this file.
model: ../../models/elf.yaml
stage_a: ../../stage_a/levenberg_marquardt.yaml
stage_b: ../../stage_b/{stage_b}
synthetic_observation:
  observation_layer: {layer}
  noise_seed: 100
  step: {step}
attacker_knowledge:
  observed_rows: {observed_rows}
  knows_time: {knows_time}
  knows_noise_scale: true
backbone:
  weights: {weights}
  bottleneck: {bottleneck}
  init: {init}
  init_std: {init_std}
  init_seed: {init_seed}
repetitions: {repetitions}
"""


def yaml_bool(value):
    return "true" if value else "false"


def grid():
    """(name, config text) of every configuration, in run order."""
    for (backbone, setup), layer, step, rows, knows_time in itertools.product(
            BACKBONES.items(), LAYERS, STEPS, OBSERVED_ROWS, KNOWS_TIME):
        if step == 0 and not knows_time:
            continue  # the first call is always at time 0: there is no time to recover
        name = (f"layer-{layer:02d}_step-{step:02d}_rows-{rows}_time-{'known' if knows_time else 'unknown'}"
                f"_{backbone}")
        yield name, TEMPLATE.format(
            stage_b=STAGE_B, layer=layer, step=step, observed_rows=rows, knows_time=yaml_bool(knows_time),
            weights=setup["weights"], bottleneck=yaml_bool(setup["bottleneck"]), init=setup["init"],
            init_std=INIT_STD, init_seed=INIT_SEED, repetitions=REPETITIONS)


def find_dataset(folder, length):
    """The Dolly dataset of ``length``-token prompts in ``folder``, or None when there is none.

    The dataset's manifest (written next to it by elf_inversion.datasets.dolly) must agree on
    the length: a file name alone does not prove what is inside.
    """
    for name in (f"dolly-elf-{length}-tokens-50.jsonl", f"dolly-{length}-tokens-50.jsonl"):
        path = folder / name
        if not path.is_file():
            continue
        manifest = path.with_suffix(".manifest.json")
        if not manifest.is_file():
            sys.exit(f"{path}: no {manifest.name} next to it, so its prompt length cannot be checked")
        recorded = json.loads(manifest.read_text())
        if recorded.get("exact_token_count") != length or recorded.get("family", "elf") != "elf":
            sys.exit(f"{path}: its manifest says {recorded.get('exact_token_count')}-token prompts for "
                     f"{recorded.get('family', 'elf')}, not {length}-token prompts for elf; regenerate it")
        return path
    return None


def select_prompts(dataset, output_root, length):
    """(dataset file to run, number of prompts): the whole dataset, or its fixed subset of PROMPTS.

    The subset keeps the dataset's order and is the same on every call, so its file (and with it
    the resume identity of every configuration) never changes.
    """
    lines = [line for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if PROMPTS is None or PROMPTS == len(lines):
        return dataset, len(lines)
    if PROMPTS > len(lines):
        sys.exit(f"{dataset} has {len(lines)} prompts, fewer than PROMPTS = {PROMPTS}")
    chosen = sorted(random.Random(PROMPT_SEED).sample(range(len(lines)), PROMPTS))
    text = "".join(lines[index] + "\n" for index in chosen)
    subset = output_root / f"prompts-{length}-tokens-{PROMPTS}.jsonl"
    if not subset.exists() or subset.read_text(encoding="utf-8") != text:
        output_root.mkdir(parents=True, exist_ok=True)
        subset.write_text(text, encoding="utf-8")
    return subset, PROMPTS


def progress(output, prompts):
    """(finished runs, total runs) of one configuration, from its summary.json."""
    total = prompts * REPETITIONS
    try:
        summary = json.loads((output / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, total
    if summary.get("prompts") != prompts or summary.get("repetitions") != REPETITIONS:
        sys.exit(f"{output}: holds a run of {summary.get('prompts')} prompts x {summary.get('repetitions')} "
                 f"repetitions, not {prompts} x {REPETITIONS}; use another --output")
    return total - summary["pending"], total


def main():
    parser = argparse.ArgumentParser(description="Run the ELF experiment grid; finished configurations are skipped.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dolly",
                        help="Folder with one Dolly dataset per prompt length (default: data/dolly)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "elf_grid",
                        help="Folder with one results folder per configuration (default: outputs/elf_grid)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--shard", default="0/1", metavar="I/N",
                        help="Run only experiments I, I+N, I+2N, ... (I from 0 to N-1); default 0/1 = all")
    parser.add_argument("--list", action="store_true", help="Show the state of every configuration and exit")
    args = parser.parse_args()

    try:
        shard, shards = (int(part) for part in args.shard.split("/"))
    except ValueError:
        parser.error("--shard must be I/N, e.g. 3/16")
    if not 0 <= shard < shards:
        parser.error("--shard I/N needs 0 <= I < N")
    data, output_root = args.data.resolve(), args.output.resolve()
    sources = {length: find_dataset(data, length) for length in PROMPT_LENGTHS}
    missing = [length for length, source in sources.items() if source is None]
    if missing:
        sys.exit(f"No dataset in {data} for prompt lengths {missing}. Make each one with\n" + "\n".join(
            f"  .venv/bin/python -m elf_inversion.datasets.dolly --family elf --tokens {length} --count 50"
            for length in missing) + f"\n(written to data/dolly; pass another folder with --data)")
    datasets = {length: select_prompts(source, output_root, length) for length, source in sources.items()}
    # Every prompt length runs the same configurations, one length after the other.
    jobs = [(length, name, text) for length in PROMPT_LENGTHS for name, text in grid()]
    print(f"{len(jobs)} experiments: {len(jobs) // len(PROMPT_LENGTHS)} configurations x prompt lengths "
          f"{list(PROMPT_LENGTHS)}, {datasets[PROMPT_LENGTHS[0]][1]} prompts x {REPETITIONS} repetitions each\n"
          f"data: {data}\nresults: {output_root}", flush=True)

    if shards > 1:
        print(f"shard {shard}/{shards}: {len(jobs[shard::shards])} of these experiments", flush=True)
    failed = []
    for number, (length, name, text) in enumerate(jobs, 1):
        if (number - 1) % shards != shard:
            continue
        dataset, prompts = datasets[length]
        run = f"tokens-{length:02d}_{name}"
        label = f"[{number}/{len(jobs)}] {run}"
        output = output_root / run
        finished, total = progress(output, prompts)
        if finished == total:
            print(f"{label}: done", flush=True)
            continue
        if args.list:
            print(f"{label}: {finished}/{total} runs", flush=True)
            continue
        config = CONFIG_DIR / f"{name}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        if not config.exists() or config.read_text() != text:
            config.write_text(text)
        print(f"\n{label}: {finished}/{total} runs, {'resuming' if finished else 'starting'}", flush=True)
        # run_pipeline.py switches to the repository's .venv by itself and resumes in --output.
        code = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(config),
                               "--dataset", str(dataset), "--output", str(output),
                               "--device", args.device]).returncode
        if code or progress(output, prompts)[0] != total:
            failed.append(run)
            print(f"{label}: FAILED (exit code {code}); going on with the next one", flush=True)

    if failed:
        print(f"\n{len(failed)} experiments did not finish:\n  " + "\n  ".join(failed), flush=True)
        sys.exit(1)
    if not args.list:
        print("\nEvery experiment" + (f" of shard {shard}/{shards}" if shards > 1 else "") + " is done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted; rerun the same command to continue.")
