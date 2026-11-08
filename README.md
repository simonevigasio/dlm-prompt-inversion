# elf-prompt-inversion

Recover the prompt given to ELF (Embedded Language Flows) from one of its hidden states.

1. **[Stage A](src/elf_inversion/backbone/README.md)**: hidden state → bottleneck codes of the prompt.
2. **[Stage B](src/elf_inversion/tokens/README.md)**: bottleneck codes → prompt tokens.

```text
run_pipeline.py          run an experiment: dataset → Stage A → Stage B → scores
configs/
  model.yaml             pinned ELF commit, tokenizer and checkpoints
  pipeline/              experiments: model + stage files + observation + attacker knowledge
  stage_a/, stage_b/     solver / search settings (only values that differ from the defaults)
src/elf_inversion/
  models/elf/            ELF adapter and forward map; upstream/ is the ELF git submodule
  backbone/              Stage A
  tokens/                Stage B
  pipeline.py            Stage A → Stage B for one prompt
  observation.py         generate, save and load hidden states
  cli.py, runs.py        the experiment command and its resumable output folders
  datasets/              prompt datasets (Dolly)
tests/
```

## Setup

Python 3.10–3.12. ELF is a git submodule, pinned to the commit in `configs/model.yaml`:

```bash
git clone --recursive <this repo>      # or, in an existing clone: git submodule update --init
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'           # CPU
```

For an NVIDIA GPU (Linux, CUDA 12), install the CPU build of `torch` first, because the
CUDA build conflicts with JAX's cuDNN:

```bash
.venv/bin/python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[cuda,test]'
```

Checkpoints are downloaded from Hugging Face on first use. The denoiser is large and its
cache defaults to `/tmp/stage-b-hf`: set `denoiser.cache_dir` in `configs/model.yaml` to a
persistent path.

## Run an experiment

```bash
python3 run_pipeline.py --config configs/pipeline/layer_3.yaml \
    --dataset prompts/example.jsonl --output outputs/example
```

For every prompt in the dataset (JSONL lines with a string `id` and `text`), this generates
ELF's hidden state, inverts it, and scores the result. The output folder holds
`results.jsonl` (one row per prompt and repetition), `summary.json`, `manifest.json`
(every effective setting) and `progress.jsonl` (live log; `tail -f` it). Rerun the same
command to resume an interrupted run. `--device cpu|gpu|auto` selects the device.

A pipeline config looks like this:

```yaml
model: ../model.yaml
stage_a: ../stage_a/levenberg_marquardt.yaml
stage_b: ../stage_b/k100.yaml
synthetic_observation:        # how each prompt's hidden state is generated
  observation_layer: 3        # state after 3 backbone blocks
  noise_seed: 100
  # time: 0.5                 # observe a later sampler call (a multiple of 1/64)
attacker_knowledge:           # optional, every flag defaults to "knows everything"
  observed_rows: all          # or "generated": the prompt rows are hidden
  knows_time: true            # false: Stage A also recovers the time (needs time > 0)
  knows_noise_scale: true
repetitions: 5                # runs per prompt, each with its own seeds
```

Each repetition derives its seeds from `(seed, prompt id, repetition)`, so no two runs
share noise or random starts.

To make a dataset of Dolly instructions with exactly 20 tokens:

```bash
.venv/bin/python -m elf_inversion.datasets.dolly --tokens 20 --count 50
```

## Use it from Python

```python
from elf_inversion import AttackConfig, ELFAdapter, invert_prompt

adapter = ELFAdapter.from_config(projected=True)
result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3,
                       search=AttackConfig(k=100, restarts=3))
print(result.text, result.evaluation["stage_b_token_accuracy"])
```

`generate`, `save_observation`, `load_observation` and `invert_observation` keep a hidden
state on disk and invert it later (see `observation.py`).

## Tests

```bash
.venv/bin/python -m unittest
```

They use a tiny random ELF and need no downloads. Set `ELF_CHECKPOINTS=1` to also check the
real checkpoints.
