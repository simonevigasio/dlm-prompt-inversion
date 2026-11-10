# elf-prompt-inversion

Recover the prompt given to a continuous diffusion / flow-matching language model from one
of its hidden states. The attack does not depend on the model; two victim models are
implemented: ELF (Embedded Language Flows) and [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq).

1. **[Stage A](src/backbone/README.md)**: hidden state → codes of the prompt
   (ELF: bottleneck codes; DiffuSeq: token embeddings).
2. **[Stage B](src/tokens/README.md)**: codes → prompt tokens.

The attacker is assumed to know the model weights, the prompt length, the observed layer and
whether the observed sampler call is the first one. The time of the call, the noise scale and
the prompt rows of the hidden state can each be hidden ([attacker knowledge](#attacker-knowledge)).

## Setup

Python 3.10–3.12. ELF and DiffuSeq are git submodules, pinned to the commits in
`configs/model.yaml` and `configs/diffuseq.yaml`:

```bash
git clone --recursive <this repo>      # or, in an existing clone: git submodule update --init
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

| extra | adds | needed for |
|---|---|---|
| `test` | PyTorch (CPU), `datasets` | the tests; it also covers DiffuSeq |
| `diffuseq` | PyTorch | DiffuSeq without the tests (upstream is PyTorch) |
| `cuda` | JAX's CUDA 12 plugin | an NVIDIA GPU (Linux) |

For a GPU, install the CPU build of `torch` first, because its CUDA build conflicts with
JAX's cuDNN:

```bash
.venv/bin/python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[cuda,test]'
```

Checkpoints are downloaded on first use. ELF's come from Hugging Face; its denoiser is large
and its cache defaults to `/tmp/stage-b-hf`: set `denoiser.cache_dir` in `configs/model.yaml`
to a persistent path. DiffuSeq's QQP checkpoint is a zip on Google Drive (370 MB), verified
by its SHA-256 and cached in `~/.cache/elf-prompt-inversion/diffuseq`.

## Run an experiment

```bash
python3 run_pipeline.py --config configs/pipeline/example.yaml \
    --dataset prompts/example.jsonl --output outputs/example
```

For every prompt in the dataset (JSONL lines with a string `id` and `text`), this runs the
victim model's own sampler, captures one hidden state, inverts it, and scores the result
against the prompt. The hidden state holds the prompt rows and the generated (noise) rows
only; a model's prefix or format rows (ELF's time/scale/mode tokens, DiffuSeq's
`[CLS]`/`[SEP]`) take part in the computation but are never observed. Rerun the same command to resume an interrupted run.
`--device cpu|gpu|auto` selects the device.

[`configs/pipeline/example.yaml`](configs/pipeline/example.yaml) lists **every option** with
its meaning and default: copy it and change what your experiment needs. In short:

```yaml
model: ../model.yaml          # the victim: ../model.yaml (ELF) or ../diffuseq.yaml (DiffuSeq)
stage_a: ../stage_a/levenberg_marquardt.yaml
stage_b: ../stage_b/k100.yaml
synthetic_observation:        # how each prompt's hidden state is generated
  observation_layer: 3        # state after 3 backbone blocks
  noise_seed: 100
  # step: 32                  # observe the 32nd sampler call instead of the first
attacker_knowledge:           # optional; the defaults mean "knows everything"
  observed_rows: all
  knows_time: true
  knows_noise_scale: true
  noise_scale_guess: 1.0
# backbone:                   # optional; default: the released checkpoint
#   weights: random
#   bottleneck: false
#   init: normal
repetitions: 5                # runs per prompt, each with its own seeds
```

Each repetition derives its noise, Stage A and Stage B seeds from
`(seed, prompt id, repetition)`, so no two runs share noise or random starts.

### Attacker knowledge

| flag | default | effect when changed |
|---|---|---|
| `observed_rows` | `all` | `generated`: the prompt rows of the hidden state are hidden |
| `knows_time` | `true` | `false`: Stage A recovers the time of the observed call too (needs `step` > 0) |
| `knows_noise_scale` | `true` | `false`: Stage A starts the generated rows from noise of std `noise_scale_guess` instead of the model's own scale |
| `noise_scale_guess` | `1.0` | the starting std used when `knows_noise_scale` is `false` |

The models' true noise scales are 2.0 (ELF) and 1.0 (DiffuSeq). The default guess of 1.0 is
therefore wrong for ELF but right for DiffuSeq: to hide the scale from a DiffuSeq attacker,
set a guess different from 1.0.

### Results

The output folder holds:

| file | content |
|---|---|
| `results.jsonl` | one row per (prompt, repetition): recovered text and tokens, scores, Stage A diagnostics, seeds |
| `summary.json` | counts and averages over the whole run |
| `manifest.json` | every effective setting, hashes of the dataset and the code, and where the model weights came from (`model_provenance`) |
| `progress.jsonl` | live log; follow it with `tail -f` |

Scores in each row:

- `exact_recovery`, `token_accuracy`: the recovered tokens against the prompt's.
- `prompt_code_error`, `noise_code_error`: L2 distance between Stage A's codes and the true
  ones. The `*_relative_error` versions divide by the norm of the true code; use them to
  compare prompts of different lengths or models with different code widths.
- `true_time`: the time of the observed sampler call, as a fraction of the sampling
  trajectory (0 is the first call).
- `recovered_time`, `time_error`: only present when `knows_time` is `false`. Both models
  measure time as a fraction of the trajectory, so `time_error` can be compared across
  them; one call is 0.016 for ELF (64 calls) and 0.0005 for DiffuSeq (2000 calls).

How to read `summary.json`:

- `exact_recovery_rate`, `mean_token_accuracy`, `stage_a_convergence_rate` are averages over
  **completed** runs. Runs counted in `skipped` are not in them.
- A prompt is skipped when it cannot be used (empty, special tokens, too long); the reason is
  in `skipped_reasons`.
- Before Stage A starts, the JAX forward map is checked against the model's own sampler on
  the true codes. If they disagree, the noise is redrawn, up to 3 times; after that the run is
  skipped. Every rejected seed is kept in the row (`rejected_noise_seeds`) and counted in
  `forward_mismatches`. This should stay at 0: a nonzero value means the reported rates are
  conditional on the seeds where the two implementations agree.

## Victim models

The `model:` file of a pipeline config names the family (`family: elf` or
`family: diffuseq`). Everything model-specific follows from it: the adapter, the forward map
Stage A inverts, the sampler that makes observations, what time value a sampler `step`
means, and the allowed `backbone:` and `synthetic_observation:` keys.

| | ELF (`configs/model.yaml`) | DiffuSeq (`configs/diffuseq.yaml`) |
|---|---|---|
| release | ELF-B, de-en | QQP paraphrase |
| framework | JAX (the attack runs on the native modules) | PyTorch; `models/diffuseq/forward.py` ports the forward pass to JAX |
| prompt | conditioning prefix, T5 tokens | source sentence `[CLS] prompt [SEP] [SEP]`, BERT tokens |
| codes Stage A recovers | 128-dim bottleneck codes of the T5 encoding | the 128-dim token embeddings |
| backbone | 12 blocks, 768 wide | 12 BERT layers, 768 wide |
| sampler calls (`step`) | 64; step `k` is time `k/64` | 2000; step `k` is diffusion timestep `1999 − k` |
| noise scale | 2.0 | 1.0 |
| layer 0 | linear in the codes: closed-form solve | nonlinear (tanh MLP + LayerNorm): iterative |

DiffuSeq notes:

- The source rows keep their clean embeddings at every sampler call (upstream's partial
  noising). `[CLS]`/`[SEP]` are known to the attacker and their rows are not observed.
- `step: k` is the `k`-th call of upstream's default 2000-step decoder (one step takes
  about 0.2 s on a CPU). Upstream's DDIM decoder (`--step` < 2000) is not supported: it
  ignores the source.
- With `knows_time: false` the time enters through a sinusoidal embedding, so the loss
  oscillates in the time and local search rarely finds it.

### Random weights and the bottleneck

`backbone:` replaces the released checkpoint with random weights, with or without the
bottleneck (the low-dimensional projection in front of the backbone). Two things to keep in
mind when reading such results:

- **Use the right control.** There is no checkpoint without the bottleneck, so
  `bottleneck: false` needs `weights: random`. Comparing it with the checkpoint changes two
  things at once (trained → random, bottleneck → none). The comparison that isolates the
  bottleneck is random weights **with** against random weights **without** it.
- **The random model is not a standard initialization.** Every parameter is drawn from
  `N(0, init_std²)`, including the normalization scales (RMSNorm/LayerNorm weights, normally
  1) and the biases (normally 0). On the full-size ELF the blocks still transform the hidden
  state about as much as with upstream's initialization (relative change after 3 blocks:
  0.43 against 0.42 without the bottleneck, 0.81 against 0.57 with it), so the model is not
  degenerate, but results on it describe this particular random model.

## Datasets

Prompt length is counted in tokens of the victim model, so a dataset is built for one family:

```bash
.venv/bin/python -m elf_inversion.datasets.dolly --family elf --tokens 20 --count 50
.venv/bin/python -m elf_inversion.datasets.dolly --family diffuseq --tokens 20 --count 50
```

This writes `data/dolly/dolly-<family>-20-tokens-50.jsonl`: Dolly instructions that need no
context and have exactly 20 tokens under that family's tokenizer.

## Use it from Python

```python
from elf_inversion import AttackConfig, ELFAdapter, invert_prompt

adapter = ELFAdapter.from_config()
result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3,
                       search=AttackConfig(k=100, restarts=3))
print(result.text, result.evaluation["stage_b_token_accuracy"])
```

`generate`, `save_observation`, `load_observation` and `invert_observation` keep a hidden
state on disk and invert it later (see `observation.py`). Importing the package sets JAX's
matmul precision to `highest`, so the Python API, the command and the tests compute alike.

## Layout

```text
run_pipeline.py          run an experiment: dataset → Stage A → Stage B → scores
configs/
  model.yaml             ELF: pinned commit, tokenizer and checkpoints (family: elf)
  diffuseq.yaml          DiffuSeq: pinned commit and checkpoint (family: diffuseq)
  pipeline/example.yaml  an experiment, with every option explained
  stage_a/, stage_b/     solver / search settings (only values that differ from the defaults)
src/                     the Python package, imported as elf_inversion
  models/                victim models; interface.py is what the attack needs from one
    elf/                 ELF adapter and forward map; upstream/ is the ELF git submodule
    diffuseq/            DiffuSeq adapter and forward map; upstream/ is the DiffuSeq git submodule
  backbone/              Stage A (model-independent)
  tokens/                Stage B (model-independent)
  pipeline.py            Stage A → Stage B for one prompt, and the scores
  observation.py         generate, save and load hidden states (any model)
  cli.py, runs.py        the experiment command and its resumable output folders
  datasets/              prompt datasets (Dolly)
tests/                   mirror src/: backbone/, tokens/, models/<family>/
```

To add a victim model, create `src/models/<name>/` with a `family.py` (see the docstring of
`src/models/__init__.py`), add the name to `FAMILIES`, and write its model settings file.
Stage A, Stage B, the pipeline and the command need no change.

## Tests

```bash
.venv/bin/python -m unittest
```

The tests check that the code runs and that the forward maps match the models' own code;
they are not the experiment. They use a tiny random ELF and a tiny random DiffuSeq and need
no downloads. Set `ELF_CHECKPOINTS=1` or `DIFFUSEQ_CHECKPOINTS=1` to also check the real
checkpoints.
