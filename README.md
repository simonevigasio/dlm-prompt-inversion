# elf-prompt-inversion

Recover the prompt given to a continuous diffusion language model from one of its hidden
states. The attack is model-independent; two victim models are implemented:
ELF (Embedded Language Flows) and [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq).

1. **[Stage A](src/backbone/README.md)**: hidden state → codes of the prompt
   (ELF: bottleneck codes; DiffuSeq: token embeddings).
2. **[Stage B](src/tokens/README.md)**: codes → prompt tokens.

```text
run_pipeline.py          run an experiment: dataset → Stage A → Stage B → scores
configs/
  model.yaml             ELF: pinned commit, tokenizer and checkpoints (family: elf)
  diffuseq.yaml          DiffuSeq: pinned commit and checkpoint (family: diffuseq)
  pipeline/              experiments: model + stage files + observation + attacker knowledge
  stage_a/, stage_b/     solver / search settings (only values that differ from the defaults)
src/                     the Python package, imported as elf_inversion
  models/                victim models; interface.py is what the attack needs from one
    elf/                 ELF adapter and forward map; upstream/ is the ELF git submodule
    diffuseq/            DiffuSeq adapter and forward map; upstream/ is the DiffuSeq git submodule
  backbone/              Stage A
  tokens/                Stage B
  pipeline.py            Stage A → Stage B for one prompt
  observation.py         generate, save and load hidden states (any model)
  cli.py, runs.py        the experiment command and its resumable output folders
  datasets/              prompt datasets (Dolly)
tests/
```

## Setup

Python 3.10–3.12. ELF and DiffuSeq are git submodules, pinned to the commits in
`configs/model.yaml` and `configs/diffuseq.yaml`:

```bash
git clone --recursive <this repo>      # or, in an existing clone: git submodule update --init
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'           # CPU (the test extra includes PyTorch for DiffuSeq)
```

For an NVIDIA GPU (Linux, CUDA 12), install the CPU build of `torch` first, because the
CUDA build conflicts with JAX's cuDNN:

```bash
.venv/bin/python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[cuda,test]'
```

Checkpoints are downloaded on first use. ELF's come from Hugging Face; its denoiser is large
and its cache defaults to `/tmp/stage-b-hf`: set `denoiser.cache_dir` in `configs/model.yaml` to
a persistent path. DiffuSeq's QQP checkpoint is a zip on Google Drive (370 MB), verified by its
SHA-256 and cached in `~/.cache/elf-prompt-inversion/diffuseq`.

## Victim models

The pipeline config's `model:` file names the family (`family: elf` or `family: diffuseq`),
and everything else follows from it: the adapter, the forward map Stage A inverts, the
sampler that makes observations, and the allowed `backbone:` and `synthetic_observation:` keys.

**DiffuSeq** (`configs/diffuseq.yaml`, QQP paraphrase release: 128-wide token embeddings,
a 12-layer BERT-base encoder, 128 rows, 2000 diffusion steps) is a sequence-to-sequence model
and PyTorch upstream. Its own model and sampler make each observation;
`models/diffuseq/forward.py` ports the forward pass to JAX for the attack, and every
observation checks the two agree. Differences from ELF:

- The prompt is DiffuSeq's source sentence, laid out as upstream does:
  `[CLS] prompt [SEP] [SEP]`, then the generated rows. The source rows keep their clean
  embeddings at every call (upstream's partial noising); `[CLS]`/`[SEP]` are known.
- No encoder and no self-conditioning. The codes are the model's input rows: the prompt's
  embeddings and the generated rows (noise at the first call, `x_t` later), each 128 wide
  against a 768-wide encoder. `input_up_proj` (a tanh MLP) and a LayerNorm make even
  layer 0 nonlinear, so there is no algebraic solve.
- `observed_rows: generated` hides the whole source block.
- Time `k / 2000` is the `k`-th call of upstream's default 2000-step decoder (one step is
  about 0.2 s on a CPU). Upstream's DDIM decoder (`--step` < 2000) is not supported: it
  ignores the source. With `knows_time: false` the timestep enters through a sinusoidal
  embedding, so the loss oscillates in the time and local search rarely finds it.
- `backbone:` has `weights`, `bottleneck`, `init`, `init_std`, `init_seed`. Without the
  bottleneck the embeddings are 768 wide and enter the encoder directly (random weights only).

```bash
python3 run_pipeline.py --config configs/pipeline/diffuseq_layer_3.yaml \
    --dataset prompts/example.jsonl --output outputs/diffuseq
```

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

They use a tiny random ELF and a tiny random DiffuSeq and need no downloads. Set
`ELF_CHECKPOINTS=1` or `DIFFUSEQ_CHECKPOINTS=1` to also check the real checkpoints.
