# Stage A: backbone inversion

Given one hidden state, find the codes that produced it. The solver and `recover()` are
model-independent: they only use a `ForwardMap` (`interface.py`), which each model builds
with `adapter.forward_map(metadata, knowledge)`. The sections below describe ELF; DiffuSeq's
map is documented in `models/diffuseq/forward.py`. The prompt's code
`zeta_p` is the target Stage B turns into tokens. A small residual means the codes
reproduce the observation; it does not prove they are the only ones that do.

```text
backbone/
  knowledge.py   AttackerKnowledge: what the attacker is assumed to know
  interface.py   ForwardMap: what Stage A needs from a model's forward map
  basis.py       row_basis: coordinates of the reachable codes
  solver.py      Levenberg–Marquardt with matrix-free conjugate gradient (model-independent)
  recovery.py    recover(): random restarts, best result, Stage B target
models/elf/forward.py        BackboneForward: the ELF forward map that Stage A inverts
models/diffuseq/forward.py   BackboneForward: the DiffuSeq forward map (JAX port of upstream)
```

```python
from elf_inversion.backbone.knowledge import AttackerKnowledge
from elf_inversion.backbone.recovery import recover
from elf_inversion.backbone.solver import SolverConfig
from elf_inversion.models.elf.forward import BackboneForward, ForwardMetadata

forward = BackboneForward(adapter, ForwardMetadata(prompt_length=8, observation_layer=3),
                          AttackerKnowledge())
result = recover(hidden_state, forward, SolverConfig(restarts=3))
result.stage_b_target   # [1, prompt_length, bottleneck_dim]
```

## What is inverted

With `C` the normalized T5 encoding of the prompt and `g` the sampler's noise, ELF
projects every text row to a 128-dim bottleneck code and then to the backbone width:

```text
zeta_p = C (Wz + Ws) W1 + c      prompt rows
zeta_g = g Wz W1 + c             generated rows (first sampling pass)
X = [zeta_p; zeta_g] W2 + b2  →  time/scale/mode prefix rows + k backbone blocks  →  H
```

`[Wz; Ws], c` come from `self_cond_proj`, `W1` from `text_proj.proj1`, `W2, b2` from
`text_proj.proj2`. At a later sampler call (`time > 0`), the generated rows carry the
unknown `[z_t, x_prev]` instead of `[g, 0]`.

Stage A searches the codes directly, in an orthonormal basis of the reachable row space of
each projection, so every candidate is a code ELF could actually produce. The observation
`H` contains only the text rows: the prefix rows are computed but not observed.

## Attacker knowledge

Always known: the weights, the prompt length, the observation layer, the forward mode.

| flag | default | when off |
|---|---|---|
| `observed_rows` | `"all"` | `"generated"`: only the generated rows are observed |
| `knows_time` | `true` | the time becomes one more unknown, `t = sigmoid(s)` (needs `step > 0`) |
| `knows_noise_scale` | `true` | random starts draw the generated rows with std `noise_scale_guess` (default 1.0) instead of the model's noise scale (ELF 2.0, DiffuSeq 1.0) |

The noise scale only sets where the solver starts; it is not a constraint on the solution.

## Solver

Levenberg–Marquardt minimizes `0.5 ||F(u) − H||²` over the code coordinates `u`, solving
each step with conjugate gradient on JVPs/VJPs (no Jacobian is built). A step is kept only
if it lowers the loss. Each restart starts from random codes (seed `seed + restart`, prompt
codes with std `prompt_start_std`), and the lowest residual wins. When the forward map has
an exact solution (`closed_form`: ELF at layer 0, with all rows observed and the time known,
when the least-squares solve is unique), it is used instead; `use_closed_form: false`
always iterates.

Settings: `SolverConfig` in `solver.py`; `configs/stage_a/` lists only changed values.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests/backbone -t .
.venv/bin/python -m unittest discover -s tests/models/elf -t .
.venv/bin/python -m unittest discover -s tests/models/diffuseq -t .
```
