# Stage A: backbone inversion

Given one ELF hidden state, find the bottleneck codes that produced it. The prompt's code
`zeta_p` is the target Stage B turns into tokens. A small residual means the codes
reproduce the observation; it does not prove they are the only ones that do.

```text
backbone/
  knowledge.py   AttackerKnowledge: what the attacker is assumed to know
  solver.py      Levenberg–Marquardt with matrix-free conjugate gradient (model-independent)
  recovery.py    recover(): random restarts, best result, Stage B target
models/elf/forward.py   BackboneForward: the ELF forward map that Stage A inverts
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
| `knows_time` | `true` | the time becomes one more unknown, `t = sigmoid(s)` (needs `time > 0`) |
| `knows_noise_scale` | `true` | random starts do not use ELF's noise scale |

## Solver

Levenberg–Marquardt minimizes `0.5 ||F(u) − H||²` over the code coordinates `u`, solving
each step with conjugate gradient on JVPs/VJPs (no Jacobian is built). A step is kept only
if it lowers the loss. Each restart starts from random codes (seed `seed + restart`), and
the lowest residual wins. At layer 0, with all rows observed and the time known, a direct
least-squares solve is used when it is unique.

Settings: `SolverConfig` in `solver.py`; `configs/stage_a/` lists only changed values.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests/backbone -t .
.venv/bin/python -m unittest discover -s tests/models/elf -t .
```
