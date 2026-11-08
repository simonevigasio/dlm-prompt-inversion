# Stage B: token search

Find prompt tokens whose ELF code matches a target: usually Stage A's `zeta_p`.

```text
tokens/
  attack.py   one greedy search from given start tokens
  search.py   random restarts of the attack; keeps the lowest loss
  data.py     replacement vocabulary and prompt validation
```

```python
from elf_inversion.tokens.attack import AttackConfig
from elf_inversion.tokens.search import search

found = search(target, adapter, AttackConfig(mode="projected", k=100, restarts=5))
found.tokens, found.final_loss, found.selected_restart
```

## Target

- **projected** (used by the pipeline): the 128-dim bottleneck code,
  `C (Wz + Ws) W1 + c`, the same quantity Stage A recovers.
- **encoder**: the T5 encoder output `C` itself, with 512 features per token.

The prompt is padded with one EOS and EOS padding; the encoder mask keeps the padding
from affecting the prompt's code.

## Search

The loss is the mean squared error between the tokens' code and the target. Each iteration:

1. takes the gradient of the loss with respect to the input embeddings;
2. uses it to shortlist the `k` most promising replacement tokens at every position;
3. computes the exact loss of every single-token swap in the shortlist;
4. keeps the best swap if it lowers the loss, otherwise stops.

The search also stops at `stopping_loss_tolerance`, `max_iterations` or
`max_candidate_evaluations` (keep it ≥ `prompt_length × k × max_iterations`). With
restarts, each starts from random tokens (seed `seed + restart`); by default the first
restart that reaches the tolerance ends the search.

Settings: `AttackConfig` in `attack.py`; `configs/stage_b/` lists only changed values.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests/tokens -t .
```

They use a tiny fake model and need neither ELF nor checkpoints.
