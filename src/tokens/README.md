# Stage B: token search

Find prompt tokens whose code matches a target: usually Stage A's prompt code. The search
only uses a `TokenModel` (`interface.py`); ELF and DiffuSeq adapters both are one.

```text
tokens/
  attack.py   one greedy search from given start tokens
  search.py   random restarts of the attack; keeps the lowest loss
  data.py     replacement vocabulary and prompt validation
```

```python
from elf_inversion.tokens.attack import AttackConfig
from elf_inversion.tokens.search import search

found = search(target, adapter, AttackConfig(k=100, restarts=5))
found.tokens, found.final_loss, found.selected_restart
```

## Target

The target is one **representation** of the prompt, named by `AttackConfig.representation`:

- **`prompt_code`** (the default, and what the pipeline uses): the codes Stage A recovers.
  Every model has it. ELF: the 128-dim bottleneck code `C (Wz + Ws) W1 + c`. DiffuSeq: the
  prompt's token embeddings (128 features), where each token's code depends on that token alone.
- A model may offer more under its own names (`TokenModel.representations`). ELF offers
  **`encoder_state`**: the T5 encoder output `C` itself, with 512 features per token.

For ELF the prompt is padded with one EOS and EOS padding; the encoder mask keeps the padding
from affecting the prompt's code.

## Search

The loss is the mean squared error between the tokens' code and the target. Each iteration:

1. takes the gradient of the loss with respect to the input embeddings;
2. uses it to shortlist the `k` most promising replacement tokens at every position;
3. computes the exact loss of every single-token swap in the shortlist;
4. keeps the best swap if it lowers the loss, otherwise stops.

The search also stops at `stopping_loss_tolerance`, `max_iterations` or
`max_candidate_evaluations` (keep it ≥ `prompt_length × k × max_iterations`). With
restarts, each starts from random tokens (seed `seed + restart`) and the lowest final loss
wins. By default every restart runs; with `stop_restarts_at_tolerance: true` (set in
`configs/stage_b/k100.yaml`) the first restart that reaches the tolerance ends the search.

Settings: `AttackConfig` in `attack.py`; `configs/stage_b/` lists only changed values.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests/tokens -t .
```

They use the tiny fake model in `tokens/testing.py` and need no victim model or checkpoint.
