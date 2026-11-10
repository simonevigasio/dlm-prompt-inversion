"""What Stage A needs from a model: a differentiable map from coordinates to the observed rows."""
from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class ForwardMap(Protocol):
    observation_shape: tuple  # (1, observed rows, hidden size)
    rank_diagnostics: dict    # ranks of the prompt and noise code spaces, saved with the result

    # Coordinates -> observed hidden-state rows. This is what Stage A inverts.
    def __call__(self, coordinates) -> jax.Array: ...
    # Coordinates -> (prompt codes, noise codes). The prompt codes are Stage B's target;
    # the noise codes are only used to score Stage A.
    def codes(self, coordinates) -> tuple[jax.Array, jax.Array]: ...
    # Inverse of codes() (time known): observation.generate checks the map on the true codes with it.
    def coordinates(self, prompt_code: jax.Array, noise_code: jax.Array) -> jax.Array: ...
    # Random start of one restart. The noise scale comes from the map's AttackerKnowledge.
    def initialize(self, seed, prompt_std=1.0) -> jax.Array: ...
    # Exact solution when the map has one (e.g. ELF at layer 0), else None.
    def closed_form(self, observation) -> jax.Array | None: ...
    # Cut a full text-row hidden state down to the rows the attacker observes (observation_shape).
    def select_rows(self, hidden_state: jax.Array) -> jax.Array: ...
    # Sampler time the coordinates assume: declared in the metadata, or recovered by Stage A.
    def time(self, coordinates: jax.Array) -> jax.Array: ...
