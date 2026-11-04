from typing import Protocol, runtime_checkable
import jax


@runtime_checkable
class ForwardMap(Protocol):
    observation_shape: tuple # (batch, rows, emb_dim)
    rank_diagnostics: dict # information about ranks of codes bot prompt and noise
    def algebraic(self, observation) -> jax.Array | None: ...
    def codes(self, coordinates) -> tuple[jax.Array, jax.Array]: ... # transform the coordinates to the original codes.
                                                                    # codes = (prompt_codes, noise_codes). prompts_codes are needed for stage B and
                                                                    # noise_codes are used to score the result of the stage A
    def __call__(self, coordinates) -> jax.Array: ... # from coordinates to hidden states
    def initialize(self, seed, prompt_std=1.0, noise_std=1.0) -> jax.Array: ... # initialize with gaussina noise the coordinates
    # The hidden_state is the full lang rows observation and the return value is compatible with observation_shape
    def select_rows(self, hidden_state: jax.Array) -> jax.Array: ...
    def time(self, coordinates: jax.Array) -> jax.Array: ... # the sampler time the coordinates assume: known from metadata, or recovered by Stage A
