"""What Stage B needs from a model: token IDs -> a differentiable representation of the prompt."""
from typing import Any, Protocol, runtime_checkable

import jax

# The representation every model has: the prompt codes Stage A recovers. A model may offer
# others under its own names (TokenModel.representations), e.g. ELF's "encoder_state".
PROMPT_CODE = "prompt_code"


@runtime_checkable
class PromptLayout(Protocol):
    prompt_length: int
    input_ids: jax.Array
    editable_mask: jax.Array
    # The same layout with the prompt tokens replaced by ``ids`` ([batch, prompt_length]).
    def with_prompt_ids(self, ids: jax.Array) -> "PromptLayout": ...


@runtime_checkable
class TokenModel(Protocol):
    prompt_length: int
    # Stage B needs the special IDs and the number of tokens the model knows.
    tokenizer: Any
    embedding_table: jax.Array
    # Names of the prompt representations a target can be given in; always includes PROMPT_CODE.
    representations: tuple[str, ...]
    # Features per prompt token of one representation.
    def target_dimension(self, representation: str) -> int: ...
    # Token IDs -> representation, [batch, prompt_length, target_dimension].
    def encode_token_ids(self, batch: PromptLayout, representation: str) -> jax.Array: ...
    # The rows of the embedding table for the batch's tokens.
    def input_embeddings(self, batch: PromptLayout) -> jax.Array: ...
    # Input embeddings -> representation; differentiable in the embeddings.
    def encode_embeddings(self, embeddings: jax.Array, batch: PromptLayout, representation: str) -> jax.Array: ...
    def prepare_token_ids(self, prompt_ids) -> PromptLayout: ...
