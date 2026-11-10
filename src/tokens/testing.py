"""A tiny Stage B model for offline tests: pure JAX, no victim model, no downloads."""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np


class TinyTokenizer:
    all_special_ids = (0, 1)

    def __len__(self):
        return 6


@dataclass(frozen=True)
class TinyBatch:
    """A PromptLayout: the prompt, then three fixed suffix tokens."""

    input_ids: jax.Array      # [batch, prompt_length + 3]
    editable_mask: jax.Array  # prompt positions that hold a non-special token
    prompt_length: int

    def with_prompt_ids(self, ids) -> "TinyBatch":
        fields = {name: jnp.repeat(getattr(self, name), ids.shape[0], axis=0) for name in ("input_ids", "editable_mask")}
        fields["input_ids"] = fields["input_ids"].at[:, :self.prompt_length].set(ids)
        return replace(self, **fields)


class TinyModel:
    """A TokenModel with six tokens and one feature: the code of a token is its embedding squared.

    Nonlinear on purpose: the gradient's ranking can overshoot the target. It offers two
    representations: "raw" (the squared embedding) and "prompt_code" (2 * raw + 0.25), which
    differ in scale and offset, so mixing them up fails the search tests.
    """
    representations = ("prompt_code", "raw")

    def __init__(self, length=2, embeddings=None):
        self.prompt_length = length
        self.tokenizer = TinyTokenizer()
        self.embedding_table = jnp.asarray(embeddings if embeddings is not None else np.arange(6),
                                           dtype=jnp.float32).reshape(6, 1)

    def prepare_token_ids(self, tokens) -> TinyBatch:
        ids = np.asarray(tokens)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or ids.shape[1] != self.prompt_length or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Invalid prompt shape or dtype")
        if np.any(ids < 0) or np.any(ids >= 6):
            raise ValueError("Invalid token ID")
        full = np.ones((len(ids), self.prompt_length + 3), dtype=np.int32)
        full[:, :self.prompt_length] = ids
        in_prompt = np.broadcast_to(np.arange(full.shape[1])[None, :] < self.prompt_length, full.shape)
        editable = in_prompt & ~np.isin(full, self.tokenizer.all_special_ids)
        return TinyBatch(jnp.asarray(full), jnp.asarray(editable), self.prompt_length)

    def input_embeddings(self, batch):
        return self.embedding_table[batch.input_ids]

    def encode_embeddings(self, embeddings, batch, representation="raw"):
        x = jnp.where(batch.editable_mask[..., None], embeddings, self.input_embeddings(batch))
        output = x[:, :batch.prompt_length] ** 2
        if representation == "prompt_code":
            output = 2 * output + 0.25
        return output

    def encode_token_ids(self, batch, representation="raw"):
        return self.encode_embeddings(self.input_embeddings(batch), batch, representation)

    def target(self, tokens, representation="raw"):
        return self.encode_token_ids(self.prepare_token_ids(tokens), representation)

    def target_dimension(self, representation) -> int:
        return 1
