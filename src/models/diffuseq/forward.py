"""DiffuSeq's forward map of one sampler call, in JAX. No example secrets live here.

Upstream DiffuSeq (models/diffuseq/upstream, PyTorch) is ported here op by op:
``TransformerNetModel.forward`` up to the BERT encoder's k-th layer. ``params`` is the
upstream state dict, by its PyTorch names, as float32 JAX arrays; PyTorch Linear
weights are [out, in]. observation.generate checks this port against the PyTorch
model on every observation.

DiffuSeq is a sequence-to-sequence model; the prompt is its source sequence. Upstream
lays out one sequence of ``seq_len`` rows as

    [CLS] prompt [SEP] [SEP] | generated rows (target + padding)

The source block keeps its clean token embeddings at every sampler call (partial
noising, ``th.where(mask == 0, x_start, sample)``); the generated rows start from
N(0, I) noise. So, with E the token embedding table:

    row 0, rows n+1 and n+2   E[CLS], E[SEP], E[SEP]: known constants
    prompt rows 1..n          p = E[prompt]                  unknown (Stage A's prompt code)
    generated rows            g = noise, or x_t at a later call   unknown
    x  →  LayerNorm(pos + up(x) + time(tau))  →  k BERT layers  →  H

``up`` is upstream's tanh MLP (absent without the bottleneck). There is no
self-conditioning, so a later call has the same unknowns as the first one.

The observation H holds the prompt rows, then the generated rows, and nothing else. The
[CLS] and [SEP] rows take part in every layer (attention) but are not observed, like ELF's
prefix rows.
"""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from ...backbone.basis import row_basis
from ...backbone.knowledge import AttackerKnowledge

ENCODER = "input_transformers.layer"


@dataclass(frozen=True)
class ForwardMetadata:
    """Declared public inputs; layer k is after k complete BERT layers."""

    prompt_length: int   # source tokens between [CLS] and [SEP] [SEP]
    observation_layer: int = 1
    # Sampler time of the observed call: 0.0 for the first call (timestep T - 1), in [0, 1).
    time: float = 0.0
    # "first_sampling_pass": time 0, generated rows are N(0, I) noise. "sampling_pass": a later
    # call, generated rows are the unknown x_t. Source rows are the clean embeddings in both.
    forward_mode: str = "first_sampling_pass"


FORWARD_MODES = ("first_sampling_pass", "sampling_pass")


# --- Upstream TransformerNetModel and HF BertLayer, op by op --------------------------------

def _linear(params, name, x):
    return x @ params[name + ".weight"].T + params[name + ".bias"]


def _layer_norm(params, name, x, eps):
    mean = x.mean(-1, keepdims=True)
    var = jnp.square(x - mean).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * params[name + ".weight"] + params[name + ".bias"]


def timestep_embedding(timesteps, dim, max_period=10000):
    """diffuseq.utils.nn.timestep_embedding (dim even)."""
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = jnp.asarray(timesteps, jnp.float32)[:, None] * freqs[None]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def embed(params, x, timesteps, hidden_t_dim, eps):
    """TransformerNetModel.forward before the encoder: LayerNorm(pos + up(x) + time), eval mode."""
    t = _linear(params, "time_embed.2", jax.nn.silu(_linear(params, "time_embed.0",
                                                             timestep_embedding(timesteps, hidden_t_dim))))
    if "input_up_proj.0.weight" in params:
        x = _linear(params, "input_up_proj.2", jnp.tanh(_linear(params, "input_up_proj.0", x)))
    positions = params["position_embeddings.weight"][:x.shape[1]]
    return _layer_norm(params, "LayerNorm", positions[None] + x + t[:, None], eps)


def bert_layer(params, i, h, n_heads, eps):
    """transformers BertLayer (absolute positions, no attention mask, eval mode)."""
    p = f"{ENCODER}.{i}."
    batch, length, dim = h.shape

    def heads(name):
        return _linear(params, p + "attention.self." + name, h).reshape(batch, length, n_heads, dim // n_heads)

    q, k, v = heads("query"), heads("key"), heads("value")
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(dim // n_heads)
    context = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), v).reshape(batch, length, dim)
    a = _layer_norm(params, p + "attention.output.LayerNorm",
                    _linear(params, p + "attention.output.dense", context) + h, eps)
    m = jax.nn.gelu(_linear(params, p + "intermediate.dense", a), approximate=False)
    return _layer_norm(params, p + "output.LayerNorm", _linear(params, p + "output.dense", m) + a, eps)


def model_timestep(time, diffusion_steps):
    """Timestep value the model receives at sampler ``time``.

    The sampler's call k of S visits timestep index i = (T - 1) - k T / S (T = diffusion_steps),
    and upstream rescales it to i * 1000 / T (rescale_timesteps). Time 0 is index T - 1.
    """
    return ((diffusion_steps - 1) - jnp.asarray(time, jnp.float32) * diffusion_steps) * (1000.0 / diffusion_steps)


# --- Stage A forward map -------------------------------------------------------------------

def validate_metadata(adapter, metadata: ForwardMetadata):
    """Reject observations this forward map cannot represent."""
    if metadata.forward_mode not in FORWARD_MODES:
        raise ValueError(f"forward_mode must be one of {FORWARD_MODES}")
    if metadata.forward_mode == "first_sampling_pass" and metadata.time != 0.0:
        raise ValueError("first_sampling_pass requires time=0; use sampling_pass for later steps")
    if not (np.isfinite(metadata.time) and 0.0 <= metadata.time < 1.0):
        raise ValueError("time must lie in [0, 1)")
    if type(metadata.observation_layer) is not int or not 0 <= metadata.observation_layer <= adapter.n_layers:
        raise ValueError("observation_layer must be an integer from 0 through the number of encoder layers")
    if type(metadata.prompt_length) is not int or not 0 < metadata.prompt_length <= adapter.max_prompt_length:
        raise ValueError(f"prompt_length must be from 1 to {adapter.max_prompt_length}: the source block "
                         "[CLS] prompt [SEP] [SEP] must leave at least one generated row")
    if metadata.prompt_length != adapter.prompt_length:
        raise ValueError("Metadata and Stage B adapter prompt lengths must agree")


class BackboneForward:
    """Optimize the unknown input rows: the prompt embeddings and the generated rows.

    The source block's [CLS] and [SEP] rows are part of DiffuSeq's public format and are
    computed, not optimized. Only one example per inversion is supported.
    """

    def __init__(self, adapter, metadata: ForwardMetadata, knowledge: AttackerKnowledge = AttackerKnowledge()):
        validate_metadata(adapter, metadata)
        if metadata.forward_mode == "first_sampling_pass" and not knowledge.knows_time:
            raise ValueError("knows_time=False requires forward_mode='sampling_pass': the first pass is always at time 0")
        self.adapter, self.metadata, self.knowledge = adapter, metadata, knowledge
        self.length, self.n = adapter.seq_len, metadata.prompt_length
        self.source_length = self.n + 3              # [CLS] prompt [SEP] [SEP]
        self.generated = self.length - self.source_length
        d = adapter.embedding_dim
        # Every row is a free embedding-width vector: no projection between the codes and the input.
        self.prompt_map = self.noise_map = jnp.eye(d, dtype=jnp.float32)
        self.bias = jnp.zeros(d, dtype=jnp.float32)
        self.prompt_basis, ps, pc = row_basis(self.prompt_map)
        self.noise_basis, gs, gc = row_basis(self.noise_map)
        self.prompt_rank, self.noise_rank = len(self.prompt_basis), len(self.noise_basis)
        self.split = self.n * self.prompt_rank
        self.noise_end = self.split + self.generated * self.noise_rank
        self.size = self.noise_end + (0 if knowledge.knows_time else 1)
        self.rank_diagnostics = {
            "prompt_rank": self.prompt_rank, "noise_rank": self.noise_rank,
            "prompt_singular_values": ps.tolist(), "noise_singular_values": gs.tolist(),
            "prompt_rank_cutoff": float(pc), "noise_rank_cutoff": float(gc),
            "latent_parameterization": "token_embedding_rows",
        }
        # Row numbering of the observation: prompt rows first, then the generated rows.
        self.first_idx_observed_row = 0 if knowledge.observed_rows == "all" else self.n
        table = adapter.embedding_matrix
        self._cls = table[adapter.cls_token_id][None, None]
        self._sep = jnp.tile(table[adapter.sep_token_id][None, None], (1, 2, 1))

    def text_rows(self, full_state):
        """The prompt rows, then the generated rows, of a full [1, seq_len, hidden] state.

        Drops [CLS] and [SEP] [SEP]: they are DiffuSeq's fixed format, neither prompt nor noise.
        """
        return jnp.concatenate([full_state[:, 1:self.n + 1], full_state[:, self.source_length:]], axis=1)

    def select_rows(self, hidden_state):
        """hidden_state holds the prompt and generated rows; the result matches observation_shape."""
        return hidden_state[:, self.first_idx_observed_row:]

    @property
    def observation_shape(self):
        return (1, self.n + self.generated - self.first_idx_observed_row, self.adapter.hidden_size)

    def codes(self, coordinates):
        p = coordinates[:self.split].reshape(1, self.n, self.prompt_rank) @ self.prompt_basis + self.bias
        g = coordinates[self.split:self.noise_end].reshape(1, self.generated, self.noise_rank) @ self.noise_basis + self.bias
        return p, g

    def coordinates(self, prompt_code, noise_code):
        p = (prompt_code - self.bias) @ self.prompt_basis.T
        g = (noise_code - self.bias) @ self.noise_basis.T
        return jnp.concatenate([p.reshape(-1), g.reshape(-1)])

    def time(self, coordinates):
        """The declared time if known, otherwise sigmoid of the last coordinate."""
        if self.knowledge.knows_time:
            return self.metadata.time
        return jax.nn.sigmoid(coordinates[-1])

    def inputs(self, coordinates):
        """The model's full input x: [CLS] prompt [SEP] [SEP] generated, [1, seq_len, embedding_dim]."""
        p, g = self.codes(coordinates)
        return jnp.concatenate([self._cls, p, self._sep, g], axis=1)

    def __call__(self, coordinates):
        a = self.adapter
        timesteps = model_timestep(self.time(coordinates), a.diffusion_steps)[None]
        h = embed(a.params, self.inputs(coordinates), timesteps, a.hidden_t_dim, a.layer_norm_eps)
        for i in range(self.metadata.observation_layer):
            h = bert_layer(a.params, i, h, a.n_heads, a.layer_norm_eps)
        # The [CLS]/[SEP] rows take part in every layer but are not observed.
        return self.text_rows(h)[:, self.first_idx_observed_row:]

    def initialize(self, seed, prompt_std=1.0):
        """Independent draws: prompt rows N(0, prompt_std² I), generated rows DiffuSeq's N(0, I).

        N(0, I) is also how upstream initializes the embeddings before training (a heuristic
        for trained ones). If the attacker does not know the noise scale, the generated rows
        use noise_scale_guess instead of 1. At a later call they start from the noise share
        of x_t, sqrt(1 - alpha_bar(t0)) times the draw: a heuristic, since x_t is unknown.
        """
        if not np.isfinite(prompt_std) or prompt_std < 0:
            raise ValueError("prompt_std must be finite and nonnegative")
        noise_std = self.knowledge.noise_scale(1.0)
        pk, gk = jax.random.split(jax.random.PRNGKey(seed))
        tk = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
        t0 = self.metadata.time if self.knowledge.knows_time else jax.random.uniform(tk, minval=0.01, maxval=0.99)
        d = self.adapter.embedding_dim
        p = jax.random.normal(pk, (1, self.n, d)) * prompt_std
        g = jax.random.normal(gk, (1, self.generated, d)) * noise_std
        if self.metadata.forward_mode == "sampling_pass":
            g = g * self.adapter.noise_share(t0)
        x = self.coordinates(p, g)
        if not self.knowledge.knows_time:
            x = jnp.append(x, jax.scipy.special.logit(t0))
        return x

    def closed_form(self, observation):
        """None: even layer 0 is nonlinear (tanh up-projection and LayerNorm)."""
        return None
