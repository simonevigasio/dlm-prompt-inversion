"""Observations: synthetic hidden states of a victim model from a known prompt, saved and loaded.

Model-independent: the model-specific parts (its sampler, metadata and forward
map) come from the adapter, a VictimModel (models/interface.py). The hidden state
comes from the model's own sampler and full model, never from Stage A's inverse
code. The prompt and its codes are kept separately and are only used afterwards
to score a recovery.

An observation directory contains:

    hidden_state.npy   text rows (prompt + generated) of the hidden state, without any
                       non-text prefix rows (e.g. ELF's time/scale/mode tokens):
                       [1, text_length, hidden_size]
    observation.json   model family and metadata declared to Stage A, plus how it was generated
    truth.npz          prompt tokens and codes, for evaluation only

To invert a hidden state from elsewhere, write the same hidden_state.npy and an
observation.json with a "metadata" section; truth.npz is then omitted.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import models

HIDDEN_STATE, METADATA, TRUTH = "hidden_state.npy", "observation.json", "truth.npz"


@dataclass(frozen=True)
class SyntheticExample:
    observation: jax.Array
    prompt_code: jax.Array
    noise_code: jax.Array
    tokens: np.ndarray


@dataclass(frozen=True)
class Observation:
    hidden_state: np.ndarray
    metadata: object               # the model family's ForwardMetadata
    truth: dict | None = None      # tokens, prompt_code, noise_code; evaluation only
    generator: dict | None = None  # how a synthetic observation was produced
    model: str = models.DEFAULT_FAMILY  # family of the model that produced it


class ForwardMismatch(AssertionError):
    """Stage A's forward map does not reproduce the native observation of one noise seed."""

    def __init__(self, message, noise_seed, maximum_absolute_error):
        super().__init__(message)
        self.noise_seed, self.maximum_absolute_error = noise_seed, maximum_absolute_error


def prompt_tokens(adapter, prompt=None, token_ids=None):
    """Token IDs for exactly one of a text prompt or explicit IDs; no special tokens."""
    if (prompt is None) == (token_ids is None):
        raise ValueError("Give exactly one of a prompt text or token IDs")
    if prompt is not None:
        token_ids = adapter.tokenizer(prompt, add_special_tokens=False, padding=False,
                                      truncation=False)["input_ids"]
    tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    if not len(tokens):
        raise ValueError("The prompt is empty")
    if np.isin(tokens, adapter.tokenizer.all_special_ids).any():
        raise ValueError("Synthetic prompts must not contain special tokens")
    return tokens


def schedule_steps(time, sampling_steps):
    """The uniform time schedule ``linspace(0, 1, sampling_steps + 1)`` and the index of ``time`` on it.

    Every model family maps its sampler calls to this grid: time k / sampling_steps is call k.
    """
    if type(sampling_steps) is not int or sampling_steps < 1:
        raise ValueError("sampling_steps must be an integer >= 1")
    steps = np.asarray(jnp.linspace(0.0, 1.0, sampling_steps + 1))
    k = int(np.argmin(np.abs(steps - time)))
    if abs(steps[k] - time) > 1e-6 or k == sampling_steps:
        raise ValueError(f"time {time} is not a sampler call of the uniform {sampling_steps}-step schedule "
                         f"(k / {sampling_steps} for k = 0 ... {sampling_steps - 1})")
    return steps, k


def observe(adapter, tokens, metadata, *, noise_seed=100, sampling_steps=None) -> SyntheticExample:
    """Run the model's own sampler up to the observed call and capture its hidden state.

    Dispatches to ``adapter.observe``; ``sampling_steps`` defaults to the model's own.
    Returned codes are for post-hoc scoring only; recover() never accepts them.
    """
    return adapter.observe(tokens, metadata, noise_seed=noise_seed,
                           sampling_steps=adapter.sampling_steps if sampling_steps is None else sampling_steps)


def generate(adapter, tokens, *, observation_layer, noise_seed=100, time=0.0, sampling_steps=None, **options):
    """Synthetic observation for ``tokens``, checked against Stage A's forward map.

    ``time`` 0 observes the first sampling pass. A later time must be a step
    ``k / sampling_steps`` of the uniform schedule (default: the model's own number
    of steps); it observes that sampler call. ``options`` are model-specific metadata
    (the family's OBSERVATION_OPTIONS, e.g. ELF's self_cond_cfg_scale and attention_mask).
    Returns an Observation (with truth) and the adapter resized to the prompt.
    """
    adapter = adapter.with_prompt_length(len(tokens))
    sampling_steps = adapter.sampling_steps if sampling_steps is None else sampling_steps
    later = time != 0.0
    if later:
        steps, k = schedule_steps(time, sampling_steps)
        time = float(steps[k])  # exactly the float32 time the sampler passes
    metadata = adapter.metadata(len(tokens), observation_layer, time=time,
                                forward_mode="sampling_pass" if later else "first_sampling_pass", **options)
    truth = observe(adapter, tokens, metadata, noise_seed=noise_seed, sampling_steps=sampling_steps)
    # Stage A's forward must reproduce the native observation from the true codes.
    forward = adapter.forward_map(metadata)
    reconstructed = jax.jit(forward)(forward.coordinates(truth.prompt_code, truth.noise_code))
    difference = np.asarray(reconstructed) - np.asarray(truth.observation)
    # atol scales with the tensor: float32 reordering under jit loses ~1e-5 of the magnitude,
    # which a per-element tolerance flags on values that cancel to near zero.
    scale = float(np.max(np.abs(truth.observation)))
    try:
        np.testing.assert_allclose(reconstructed, truth.observation, atol=1e-5 * scale, rtol=1e-4,
                                   err_msg="Stage A forward does not reproduce the native observation")
    except AssertionError as error:
        raise ForwardMismatch(str(error), noise_seed, float(np.max(np.abs(difference)))) from None
    generator = {"noise_seed": noise_seed, **({"sampling_steps": sampling_steps, "schedule": "uniform"}
                                              if later else {}), "forward_equivalence": {
        "absolute_residual": float(np.linalg.norm(difference)),
        "maximum_absolute_error": float(np.max(np.abs(difference)))}}
    observation = Observation(np.asarray(truth.observation), metadata, {
        "tokens": truth.tokens, "prompt_code": np.asarray(truth.prompt_code),
        "noise_code": np.asarray(truth.noise_code)}, generator, adapter.family)
    return observation, adapter


def save_observation(directory, observation: Observation):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / HIDDEN_STATE, observation.hidden_state, allow_pickle=False)
    info = {"model": observation.model, "metadata": asdict(observation.metadata)}
    if observation.generator is not None:
        info["generator"] = observation.generator
    (directory / METADATA).write_text(json.dumps(info, indent=2) + "\n")
    if observation.truth is not None:
        np.savez(directory / TRUTH, **observation.truth)


def load_observation(directory) -> Observation:
    directory = Path(directory)
    if not (directory / HIDDEN_STATE).is_file() or not (directory / METADATA).is_file():
        raise FileNotFoundError(f"{directory} needs {HIDDEN_STATE} and {METADATA}")
    info = json.loads((directory / METADATA).read_text())
    # Files written before other families existed carry no "model" key: they are ELF observations.
    family = info.get("model", models.DEFAULT_FAMILY)
    metadata = models.family(family).load_metadata(info["metadata"])
    truth = None
    if (directory / TRUTH).is_file():
        with np.load(directory / TRUTH, allow_pickle=False) as arrays:
            truth = {key: arrays[key] for key in arrays.files}
    return Observation(np.load(directory / HIDDEN_STATE, allow_pickle=False),
                       metadata, truth, info.get("generator"), family)

