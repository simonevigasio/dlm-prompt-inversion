"""Sampler calls on a uniform time grid: a helper for model families whose sampler uses one.

The attack itself only speaks of sampler calls ("step" k = the k-th call, 0 = the first).
What time value a call corresponds to is the family's business; families with a uniform
grid ``linspace(0, 1, sampling_steps + 1)`` (ELF, DiffuSeq) convert with these functions.
"""
import jax.numpy as jnp
import numpy as np


def uniform_schedule(sampling_steps):
    """The float32 grid ``linspace(0, 1, sampling_steps + 1)``; call k is at entry k."""
    if type(sampling_steps) is not int or sampling_steps < 1:
        raise ValueError("sampling_steps must be an integer >= 1")
    return np.asarray(jnp.linspace(0.0, 1.0, sampling_steps + 1))


def step_time(step, sampling_steps):
    """Time of sampler call ``step``: exactly the float32 value the sampler passes."""
    steps = uniform_schedule(sampling_steps)
    if type(step) is not int or not 0 <= step < sampling_steps:
        raise ValueError(f"step must be a sampler call of the {sampling_steps}-step schedule "
                         f"(an integer from 0 to {sampling_steps - 1}), not {step!r}")
    return float(steps[step])


def time_step(time, sampling_steps):
    """Inverse of step_time: the grid and the call k whose time is ``time``."""
    steps = uniform_schedule(sampling_steps)
    k = int(np.argmin(np.abs(steps - time)))
    if abs(steps[k] - time) > 1e-6 or k == sampling_steps:
        raise ValueError(f"time {time} is not a sampler call of the uniform {sampling_steps}-step schedule "
                         f"(k / {sampling_steps} for k = 0 ... {sampling_steps - 1})")
    return steps, k
