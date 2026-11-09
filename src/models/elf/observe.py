"""ELF's native sampler as the observation oracle: run it and capture one hidden state."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ...observation import SyntheticExample, schedule_steps
from .forward import BackboneForward, ForwardMetadata


def _apply_with_mask(adapter, attention_mask):
    def apply_fn(variables, x, t, **kwargs):
        if attention_mask is not None:
            kwargs["attention_mask"] = jnp.asarray(attention_mask)[None, :]
        return adapter.model.apply(variables, x, t, **kwargs)
    return apply_fn


def observe(adapter, tokens, metadata: ForwardMetadata, *, noise_seed, sampling_steps):
    """Run ELF's native sampler up to the observed call and capture its hidden state.

    A "sampling_pass" at time t first takes ELF's native Euler ODE steps from 0 to t
    on the uniform ``sampling_steps`` schedule (conditional calls only, no external
    CFG), carrying z_t and the previous prediction exactly as the sampler does.
    Returned codes are for post-hoc scoring only; recover() never accepts them.
    """
    forward = BackboneForward(adapter, metadata)  # validates the declared layout
    batch = adapter.prepare_token_ids(tokens)
    if batch.input_ids.shape[0] != 1:
        raise ValueError("One synthetic example at a time")
    c = adapter.encode_text(batch)
    noise = jax.random.normal(jax.random.PRNGKey(noise_seed), c.shape) * adapter.config.denoiser_noise_scale
    sampling = adapter.native["sampling"]
    z = sampling.restore_cond(noise, c, batch.cond_seq_mask)
    x_pred = None
    if metadata.forward_mode == "sampling_pass":
        steps, k = schedule_steps(metadata.time, sampling_steps)
        x_pred = sampling.restore_cond(jnp.zeros_like(z), c, batch.cond_seq_mask)
        step_apply = _apply_with_mask(adapter, metadata.attention_mask)
        for i in range(k):
            z, x_pred = sampling._ode_step(
                step_apply, adapter.model_params, z, jnp.asarray(steps[i]), jnp.asarray(steps[i + 1]), x_pred,
                adapter.config, 1.0, metadata.self_cond_cfg_scale, c, batch.cond_seq_mask)
    captured = {}

    def intercept(next_fun, args, kwargs, context):
        if context.module.name == "blocks_0" and context.method_name == "__call__":
            captured["initial"] = args[0]
        return next_fun(*args, **kwargs)

    def apply_fn(variables, x, t, **kwargs):
        if metadata.attention_mask is not None:
            kwargs["attention_mask"] = jnp.asarray(metadata.attention_mask)[None, :]
        output, state = adapter.model.apply(
            variables, x, t, **kwargs,
            capture_intermediates=lambda mod, method: method == "__call__" and
                (mod.name == "proj1" or mod.name == f"blocks_{metadata.observation_layer - 1}"),
            mutable=["intermediates"])
        captured["state"] = state["intermediates"]
        return output

    with jax.disable_jit(), nn.intercept_methods(intercept):
        adapter.native["sampling"]._forward_sample_self_cond(
            apply_fn, adapter.model_params, z, jnp.asarray([metadata.time]), x_pred,
            adapter.config, metadata.self_cond_cfg_scale, c, batch.cond_seq_mask)
    state = captured["state"]
    layer = metadata.observation_layer
    h = captured["initial"] if layer == 0 else state[f"blocks_{layer - 1}"]["__call__"][0]
    codes = state["text_proj"]["proj1"]["__call__"][0]
    h = h[:, forward.prefix_length:]  # the prefix rows are not part of the observation
    assert h.shape == forward.observation_shape
    return SyntheticExample(h, codes[:, :metadata.prompt_length], codes[:, metadata.prompt_length:],
                            np.asarray(tokens).reshape(-1))
