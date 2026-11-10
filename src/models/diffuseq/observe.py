"""DiffuSeq's own PyTorch sampler as the observation oracle: run it and capture one hidden state.

Mirrors upstream sample_seq2seq.py with its default decoding (run_decode.py --step 2000) for
one source sequence: the source block keeps its clean embeddings, the generated rows start
from ``th.randn_like`` noise, and every diffusion step is one ``p_sample`` call, with
upstream's rounding ``denoised_fn_round`` and clip_denoised=False. Runs on the CPU in
float32, with the global PyTorch RNG forked and seeded by noise_seed.

Upstream's other decoder (``--step`` below the diffusion steps, DDIM with a gap) is not
supported: its ``ddim_sample`` steps to the next timestep index while the loop skips
``gap`` of them, and the samples it produces ignore the source.
"""
from functools import partial

import numpy as np

from ...observation import SyntheticExample
from ..schedule import time_step
from .forward import BackboneForward, ForwardMetadata


def observe(adapter, tokens, metadata: ForwardMetadata, *, noise_seed, sampling_steps):
    """Run upstream's sampler up to the observed call and capture its hidden state.

    A "sampling_pass" at time t = k / sampling_steps first takes upstream's k sampler steps.
    The observation holds the prompt rows, then the generated rows; the [CLS] and [SEP] rows of
    the source block are dropped. Returned codes (the prompt and generated rows of the model
    input) are for scoring only.
    """
    import torch

    forward = BackboneForward(adapter, metadata)  # validates the declared layout
    tokens = np.asarray(tokens).reshape(-1)
    adapter.prepare_token_ids(tokens)  # validates length and vocabulary
    steps = adapter.diffusion_steps
    if sampling_steps != steps:
        raise ValueError(f"DiffuSeq observations use upstream's default decoder: sampling_steps must be "
                         f"the {steps} diffusion steps")
    k = 0 if metadata.forward_mode == "first_sampling_pass" else time_step(metadata.time, sampling_steps)[1]
    indices = list(range(steps))[::-1]
    model, diffusion = adapter.torch_model, adapter.diffusion
    n, layer = metadata.prompt_length, metadata.observation_layer
    captured = {}

    def capture_input(module, args):
        captured["x"] = args[0].detach().clone()

    def capture_layer(module, args, output=None):
        captured["h"] = (args[0] if output is None else output[0]).detach().clone()

    ids, mask = adapter.source_ids(tokens)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(noise_seed))
        input_ids = torch.as_tensor(ids)[None]
        x_start = model.get_embeds(input_ids)
        input_mask = torch.broadcast_to(torch.as_tensor(mask)[None, :, None], x_start.shape)
        x = torch.where(input_mask == 0, x_start, torch.randn_like(x_start))
        model_emb = torch.nn.Embedding(num_embeddings=adapter.embedding_matrix.shape[0],
                                       embedding_dim=adapter.embedding_dim,
                                       _weight=model.word_embedding.weight.clone()).eval()
        denoised_fn = partial(adapter.upstream["rounding"].denoised_fn_round, None, model_emb)
        for i in indices[:k]:
            x = diffusion.p_sample(model, x, torch.tensor([i]), clip_denoised=False, denoised_fn=denoised_fn,
                                   model_kwargs={}, top_p=None, mask=input_mask, x_start=x_start)["sample"]
        encoder = model.input_transformers
        hooks = [model.register_forward_pre_hook(capture_input),
                 encoder.register_forward_pre_hook(capture_layer) if layer == 0
                 else encoder.layer[layer - 1].register_forward_hook(capture_layer)]
        try:
            diffusion.p_mean_variance(model, x, torch.tensor([indices[k]]), clip_denoised=False,
                                      denoised_fn=denoised_fn, model_kwargs={})
        finally:
            for hook in hooks:
                hook.remove()
    x, h = captured["x"].numpy(), captured["h"].numpy()
    assert h.shape == (1, adapter.seq_len, adapter.hidden_size)
    return SyntheticExample(np.asarray(forward.text_rows(h)), x[:, 1:n + 1], x[:, n + 3:], tokens)
