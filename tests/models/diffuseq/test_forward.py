"""DiffuSeq's JAX forward map against upstream's PyTorch model and sampler, on a tiny random DiffuSeq."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

from elf_inversion.backbone.interface import ForwardMap
from elf_inversion.backbone.knowledge import AttackerKnowledge
from elf_inversion.models.diffuseq.forward import BackboneForward, ForwardMetadata, model_timestep
from elf_inversion.models.diffuseq.testing import tiny_diffuseq_adapter
from elf_inversion.models.interface import VictimModel
from elf_inversion.observation import generate, observe
from elf_inversion.tokens.interface import TokenModel


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_diffuseq_adapter(prompt_length=2)
        cls.wide = tiny_diffuseq_adapter(prompt_length=2, bottleneck=False)

    def test_interfaces(self):
        self.assertIsInstance(self.adapter, VictimModel)
        self.assertIsInstance(self.adapter, TokenModel)
        self.assertIsInstance(BackboneForward(self.adapter, ForwardMetadata(2, 1)), ForwardMap)

    def test_full_model_matches_upstream(self):
        # The JAX port up to the last layer, against upstream's whole TransformerNetModel encoder.
        for adapter in (self.adapter, self.wide):
            with self.subTest(embedding_dim=adapter.embedding_dim):
                f = BackboneForward(adapter, ForwardMetadata(2, adapter.n_layers))
                x = f.initialize(3)
                captured = {}
                hook = adapter.torch_model.input_transformers.register_forward_hook(
                    lambda module, args, output: captured.update(h=output.last_hidden_state))
                try:
                    with torch.no_grad():
                        adapter.torch_model(torch.from_numpy(np.array(f.inputs(x))),
                                            torch.tensor([float(model_timestep(0.0, adapter.diffusion_steps))]))
                finally:
                    hook.remove()
                # The map returns the prompt and generated rows of upstream's state.
                np.testing.assert_allclose(f(x), f.text_rows(captured["h"].numpy()), atol=1e-4, rtol=1e-4)

    def test_native_forward_equivalence(self):
        for adapter in (self.adapter, self.wide):
            for layer in (0, 1, 2):
                for metadata, steps in ((ForwardMetadata(2, layer), None),
                                        (ForwardMetadata(2, layer, time=0.5, forward_mode="sampling_pass"), None),
                                        (ForwardMetadata(2, layer, time=0.25, forward_mode="sampling_pass"), 20)):
                    with self.subTest(embedding_dim=adapter.embedding_dim, layer=layer, steps=steps,
                                      forward_mode=metadata.forward_mode):
                        f = BackboneForward(adapter, metadata)
                        truth = observe(adapter, [4, 7], metadata, sampling_steps=steps)
                        coordinates = f.coordinates(truth.prompt_code, truth.noise_code)
                        np.testing.assert_allclose(jax.jit(f)(coordinates), truth.observation, atol=1e-5, rtol=1e-4)
                        # Stage A's prompt code is exactly Stage B's target: the prompt's embeddings.
                        target = adapter.encode_token_ids(adapter.prepare_token_ids([4, 7]), "prompt_code")
                        np.testing.assert_array_equal(truth.prompt_code, target)
                        self.assertEqual(truth.observation.shape, f.observation_shape)

    def test_source_layout(self):
        ids, mask = self.adapter.source_ids([4, 7])
        np.testing.assert_array_equal(ids, [1, 4, 7, 2, 2, 0, 0, 0])   # [CLS] p [SEP] [SEP] padding
        np.testing.assert_array_equal(mask, [0, 0, 0, 0, 0, 1, 1, 1])
        f = BackboneForward(self.adapter, ForwardMetadata(2, 1))
        self.assertEqual((f.source_length, f.generated, f.size), (5, 3, 2 * 4 + 3 * 4))

    def test_later_calls_change_only_the_generated_rows(self):
        first = observe(self.adapter, [4, 7], ForwardMetadata(2, 1))
        later = observe(self.adapter, [4, 7], ForwardMetadata(2, 1, time=0.5, forward_mode="sampling_pass"))
        np.testing.assert_array_equal(later.prompt_code, first.prompt_code)
        self.assertFalse(np.allclose(later.noise_code, first.noise_code, atol=1e-3))

    def test_generate_records_the_family(self):
        # Call 15 of the tiny model's 20 diffusion steps: time 15/20.
        observation, _ = generate(self.adapter, [4, 7], observation_layer=1, step=15)
        self.assertEqual(observation.model, "diffuseq")
        self.assertEqual(observation.metadata, ForwardMetadata(2, 1, 0.75, "sampling_pass"))
        with self.assertRaises(TypeError):  # ELF's options do not exist for DiffuSeq
            generate(self.adapter, [4, 7], observation_layer=1, self_cond_cfg_scale=2.0)
        with self.assertRaisesRegex(ValueError, "default decoder"):  # upstream's DDIM path is not supported
            generate(self.adapter, [4, 7], observation_layer=1, step=2, sampling_steps=4)
        with self.assertRaises(ValueError):  # a 20-step sampler has calls 0..19
            generate(self.adapter, [4, 7], observation_layer=1, step=20)

    def test_jvp_vjp_adjoint(self):
        f = BackboneForward(self.adapter, ForwardMetadata(2, 2, time=0.5, forward_mode="sampling_pass"),
                            AttackerKnowledge(knows_time=False))
        x = f.initialize(8)
        self.assertEqual(x.shape, (f.size,))
        v = jax.random.normal(jax.random.PRNGKey(9), x.shape)
        y, jv = jax.jvp(f, (x,), (v,))
        w = jax.random.normal(jax.random.PRNGKey(10), y.shape)
        _, pullback = jax.vjp(f, x)
        np.testing.assert_allclose(jnp.vdot(jv, w), jnp.vdot(v, pullback(w)[0]), atol=2e-4, rtol=2e-4)

    def test_observation_has_only_prompt_and_generated_rows(self):
        # 8 rows = [CLS] + 2 prompt + [SEP] [SEP] + 3 generated. [CLS]/[SEP] are never observed.
        metadata = ForwardMetadata(2, 1)
        f = BackboneForward(self.adapter, metadata)
        self.assertEqual(f.observation_shape, (1, 2 + 3, 16))
        self.assertEqual(f(f.initialize(0)).shape, f.observation_shape)
        truth = observe(self.adapter, [4, 7], metadata)
        self.assertEqual(truth.observation.shape, f.observation_shape)
        # text_rows keeps rows 1, 2 (prompt) and 5, 6, 7 (generated) of a full state.
        full = np.arange(8, dtype=np.float32)[None, :, None] * np.ones((1, 8, 16), np.float32)
        np.testing.assert_array_equal(np.asarray(f.text_rows(full))[0, :, 0], [1, 2, 5, 6, 7])

    def test_observed_rows(self):
        f = BackboneForward(self.adapter, ForwardMetadata(2, 1), AttackerKnowledge(observed_rows="generated"))
        self.assertEqual(f.observation_shape, (1, 3, 16))  # only the generated rows
        full = observe(self.adapter, [4, 7], ForwardMetadata(2, 1)).observation
        np.testing.assert_array_equal(f.select_rows(full), full[:, 2:])
        self.assertEqual(f(f.initialize(0)).shape, f.observation_shape)

    def test_unsupported_metadata(self):
        for kwargs in ({"time": .5}, {"observation_layer": 3}, {"prompt_length": 5},
                       {"forward_mode": "later_sampling_pass"},
                       {"time": 1.0, "forward_mode": "sampling_pass"},
                       {"time": float("nan"), "forward_mode": "sampling_pass"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BackboneForward(self.adapter, ForwardMetadata(**{"prompt_length": 2, **kwargs}))
        with self.assertRaises(ValueError):
            BackboneForward(self.adapter, ForwardMetadata(2, 1), AttackerKnowledge(knows_time=False))

    def test_random_parameters_are_reproducible(self):
        name = "input_transformers.layer.0.attention.self.query.weight"
        np.testing.assert_array_equal(tiny_diffuseq_adapter().params[name], self.adapter.params[name])
        self.assertFalse(np.allclose(tiny_diffuseq_adapter(init_seed=1).params[name], self.adapter.params[name]))
        self.assertNotIn("input_up_proj.0.weight", self.wide.params)


if __name__ == "__main__":
    unittest.main()
