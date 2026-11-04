import copy
import unittest

import flax.core
import jax
import jax.numpy as jnp
import numpy as np

from elf_inversion.models.elf.forward import BackboneForward, ForwardMetadata
from elf_inversion.models.elf.testing import tiny_native_adapter
from elf_inversion.observation import generate, observe
from elf_inversion.backbone.interface import ForwardMap
from elf_inversion.backbone.knowledge import AttackerKnowledge


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_native_adapter()

    def test_native_forward_equivalence_all_layers_and_masks(self):
        for layer in (0, 1, 2):
            for mask in (None, (1, 1, 0, 0)):
                with self.subTest(layer=layer, mask=mask):
                    metadata = ForwardMetadata(1, layer, self_cond_cfg_scale=1.5, attention_mask=mask)
                    f = BackboneForward(self.adapter, metadata)
                    truth = observe(self.adapter, [4], metadata)
                    coordinates = f.coordinates(truth.prompt_code, truth.noise_code)
                    np.testing.assert_allclose(jax.jit(f)(coordinates), truth.observation, atol=3e-5, rtol=3e-5)
                    target = self.adapter.encode_token_ids(self.adapter.prepare_token_ids([4]), "projected")
                    np.testing.assert_allclose(f.codes(coordinates)[0], target, atol=2e-6)
                    self.assertGreater(float(jnp.linalg.norm(f.bias)), 0)

    def test_observation_has_only_text_rows(self):
        # The time/scale/mode prefix rows are computed but not observed.
        model = self.adapter.model
        for layer in (0, 1, 2):
            with self.subTest(layer=layer):
                metadata = ForwardMetadata(1, layer)
                f = BackboneForward(self.adapter, metadata)
                self.assertEqual(f.observation_shape, (1, model.max_length, model.hidden_size))
                self.assertEqual(f(f.initialize(seed=0)).shape, f.observation_shape)
                self.assertEqual(observe(self.adapter, [4], metadata).observation.shape, f.observation_shape)

    def test_later_sampling_pass_forward_equivalence(self):
        # Two Euler steps of a 4-step schedule, then the observed call at t = 0.5.
        for layer in (0, 1, 2):
            for mask in (None, (1, 1, 0, 0)):
                with self.subTest(layer=layer, mask=mask):
                    metadata = ForwardMetadata(1, layer, time=0.5, self_cond_cfg_scale=1.5, attention_mask=mask,
                                               forward_mode="sampling_pass")
                    f = BackboneForward(self.adapter, metadata)
                    truth = observe(self.adapter, [4], metadata, sampling_steps=4)
                    coordinates = f.coordinates(truth.prompt_code, truth.noise_code)
                    np.testing.assert_allclose(f.codes(coordinates)[1], truth.noise_code, atol=2e-6)
                    np.testing.assert_allclose(jax.jit(f)(coordinates), truth.observation, atol=3e-5, rtol=3e-5)
                    target = self.adapter.encode_token_ids(self.adapter.prepare_token_ids([4]), "projected")
                    np.testing.assert_allclose(f.codes(coordinates)[0], target, atol=2e-6)

    def test_later_sampling_pass_differs_from_first_pass(self):
        first = observe(self.adapter, [4], ForwardMetadata(1, 1))
        later = observe(self.adapter, [4], ForwardMetadata(1, 1, time=0.5, forward_mode="sampling_pass"),
                        sampling_steps=4)
        np.testing.assert_allclose(later.prompt_code, first.prompt_code, atol=2e-6)
        self.assertFalse(np.allclose(later.noise_code, first.noise_code, atol=1e-3))
        f = BackboneForward(self.adapter, ForwardMetadata(1, 1, time=0.5, forward_mode="sampling_pass"))
        self.assertEqual(f.noise_map.shape[0], 2 * self.adapter.model.text_encoder_dim)

    def test_generate_later_sampling_pass(self):
        observation, _ = generate(self.adapter, [4], observation_layer=1, time=0.75, sampling_steps=4)
        self.assertEqual(observation.metadata.forward_mode, "sampling_pass")
        self.assertEqual(observation.metadata.time, 0.75)
        self.assertEqual(observation.generator["sampling_steps"], 4)
        first, _ = generate(self.adapter, [4], observation_layer=1)
        self.assertEqual(first.metadata.forward_mode, "first_sampling_pass")
        self.assertNotIn("sampling_steps", first.generator)
        for time in (0.3, 1.0):  # off the 4-step grid; the final time is never a call
            with self.subTest(time=time), self.assertRaises(ValueError):
                generate(self.adapter, [4], observation_layer=1, time=time, sampling_steps=4)

    def test_jvp_vjp_adjoint_and_finite_difference(self):
        f = BackboneForward(self.adapter, ForwardMetadata(1, 1))
        x = f.initialize(8)
        v = jax.random.normal(jax.random.PRNGKey(9), x.shape)
        y, jv = jax.jvp(f, (x,), (v,))
        w = jax.random.normal(jax.random.PRNGKey(10), y.shape)
        _, pullback = jax.vjp(f, x)
        np.testing.assert_allclose(jnp.vdot(jv, w), jnp.vdot(v, pullback(w)[0]), atol=2e-5, rtol=2e-5)
        finite_difference = (f(x + 1e-3 * v) - f(x - 1e-3 * v)) / 2e-3
        np.testing.assert_allclose(jv, finite_difference, atol=2e-3, rtol=2e-3)

    def test_algebraic_code_recovery(self):
        for m in (ForwardMetadata(1, 0), ForwardMetadata(1, 0, time=0.5, forward_mode="sampling_pass")):
            with self.subTest(forward_mode=m.forward_mode):
                f = BackboneForward(self.adapter, m)
                truth = observe(self.adapter, [4], m, sampling_steps=4)
                recovered = f.algebraic(truth.observation)
                self.assertIsNotNone(recovered)
                p, g = f.codes(recovered)
                np.testing.assert_allclose(p, truth.prompt_code, atol=3e-6)
                np.testing.assert_allclose(g, truth.noise_code, atol=3e-6)

    def test_unsupported_metadata(self):
        for kwargs in ({"time": .5}, {"observation_layer": 3}, {"attention_mask": (1, 0)},
                       {"forward_mode": "later_sampling_pass"},
                       {"time": 1.0, "forward_mode": "sampling_pass"},
                       {"time": -0.25, "forward_mode": "sampling_pass"},
                       {"time": float("nan"), "forward_mode": "sampling_pass"}):
            with self.assertRaises(ValueError):
                BackboneForward(self.adapter, ForwardMetadata(1, **kwargs))

    def test_initialization_enforces_deficient_affine_code_spaces(self):
        adapter = copy.copy(self.adapter)
        params = flax.core.unfreeze(adapter.model_params)
        w1 = params["text_proj"]["proj1"]["kernel"]
        params["text_proj"]["proj1"]["kernel"] = w1.at[:, 2].set(w1[:, 0])
        adapter.model_params = flax.core.freeze(params)
        f = BackboneForward(adapter, ForwardMetadata(1, 0))
        self.assertEqual(f.prompt_rank, 2)
        self.assertEqual(f.noise_rank, 2)
        p, g = f.codes(f.initialize(123))
        np.testing.assert_allclose(p[..., 0], p[..., 2], atol=1e-6)
        np.testing.assert_allclose(g[..., 0], g[..., 2], atol=1e-6)
        # Initialization is a pushforward of raw Gaussian draws, including ELF's
        # scale and affine bias, rather than standard Gaussian projected codes.
        pk, gk = jax.random.split(jax.random.PRNGKey(123))
        raw_prompt = jax.random.normal(pk, (1, 1, 8))
        raw_noise = jax.random.normal(gk, (1, 3, 8)) * adapter.config.denoiser_noise_scale
        np.testing.assert_allclose(p, raw_prompt @ f.prompt_map + f.bias, atol=2e-6)
        np.testing.assert_allclose(g, raw_noise @ f.noise_map + f.bias, atol=2e-6)
    
    def test_noise_std_is_used_only_when_noise_scale_is_unknown(self):
        true_scale = self.adapter.config.denoiser_noise_scale
        for m in (ForwardMetadata(1), ForwardMetadata(1, time=0.5, forward_mode="sampling_pass")):
            with self.subTest(forward_mode=m.forward_mode):
                f_known = BackboneForward(self.adapter, m)
                f_unknown = BackboneForward(self.adapter, m, AttackerKnowledge(knows_noise_scale=False))
                # Known scale: the noise_std argument is ignored.
                np.testing.assert_allclose(f_known.initialize(10, noise_std=5.0), f_known.initialize(10))
                # Unknown scale guessed right: same start as the known map.
                np.testing.assert_allclose(f_unknown.initialize(10, noise_std=true_scale), f_known.initialize(10))
                # Unknown scale guessed wrong: different noise, same prompt.
                p_known, g_known = f_known.codes(f_known.initialize(10))
                p_unknown, g_unknown = f_unknown.codes(f_unknown.initialize(10, noise_std=5.0))
                np.testing.assert_allclose(p_unknown, p_known)
                self.assertFalse(np.allclose(g_unknown, g_known))

    def test_the_cut_using_only_generated_rows(self):
        # "generated" hides the prompt rows: the forward map still computes them
        # (they reach the generated rows through attention) but does not return them.
        model = self.adapter.model
        for layer in (0, 1, 2):
            with self.subTest(layer=layer):
                m = ForwardMetadata(1, layer)
                n = m.prompt_length
                f_all = BackboneForward(self.adapter, m)
                f_gen = BackboneForward(self.adapter, m, AttackerKnowledge(observed_rows="generated"))
                self.assertEqual(f_gen.observation_shape, (1, model.max_length - n, model.hidden_size))

                # Forward map: same coordinates, only the prompt rows are dropped.
                coordinates = f_all.initialize(seed=0)
                np.testing.assert_allclose(f_gen(coordinates), f_all(coordinates)[:, n:])

                # Data: select_rows applies the same cut to a full text-row observation.
                full = observe(self.adapter, [4], m).observation
                np.testing.assert_allclose(f_gen.select_rows(full), full[:, n:])
                np.testing.assert_allclose(f_all.select_rows(full), full)

                # No algebraic shortcut without the prompt rows (at layer 0 they are
                # the only place the prompt code appears).
                self.assertIsNone(f_gen.algebraic(f_gen.select_rows(full)))

    def test_unknown_time_is_the_last_coordinate(self):
        # Layer 1: at layer 0 the time rows are cut before any attention, so time has no effect.
        m = ForwardMetadata(1, 1, time=0.5, forward_mode="sampling_pass")
        f_known = BackboneForward(self.adapter, m)
        f_unknown = BackboneForward(self.adapter, m, AttackerKnowledge(knows_time=False))
        # The unknown-time map must still satisfy the model-agnostic protocol used by recover().
        self.assertIsInstance(f_unknown, ForwardMap)

        # Size: the unknown time adds exactly one coordinate, and initialize respects size.
        self.assertEqual(f_unknown.size, f_known.size + 1)
        for f in (f_known, f_unknown):
            self.assertEqual(len(f.initialize(0)), f.size)

        # The first pass reveals time 0, so an unknown time is rejected there.
        with self.assertRaises(ValueError):
            BackboneForward(self.adapter, ForwardMetadata(1), AttackerKnowledge(knows_time=False))

        # The last coordinate is the logit of the time: logit(0.5) = 0, so 0.0 means t = 0.5.
        x = f_unknown.initialize(0)
        np.testing.assert_allclose(f_unknown.time(x.at[-1].set(0.0)), 0.5)
        # With the true time in the last coordinate, the unknown-time map equals the known one.
        np.testing.assert_allclose(f_unknown(x.at[-1].set(0.0)), f_known(x[:-1]), atol=1e-6)
        # A wrong time (logit(0.9)) changes the output: the time really is a variable.
        wrong = float(jax.scipy.special.logit(0.9))
        self.assertFalse(np.allclose(f_unknown(x.at[-1].set(wrong)), f_known(x[:-1]), atol=1e-6))

        # Restarts start from different times, each in [0, 1).
        t0, t1 = (float(f_unknown.time(f_unknown.initialize(seed))) for seed in (0, 1))
        self.assertNotEqual(t0, t1)
        for t in (t0, t1):
            self.assertTrue(0.0 <= t < 1.0)


if __name__ == "__main__":
    unittest.main()
