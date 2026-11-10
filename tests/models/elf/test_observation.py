"""ELF observations: generated, saved, loaded and inverted again, on a tiny native ELF."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from elf_inversion.backbone.solver import SolverConfig
from elf_inversion.models.diffuseq.testing import tiny_diffuseq_adapter
from elf_inversion.models.elf.forward import ForwardMetadata
from elf_inversion.models.elf.testing import tiny_native_adapter
from elf_inversion.models.schedule import step_time, time_step
from elf_inversion.observation import (ForwardMismatch, Observation, generate, load_observation, prompt_tokens,
                                       save_observation)
from elf_inversion.pipeline import invert_observation
from elf_inversion.tokens.attack import AttackConfig

SEARCH = AttackConfig(k=9, max_iterations=3)


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_native_adapter()

    def test_generate_describes_the_observation(self):
        observation, adapter = generate(self.adapter, [4], observation_layer=1, noise_seed=7)
        self.assertEqual(observation.model, "elf")
        self.assertEqual(observation.metadata, ForwardMetadata(1, 1))
        # Text rows only: [1, max_length, hidden_size].
        self.assertEqual(observation.hidden_state.shape, (1, adapter.model.max_length, adapter.model.hidden_size))
        self.assertEqual(set(observation.truth), {"tokens", "prompt_code", "noise_code"})
        self.assertEqual((observation.generator["noise_seed"], observation.generator["step"]), (7, 0))
        self.assertLess(observation.generator["forward_equivalence"]["maximum_absolute_error"], 1e-4)
        # Another noise seed gives another hidden state for the same prompt.
        other, _ = generate(self.adapter, [4], observation_layer=1, noise_seed=8)
        self.assertFalse(np.allclose(other.hidden_state, observation.hidden_state))
        np.testing.assert_allclose(other.truth["prompt_code"], observation.truth["prompt_code"], atol=1e-6)

    def test_generate_resizes_the_adapter_to_the_prompt(self):
        observation, adapter = generate(self.adapter, [4, 5], observation_layer=0)
        self.assertEqual((adapter.prompt_length, observation.metadata.prompt_length), (2, 2))
        self.assertEqual(self.adapter.prompt_length, 1)  # the original adapter is untouched

    def test_save_and_load_round_trip(self):
        # ELF's own options travel in the metadata; the mask is a list in JSON and a tuple again after loading.
        observation, _ = generate(self.adapter, [4], observation_layer=1, step=2, sampling_steps=4,
                                  self_cond_cfg_scale=1.5, attention_mask=[1, 1, 0, 0])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "obs"
            save_observation(folder, observation)
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["hidden_state.npy", "observation.json", "truth.npz"])
            self.assertEqual(json.loads((folder / "observation.json").read_text())["model"], "elf")
            with self.assertRaises(FileExistsError):  # an observation is never overwritten
                save_observation(folder, observation)
            loaded = load_observation(folder)
        self.assertEqual(loaded.metadata, observation.metadata)
        self.assertEqual(loaded.metadata.attention_mask, (1, 1, 0, 0))
        self.assertEqual((loaded.model, loaded.generator), ("elf", observation.generator))
        np.testing.assert_array_equal(loaded.hidden_state, observation.hidden_state)
        for key in ("tokens", "prompt_code", "noise_code"):
            np.testing.assert_array_equal(loaded.truth[key], observation.truth[key])

    def test_loaded_observation_is_inverted_like_the_original(self):
        observation, _ = generate(self.adapter, [4], observation_layer=0)
        with tempfile.TemporaryDirectory() as directory:
            save_observation(Path(directory) / "obs", observation)
            loaded = load_observation(Path(directory) / "obs")
        result = invert_observation(loaded, self.adapter, solver=SolverConfig(restarts=1), search=SEARCH,
                                    start_tokens=[5])
        np.testing.assert_array_equal(result.tokens, [4])
        self.assertTrue(result.evaluation["stage_b_exact_tokens"])

    def test_observation_from_elsewhere_has_no_truth(self):
        # A hidden state that was not generated here: only the array and its metadata, no "model" key.
        generated, _ = generate(self.adapter, [4], observation_layer=0)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.save(folder / "hidden_state.npy", generated.hidden_state)
            (folder / "observation.json").write_text(json.dumps(
                {"metadata": {"prompt_length": 1, "observation_layer": 0}}))
            loaded = load_observation(folder)
            self.assertEqual((loaded.model, loaded.truth, loaded.generator), ("elf", None, None))
            with self.assertRaises(FileNotFoundError):
                load_observation(folder / "missing")
        result = invert_observation(loaded, self.adapter, solver=SolverConfig(restarts=1), search=SEARCH,
                                    start_tokens=[5])
        self.assertIsNone(result.evaluation)  # nothing to score against
        np.testing.assert_array_equal(result.tokens, [4])

    def test_observation_needs_a_model_of_its_family(self):
        observation, _ = generate(self.adapter, [4], observation_layer=0)
        with self.assertRaisesRegex(ValueError, "elf observation"):
            invert_observation(observation, tiny_diffuseq_adapter(), search=SEARCH, start_tokens=[5])

    def test_forward_mismatch_is_reported_with_its_noise_seed(self):
        # A hidden state the forward map cannot reproduce: generate() refuses to return it.
        original = type(self.adapter).observe

        def shifted(adapter, tokens, metadata, **kwargs):
            example = original(adapter, tokens, metadata, **kwargs)
            return type(example)(example.observation + 1.0, example.prompt_code, example.noise_code, example.tokens)

        adapter = self.adapter.with_prompt_length(1)
        adapter.observe = lambda *args, **kwargs: shifted(adapter, *args, **kwargs)
        with self.assertRaises(ForwardMismatch) as raised:
            generate(adapter, [4], observation_layer=1, noise_seed=9)
        self.assertEqual(raised.exception.noise_seed, 9)
        self.assertGreater(raised.exception.maximum_absolute_error, 0.5)

    def test_prompt_tokens_and_schedule(self):
        np.testing.assert_array_equal(prompt_tokens(self.adapter, token_ids=[[4, 5]]), [4, 5])
        for bad in ({}, {"token_ids": []}, {"token_ids": [1]}, {"prompt": "x", "token_ids": [4]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):  # none, empty, special token, both
                prompt_tokens(self.adapter, **bad)
        # The uniform schedule ELF uses: call 2 of 4 is at time 0.5, and back.
        self.assertEqual(step_time(2, 4), 0.5)
        steps, k = time_step(0.5, 4)
        self.assertEqual((len(steps), k), (5, 2))
        for step, sampling_steps in ((4, 4), (-1, 4), (1.5, 4), (0, 0)):  # the end, negative, no call, no steps
            with self.subTest(step=step, sampling_steps=sampling_steps), self.assertRaises(ValueError):
                step_time(step, sampling_steps)
        for time in (0.3, 1.0):  # off the grid; the final time is never a call
            with self.subTest(time=time), self.assertRaises(ValueError):
                time_step(time, 4)
        self.assertIsNone(Observation(np.zeros((1, 4, 16)), ForwardMetadata(1)).truth)


if __name__ == "__main__":
    unittest.main()
