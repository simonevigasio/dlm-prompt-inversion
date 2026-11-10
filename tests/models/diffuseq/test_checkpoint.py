"""Explicit opt-in validation with the pinned released DiffuSeq checkpoint (QQP)."""
import os
import unittest

import numpy as np

from elf_inversion.models.diffuseq.adapter import DiffuSeqAdapter
from elf_inversion.observation import generate, prompt_tokens


@unittest.skipUnless(os.environ.get("DIFFUSEQ_CHECKPOINTS") == "1",
                     "Set DIFFUSEQ_CHECKPOINTS=1 with pinned artifacts cached")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DiffuSeqAdapter.from_config(local_files_only=True)

    def test_release(self):
        a = self.adapter
        self.assertEqual((a.embedding_dim, a.hidden_size, a.n_layers, a.seq_len, a.diffusion_steps),
                         (128, 768, 12, 128, 2000))
        self.assertEqual(a.embedding_table.shape[0], len(a.tokenizer))

    def test_native_forward_equivalence(self):
        # generate() raises ForwardMismatch unless the JAX map reproduces the PyTorch sampler.
        tokens = prompt_tokens(self.adapter, prompt="how can i learn python quickly?")
        for layer, step in ((0, 0), (3, 0), (12, 0), (3, 20)):
            with self.subTest(layer=layer, step=step):
                observation, adapter = generate(self.adapter, tokens, observation_layer=layer, step=step)
                # Prompt rows + generated rows: the 128 rows without [CLS] and [SEP] [SEP].
                self.assertEqual(observation.hidden_state.shape, (1, 125, 768))
                self.assertLess(observation.generator["forward_equivalence"]["maximum_absolute_error"], 1e-4)
                np.testing.assert_array_equal(
                    observation.truth["prompt_code"],
                    adapter.encode_token_ids(adapter.prepare_token_ids(tokens), "prompt_code"))


if __name__ == "__main__":
    unittest.main()
