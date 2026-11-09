"""The model-independent attack (pipeline, observation files, CLI) on a tiny random DiffuSeq."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
import yaml

from elf_inversion.backbone.knowledge import AttackerKnowledge
from elf_inversion.backbone.solver import SolverConfig
from elf_inversion.cli import load_config, main
from elf_inversion.models.diffuseq.forward import ForwardMetadata
from elf_inversion.models.diffuseq.setup import BackboneSetup
from elf_inversion.models.diffuseq.testing import tiny_diffuseq_adapter
from elf_inversion.models.elf.testing import tiny_native_adapter
from elf_inversion.observation import load_observation, save_observation
from elf_inversion.pipeline import invert_observation, invert_prompt
from elf_inversion.tokens.attack import AttackConfig

SOLVER = SolverConfig(restarts=3, max_iterations=100, residual_atol=1e-4, residual_rtol=1e-5)
SEARCH = AttackConfig(mode="projected", k=9, max_iterations=3, stopping_loss_tolerance=1e-8)


class DiffuSeqPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_default_matmul_precision", "highest")
        cls.adapter = tiny_diffuseq_adapter()

    def assert_converged(self, result):
        self.assertEqual(result.stage_a.method, "levenberg_marquardt")
        self.assertTrue(result.stage_a.converged, (result.stage_a.stop_reason, result.stage_a.relative_residual))

    def assert_recovered(self, result, tokens, code_error=1e-2):
        self.assert_converged(result)
        np.testing.assert_array_equal(result.tokens, tokens)
        self.assertLess(result.evaluation["prompt_code_error"], code_error)

    def test_layers_recover_the_prompt(self):
        for layer in (0, 1, 2):
            with self.subTest(layer=layer):
                result = invert_prompt([4, 7], self.adapter, observation_layer=layer, solver=SOLVER,
                                       search=SEARCH, start_tokens=[5, 5])
                self.assert_recovered(result, [4, 7])

    def test_hidden_source_rows(self):
        # With the bottleneck, the generated rows alone still give the prompt token, though the
        # prompt reaches them only through attention, so its code is recovered less precisely.
        result = invert_prompt([4], self.adapter, observation_layer=2, solver=SOLVER, search=SEARCH,
                               start_tokens=[5], knowledge=AttackerKnowledge(observed_rows="generated"))
        self.assertEqual(result.stage_a.diagnostics["prompt_rank"], 4)
        self.assert_recovered(result, [4], code_error=0.5)

    def test_unknown_time_runs_end_to_end(self):
        # Plumbing only. The time enters through a sinusoidal embedding of tau in [0, 1000], so the
        # loss oscillates in the time coordinate and local search from a random time rarely finds it.
        result = invert_prompt([4], self.adapter, observation_layer=2, time=0.5, solver=SolverConfig(restarts=1),
                               search=SEARCH, start_tokens=[5], knowledge=AttackerKnowledge(knows_time=False))
        self.assertTrue(np.isfinite(result.stage_a.absolute_residual))
        self.assertEqual(result.evaluation["true_time"], 0.5)
        self.assertTrue(0 < result.evaluation["recovered_time"] < 1)

    def test_without_the_bottleneck(self):
        # Every row is a free hidden-width vector. Stage A fits the observation, but on this tiny random
        # model the fit is not unique: a different prompt row explains it too, so tokens are not asserted.
        wide = tiny_diffuseq_adapter(bottleneck=False)
        result = invert_prompt([4], wide, observation_layer=2, solver=SOLVER, search=SEARCH, start_tokens=[5])
        self.assertEqual(result.stage_a.diagnostics["prompt_rank"], wide.hidden_size)
        self.assert_converged(result)

    def test_observation_files_keep_the_family(self):
        result = invert_prompt([4], self.adapter, observation_layer=1, solver=SOLVER, search=SEARCH,
                               start_tokens=[5])
        with tempfile.TemporaryDirectory() as directory:
            save_observation(Path(directory) / "obs", result.observation)
            self.assertEqual(json.loads((Path(directory) / "obs/observation.json").read_text())["model"], "diffuseq")
            loaded = load_observation(Path(directory) / "obs")
        self.assertEqual(loaded.model, "diffuseq")
        self.assertEqual(loaded.metadata, ForwardMetadata(1, 1))
        again = invert_observation(loaded, self.adapter, solver=SOLVER, search=SEARCH, start_tokens=[5])
        np.testing.assert_array_equal(again.tokens, result.tokens)
        # An observation is inverted only with a model of its own family.
        with self.assertRaisesRegex(ValueError, "diffuseq observation"):
            invert_observation(loaded, tiny_native_adapter(), search=SEARCH, start_tokens=[5])


class DiffuSeqCommandTests(unittest.TestCase):
    def write_config(self, root, **overrides):
        (root / "model.yaml").write_text("family: diffuseq\n")
        (root / "stage_a.yaml").write_text(yaml.safe_dump({"restarts": 2, "max_iterations": 100}))
        (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "k": 9, "max_iterations": 2}))
        config = {"model": "model.yaml", "stage_a": "stage_a.yaml", "stage_b": "stage_b.yaml",
                  "synthetic_observation": {"observation_layer": 1}, **overrides}
        (root / "pipeline.yaml").write_text(yaml.safe_dump(config))
        return root / "pipeline.yaml"

    def test_config_uses_the_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            section = {"weights": "random", "bottleneck": False, "init": "normal"}
            config = load_config(self.write_config(root, backbone=section))
            self.assertEqual(config.family, "diffuseq")
            self.assertEqual(config.backbone, BackboneSetup(**section))
            self.assertEqual(config.resolved["family"], "diffuseq")
            with self.assertRaisesRegex(ValueError, "allowed keys for diffuseq"):  # ELF-only option
                load_config(self.write_config(root, synthetic_observation={"observation_layer": 0,
                                                                           "self_cond_cfg_scale": 1.0}))

    def test_dataset_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root, backbone={"weights": "random", "init": "normal", "init_seed": 2})
            dataset = root / "prompts.jsonl"
            dataset.write_text(json.dumps({"id": "a", "text": "4"}) + "\n" + json.dumps({"id": "b", "text": "1"}) + "\n")
            with patch("elf_inversion.models.diffuseq.adapter.DiffuSeqAdapter.from_config",
                       return_value=tiny_diffuseq_adapter()) as from_config:
                main(["--config", str(config), "--dataset", str(dataset), "--output", str(root / "run"),
                      "--device", "cpu"])
            self.assertEqual(from_config.call_args.kwargs["setup"],
                             BackboneSetup(weights="random", init="normal", init_seed=2))
            rows = [json.loads(line) for line in (root / "run/results.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["done", "skipped"])  # token 1 is special
            self.assertTrue(rows[0]["exact_recovery"])
            manifest = json.loads((root / "run/manifest.json").read_text())
            self.assertEqual(manifest["configuration"]["family"], "diffuseq")


if __name__ == "__main__":
    unittest.main()
