import unittest

from elf_inversion.models.diffuseq.setup import BackboneSetup


class BackboneSetupTests(unittest.TestCase):
    def test_defaults_are_the_released_checkpoint(self):
        self.assertEqual(BackboneSetup(), BackboneSetup("checkpoint", True, "upstream", 0.02, 0))

    def test_rules(self):
        BackboneSetup(weights="random", init="normal", init_std=0.1, init_seed=3)
        BackboneSetup(weights="random", bottleneck=False, init="normal")
        for bad in ({"weights": "pretrained"}, {"init": "xavier"}, {"init": "normal"}, {"bottleneck": False},
                    {"bottleneck": 0}, {"weights": "random"},
                    {"weights": "random", "init": "normal", "init_std": 0},
                    {"weights": "random", "init": "normal", "init_seed": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BackboneSetup(**bad)


if __name__ == "__main__":
    unittest.main()
