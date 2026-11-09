"""BackboneSetup validation: which victim-model setups are accepted and which are rejected."""
import unittest

from elf_inversion.models.elf.setup import BackboneSetup


class BackboneSetupDefaults(unittest.TestCase):
    def test_defaults_reproduce_the_released_checkpoint(self):
        setup = BackboneSetup()
        self.assertEqual(setup.weights, "checkpoint")
        self.assertTrue(setup.bottleneck)
        self.assertEqual(setup.init, "upstream")
        self.assertEqual(setup.init_std, 0.02)
        self.assertEqual(setup.init_seed, 0)

    def test_is_frozen(self):
        with self.assertRaises(AttributeError):
            BackboneSetup().weights = "random"


class BackboneSetupAccepted(unittest.TestCase):
    def test_random_normal_with_bottleneck(self):
        BackboneSetup(weights="random", init="normal", init_std=0.1, init_seed=3)

    def test_random_normal_without_bottleneck(self):
        BackboneSetup(weights="random", bottleneck=False, init="normal")

    def test_integer_init_std_from_yaml(self):
        self.assertEqual(BackboneSetup(weights="random", init="normal", init_std=1).init_std, 1)


class BackboneSetupRejected(unittest.TestCase):
    def assertRejected(self, **fields):
        with self.assertRaises(ValueError):
            BackboneSetup(**fields)

    def test_unknown_weights(self):
        self.assertRejected(weights="pretrained")

    def test_unknown_init(self):
        self.assertRejected(weights="random", init="xavier")

    def test_bottleneck_must_be_bool(self):
        self.assertRejected(weights="random", bottleneck=0)

    def test_init_std_must_be_positive(self):
        self.assertRejected(weights="random", init="normal", init_std=0.0)
        self.assertRejected(weights="random", init="normal", init_std=-0.1)

    def test_random_rejects_upstream_init(self):
        self.assertRejected(weights="random")
        self.assertRejected(weights="random", init="upstream")

    def test_init_std_must_be_a_number(self):
        self.assertRejected(weights="random", init="normal", init_std="0.1")
        self.assertRejected(weights="random", init="normal", init_std=True)

    def test_init_seed_must_be_int(self):
        self.assertRejected(weights="random", init="normal", init_seed=1.0)
        self.assertRejected(weights="random", init="normal", init_seed=True)

    def test_checkpoint_requires_bottleneck(self):
        self.assertRejected(bottleneck=False)

    def test_checkpoint_requires_upstream_init(self):
        self.assertRejected(init="normal")


if __name__ == "__main__":
    unittest.main()
