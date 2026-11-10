"""local_files_only modes of the pinned Hugging Face downloads, shared by every model family."""
import unittest

from elf_inversion.models.hub import from_hub


class HubLoadingModes(unittest.TestCase):
    def test_auto_reads_cache_first_and_downloads_only_on_a_miss(self):
        calls = []

        def cached(local):
            calls.append(local)
            return "cache"
        self.assertEqual(from_hub(cached, "auto"), "cache")
        self.assertEqual(calls, [True])

        calls.clear()

        def missing(local):
            calls.append(local)
            if local:
                raise FileNotFoundError("not cached")
            return "hub"
        self.assertEqual(from_hub(missing, "auto"), "hub")
        self.assertEqual(calls, [True, False])
        with self.assertRaises(FileNotFoundError):
            from_hub(missing, True)
        with self.assertRaises(ValueError):
            from_hub(missing, "yes")


if __name__ == "__main__":
    unittest.main()
