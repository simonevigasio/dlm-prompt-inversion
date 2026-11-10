"""The Dolly dataset command: prompt length is counted with the chosen family's tokenizer."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from elf_inversion.datasets import dolly


class WordTokenizer:
    """One token per word; "<s>" is a special token and "rare" lies outside the vocabulary."""
    all_special_ids, vocab_size, name_or_path = [0], 100, "word-tokenizer"

    def __call__(self, text, **kwargs):
        return {"input_ids": [{"<s>": 0, "rare": 100}.get(word, 1 + len(word)) for word in text.split()]}


SOURCE = "\n".join(json.dumps(row) for row in (
    {"instruction": "name three colours", "context": ""},
    {"instruction": "name three colours", "context": ""},           # duplicate text
    {"instruction": "summarise the text", "context": "some text"},   # needs a context
    {"instruction": "too short", "context": ""},
    {"instruction": "contains <s> token", "context": ""},            # special token
    {"instruction": "contains rare word", "context": ""},            # outside the vocabulary
    {"instruction": "list four fruits", "context": ""},
)).encode()


class DollyTests(unittest.TestCase):
    def test_eligible_prompts(self):
        eligible, total = dolly.eligible_prompts(SOURCE, WordTokenizer(), tokens=3)
        self.assertEqual(total, 7)
        self.assertEqual(eligible, [{"id": "dolly-00000", "text": "name three colours"},
                                    {"id": "dolly-00006", "text": "list four fruits"}])

    def test_command_uses_the_family_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompts.jsonl"
            with patch("elf_inversion.datasets.dolly.download_source", return_value=SOURCE), \
                    patch("elf_inversion.models.diffuseq.family.load_tokenizer", return_value=WordTokenizer()) as load:
                dolly.main(["--family", "diffuseq", "--tokens", "3", "--count", "2", "--output", str(output)])
            load.assert_called_once()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["dolly-00000", "dolly-00006"])
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual((manifest["family"], manifest["tokenizer"], manifest["exact_token_count"]),
                             ("diffuseq", "word-tokenizer", 3))

    def test_family_is_required(self):
        for arguments in ([], ["--family", "gpt"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                dolly.main(arguments)


if __name__ == "__main__":
    unittest.main()
