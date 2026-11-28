"""The CLI must not emit or score with an invalid distribution temperature."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synthetic_panel import cli


def reject_nonstandard_number(value):
    raise ValueError(f"non-JSON number: {value}")


class TemperatureCLITests(unittest.TestCase):
    def setUp(self):
        scratch = self.enterContext(tempfile.TemporaryDirectory())
        self.reactions = Path(scratch) / "reactions.json"
        self.output_file = Path(scratch) / "score.json"
        self.reactions.write_text(
            json.dumps({
                "schema_version": 1,
                "variants": {"a": "A sample option"},
                "personas": ["example"],
                "n_samples": 1,
                "records": [
                    {"variant_id": "a", "persona_id": "example",
                     "sample_index": 0, "reaction": "Too short"}
                ],
            }),
            encoding="utf-8",
        )
        # An actual scoring/embedding invocation would make this fixture unsafe.
        for target in ("synthetic_panel.scoring.get_embedding",
                       "synthetic_panel.embeddings.get_embeddings_batch"):
            self.enterContext(patch(target, side_effect=AssertionError("unexpected embedding")))

    def invoke(self, *args):
        return cli.main(["score-reactions", str(self.reactions), "--json",
                         "--output", str(self.output_file), *args])

    def test_invalid_temperatures_fail_before_writing_either_json_sink(self):
        for raw in ("nan", "inf", "-inf", "1e999", "0", "-0", "-1"):
            with self.subTest(raw=raw):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    self.invoke(f"--dist-temperature={raw}")
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--dist-temperature", stderr.getvalue())
                self.assertIn("finite", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(self.output_file.exists())

    def test_finite_positive_values_and_default_produce_standard_json(self):
        for args, expected in (
            ((), 1.0),
            (("--dist-temperature=0.001",), 0.001),
            (("--dist-temperature=5e-324",), 5e-324),
            (("--dist-temperature=2.25",), 2.25),
        ):
            with self.subTest(args=args):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(self.invoke(*args), 0)
                document = json.loads(
                    stdout.getvalue(), parse_constant=reject_nonstandard_number
                )
                self.assertEqual(document["dist_temperature"], expected)
                self.assertEqual(document["comparability"]["verdict"], "not_comparable")
                self.assertEqual(self.output_file.read_text(), stdout.getvalue().rstrip("\n"))
