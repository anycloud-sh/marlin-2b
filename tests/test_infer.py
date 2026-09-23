import copy
import json
import tempfile
import unittest
from pathlib import Path

from infer import read_requests, validate_result

ROOT = Path(__file__).resolve().parents[1]


class InputAndResultTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads((ROOT / "validation/reference.json").read_text())

    def test_actual_creator_results_pass_without_mutation(self):
        for row in self.reference:
            result = copy.deepcopy(row["result"])
            validate_result(
                result, row["case"]["mode"], row["clip"]["duration_seconds"]
            )
            self.assertEqual(result, row["result"])

    def test_rejects_out_of_range_nonfinite_and_reversed_timestamps(self):
        for start, end in [(0, 30), (float("nan"), 2), (2, 1), (True, 3)]:
            result = copy.deepcopy(self.reference[0]["result"])
            result["events"][0].update(start=start, end=end)
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_result(result, "caption", 11)

    def test_rejects_parser_failure_and_empty_caption_events(self):
        for result, mode in [
            ({"format_ok": False, "span": None}, "find"),
            ({"scene": "Tree", "events": []}, "caption"),
        ]:
            with self.assertRaises(ValueError):
                validate_result(result, mode, 11)

    def test_rejects_duplicate_ids_bad_modes_and_empty_queries(self):
        row = {"id": "one", "video": "video.mp4", "mode": "caption"}
        for rows in [
            [row, row],
            [{**row, "mode": "train"}],
            [{**row, "mode": "find", "event": " "}],
        ]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "inputs.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    read_requests(path)

    def test_custom_video_paths_are_relative_to_uploaded_manifest(self):
        rows, raw = read_requests(ROOT / "examples/find.jsonl")
        self.assertEqual(rows[0]["path"], (ROOT / "examples/bbb-3.mp4").resolve())
        self.assertEqual(rows[0]["event"], "a squirrel looks at a purple butterfly")
        self.assertTrue(raw)


if __name__ == "__main__":
    unittest.main()
