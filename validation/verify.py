"""Verify exact creator-result, preprocessing, and token parity on Lambda."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(root, image):
    from infer import validate_result

    assert re.fullmatch(r"ghcr.io/anycloud-sh/marlin-2b@sha256:[0-9a-f]{64}", image)
    reference = {
        r["case"]["id"]: r
        for r in json.loads((ROOT / "validation/reference.json").read_text())
    }
    metadata = json.loads((ROOT / "validation/reference-metadata.json").read_text())
    checks = {}
    for name, fixture in [("default", "captions.jsonl"), ("custom", "find.jsonl")]:
        directory = root / name
        status = json.loads((directory / "workload.json").read_text())
        assert status["state"] == "completed" and status["cleanedAt"]
        assert status["imageDigest"] == image.split("@", 1)[1]
        summary = json.loads((directory / "output/summary.json").read_text())
        for key in [
            "model",
            "model_revision",
            "modeling_sha256",
            "processor_config_sha256",
            "torch",
            "transformers",
            "cuda",
            "gpu",
            "video_environment",
            "video_processor",
            "compile",
        ]:
            assert summary[key] == metadata[key], f"{name}: mismatched {key}"
        assert summary["model_revision"] == "fd111fca4fc7897876fb0d7e9df22ca5ac8ab965"
        assert (
            summary["cuda_matrix_check"] == "passed"
            and summary["all_parameters_on_cuda"] is True
        )
        raw = (ROOT / "examples" / fixture).read_bytes()
        assert (directory / "output/inputs.jsonl").read_bytes() == raw
        assert summary["inputs_sha256"] == hashlib.sha256(raw).hexdigest()
        inputs = [json.loads(line) for line in raw.splitlines()]
        predictions = [
            json.loads(line)
            for line in (directory / "output/predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(predictions) == len(inputs) == summary["input_count"]
        assert summary["failed_rows"] == 0
        for supplied, actual in zip(inputs, predictions):
            expected = reference[supplied["id"]]
            case = expected["case"]
            assert actual["id"] == supplied["id"] and actual["mode"] == supplied["mode"]
            assert actual["error"] is None
            assert actual["clip"]["sha256"] == expected["clip"]["sha256"]
            assert actual["result"] == expected["result"], (
                "upstream helper result changed"
            )
            assert actual["generation"] == expected["generation"], (
                "preprocessed input or generated token mismatch"
            )
            validate_result(
                actual["result"], actual["mode"], actual["clip"]["duration_seconds"]
            )
            if actual["mode"] == "caption":
                text = actual["result"]["caption"].lower()
                assert all(
                    any(word in text for word in group)
                    for group in case["visible_concepts"]
                )
            else:
                assert actual["event"] == case["event"]
                a, b = actual["result"]["span"]
                x, y = case["expected_span_seconds"]
                iou = max(0, min(b, y) - max(a, x)) / (max(b, y) - min(a, x))
                assert iou >= case["minimum_iou"]
        checks[name] = {
            "rows": len(predictions),
            "helper_result_parity": "exact",
            "token_parity": "exact",
            "preprocessed_tensor_parity": "exact",
            "persisted_after_cleanup": True,
        }
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence, args.image), indent=2))
