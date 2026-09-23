"""Exercise real candidate failures on CPU without model downloads."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

image, report = sys.argv[1:]
checks = {}


def fails(name, command, expected):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 1 or expected not in result.stderr:
        raise RuntimeError(
            f"{name}: unexpected outcome {result.returncode}: {result.stderr[-2000:]}"
        )
    checks[name] = {"exit_code": result.returncode, "expected_error": expected}


fails(
    "missing_model_access",
    ["docker", "run", "--rm", "--pull=never", "--network=none", image],
    "HF_TOKEN is required",
)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "bad.mp4").write_text("This is not a video.")
    row = {"id": "bad-video", "video": "bad.mp4", "mode": "caption"}
    (root / "invalid.jsonl").write_text(json.dumps(row) + "\n")
    common = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "-v",
        directory + ":/inputs:ro",
    ]
    fails(
        "unreadable_video",
        [*common, image, "--input", "/inputs/invalid.jsonl"],
        "unreadable video",
    )
    (root / "duplicate.jsonl").write_text((json.dumps(row) + "\n") * 2)
    fails(
        "duplicate_ids",
        [*common, image, "--input", "/inputs/duplicate.jsonl"],
        "unique non-empty string",
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "-v",
            directory + ":/inputs",
            "--entrypoint",
            "/usr/bin/ffmpeg",
            image,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=16x16:rate=1:duration=121",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            "/inputs/long.mp4",
        ],
        check=True,
        timeout=60,
    )
    (root / "long.jsonl").write_text(
        json.dumps({"id": "too-long", "video": "long.mp4", "mode": "caption"}) + "\n"
    )
    fails(
        "unsupported_duration",
        [*common, image, "--input", "/inputs/long.jsonl"],
        "at most 120 seconds",
    )
Path(report).write_text(json.dumps({"image": image, "checks": checks}, indent=2) + "\n")
print(json.dumps(checks, indent=2))
