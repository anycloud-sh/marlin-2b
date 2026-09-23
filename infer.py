"""AnyCloud-powered batch calls to the pinned creator's Marlin helpers."""

import argparse
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "NemoStation/Marlin-2B"
REVISION = "fd111fca4fc7897876fb0d7e9df22ca5ac8ab965"
VIDEO_ENV = {
    "FORCE_QWENVL_VIDEO_READER": "torchcodec",
    "VIDEO_MAX_PIXELS": "200704",
    "FPS": "2.0",
    "FPS_MAX_FRAMES": "240",
    "FPS_MIN_FRAMES": "4",
}
DEFAULT_INPUT = Path(__file__).parent / "examples/captions.jsonl"


def read_requests(path):
    path = Path(path)
    raw = path.read_bytes()
    rows, seen = [], set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: expected an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"line {number}: id must be a unique non-empty string")
        if not isinstance(row.get("video"), str) or not row["video"]:
            raise ValueError(f"{identifier}: video must be a file path")
        mode = row.get("mode", "caption")
        if not isinstance(mode, str) or mode not in {"caption", "find"}:
            raise ValueError(f"{identifier}: unsupported mode {mode}")
        if mode == "find" and (
            not isinstance(row.get("event"), str) or not row["event"].strip()
        ):
            raise ValueError(f"{identifier}: find requires a non-empty event query")
        rows.append(
            {**row, "mode": mode, "path": (path.parent / row["video"]).resolve()}
        )
        seen.add(identifier)
    if not rows:
        raise ValueError("input contains no requests")
    return rows, raw


def probe_video(path):
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode:
        raise ValueError(f"unreadable video: {path.name}")
    data = json.loads(probe.stdout)
    video = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise ValueError(f"no video stream: {path.name}")
    duration = float(video.get("duration") or data["format"]["duration"])
    if not math.isfinite(duration) or not 0 < duration <= 120:
        raise ValueError("video duration must be positive and at most 120 seconds")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "file": path.name,
        "sha256": digest,
        "duration_seconds": duration,
        "width": video["width"],
        "height": video["height"],
        "source_fps": video.get("avg_frame_rate"),
    }


def validate_result(result, mode, duration):
    def interval(start, end):
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in [start, end]
        ):
            raise ValueError("timestamps must be finite numbers")
        if not 0 <= start < end <= duration:
            raise ValueError("timestamp interval is outside the clip or non-positive")

    if mode == "caption":
        if not result.get("scene") or not result.get("events"):
            raise ValueError("upstream result has no parsed scene or events")
        for event in result["events"]:
            interval(event["start"], event["end"])
            if not event.get("description"):
                raise ValueError("event description is empty")
    else:
        if not result.get("format_ok") or result.get("span") is None:
            raise ValueError("upstream could not parse the event span")
        interval(*result["span"])


def run(args):
    requests, raw_input = read_requests(args.input)
    clips = {row["id"]: probe_video(row["path"]) for row in requests}
    if not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN is required; provide an AnyCloud Secret for an account with Marlin access"
        )
    for key, value in VIDEO_ENV.items():
        os.environ.setdefault(key, value)
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    start = time.monotonic()
    matrix = torch.ones((32, 32), device="cuda")
    product = matrix @ matrix
    torch.cuda.synchronize()
    if not product.is_cuda or not bool(torch.all(product == 32).item()):
        raise RuntimeError("CUDA matrix check failed")
    del matrix, product
    torch.manual_seed(42)
    snapshot = snapshot_download(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    model.eval()
    if not all(p.device.type == "cuda" for p in model.parameters()):
        raise RuntimeError("Model parameters are not all on CUDA")
    processor = model.processor
    loaded = time.monotonic()
    original_generate = model.generate
    capture = {}

    @functools.wraps(original_generate)
    def observed_generate(*arguments, **keywords):
        capture.clear()
        for key in ["input_ids", "pixel_values_videos", "video_grid_thw"]:
            value = keywords.get(key)
            if isinstance(value, torch.Tensor):
                raw = (
                    value.detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                capture[key] = {
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                if key == "video_grid_thw":
                    capture[key]["values"] = value.tolist()
        result = original_generate(*arguments, **keywords)
        tokens = result[:, keywords["input_ids"].shape[1] :]
        capture["generated_token_ids"] = tokens[0].tolist()
        capture["raw_generation"] = processor.batch_decode(
            tokens, skip_special_tokens=True
        )[0]
        return result

    model.generate = observed_generate
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs.jsonl").write_bytes(raw_input)
    failures = 0
    with (output / "predictions.jsonl").open("w") as stream:
        for row in requests:
            prediction = {
                "id": row["id"],
                "mode": row["mode"],
                "clip": clips[row["id"]],
                "event": row.get("event"),
                "result": None,
                "generation": None,
                "error": None,
            }
            before = time.monotonic()
            capture.clear()
            try:
                result = (
                    model.caption(str(row["path"]))
                    if row["mode"] == "caption"
                    else model.find(str(row["path"]), event=row["event"])
                )
                torch.cuda.synchronize()
                prediction["result"] = result
                validate_result(
                    result, row["mode"], prediction["clip"]["duration_seconds"]
                )
            except Exception as error:
                prediction["error"] = f"{type(error).__name__}: {error}"
                failures += 1
            prediction["generation"] = dict(capture)
            prediction["elapsed_seconds"] = round(time.monotonic() - before, 3)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(prediction, ensure_ascii=False), flush=True)
    summary = {
        "model": MODEL,
        "model_revision": REVISION,
        "modeling_sha256": hashlib.sha256(
            (Path(snapshot) / "modeling_marlin.py").read_bytes()
        ).hexdigest(),
        "processor_config_sha256": hashlib.sha256(
            (Path(snapshot) / "processor_config.json").read_bytes()
        ).hexdigest(),
        "inputs_sha256": hashlib.sha256(raw_input).hexdigest(),
        "input_count": len(requests),
        "failed_rows": failures,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_matrix_check": "passed",
        "all_parameters_on_cuda": True,
        "video_environment": {key: os.environ[key] for key in VIDEO_ENV},
        "video_processor": processor.video_processor.to_dict(),
        "model_load_seconds": round(loaded - start, 3),
        "total_seconds": round(time.monotonic() - start, 3),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "compile": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch timestamped video captions and event localization with Marlin-2B."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/output"))
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
