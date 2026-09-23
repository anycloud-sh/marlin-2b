"""Run the pinned creator's caption/find helpers without a portability wrapper."""

import functools
import hashlib
import json
import os
import time
from pathlib import Path

# Keep the creator's documented environment in place before loading its code.
VIDEO_ENV = {"FORCE_QWENVL_VIDEO_READER": "torchcodec", "VIDEO_MAX_PIXELS": "200704", "FPS": "2.0", "FPS_MAX_FRAMES": "240", "FPS_MIN_FRAMES": "4"}
os.environ.update(VIDEO_ENV)

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

MODEL = "NemoStation/Marlin-2B"
REVISION = "fd111fca4fc7897876fb0d7e9df22ca5ac8ab965"
INPUT = Path("/mnt/input/fixtures")
OUTPUT = Path("/mnt/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((INPUT / "manifest.json").read_text())
cases = json.loads(os.environ["MARLIN_CASES_JSON"])
start = time.monotonic()
assert torch.cuda.is_available()
matrix = torch.ones((32, 32), device="cuda")
product = matrix @ matrix
torch.cuda.synchronize()
assert product.is_cuda and bool(torch.all(product == 32).item())
del matrix, product
torch.manual_seed(42)
snapshot = snapshot_download(MODEL, revision=REVISION)
# A local pinned snapshot also pins the helper's lazy AutoProcessor lookup,
# which otherwise omits a revision and follows the remote repository's HEAD.
model = AutoModelForCausalLM.from_pretrained(
    snapshot, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"},
)
model.eval()
assert all(p.device.type == "cuda" for p in model.parameters())
processor = model.processor
loaded = time.monotonic()
original_generate = model.generate
capture = {}


@functools.wraps(original_generate)
def observed_generate(*args, **kwargs):
    capture.clear()
    for key in ["input_ids", "pixel_values_videos", "video_grid_thw"]:
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            capture[key] = {"shape": list(value.shape), "sha256": hashlib.sha256(raw).hexdigest()}
            if key == "video_grid_thw":
                capture[key]["values"] = value.tolist()
    result = original_generate(*args, **kwargs)
    tokens = result[:, kwargs["input_ids"].shape[1]:]
    capture["generated_token_ids"] = tokens[0].tolist()
    capture["raw_generation"] = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    return result


model.generate = observed_generate
results = []
for case in cases:
    clip = next(c for c in manifest["clips"] if c["id"] == case["clip_id"])
    path = INPUT / clip["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"]
    before = time.monotonic()
    if case["mode"] == "caption":
        result = model.caption(str(path))
    else:
        assert case["mode"] == "find"
        result = model.find(str(path), event=case["event"])
    torch.cuda.synchronize()
    row = {"case": case, "clip": clip, "result": result, "generation": dict(capture), "elapsed_seconds": round(time.monotonic() - before, 3)}
    results.append(row)
    (OUTPUT / "reference.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(row), flush=True)
metadata = {
    "model": MODEL, "model_revision": REVISION,
    "modeling_sha256": hashlib.sha256((Path(snapshot) / "modeling_marlin.py").read_bytes()).hexdigest(),
    "processor_config_sha256": hashlib.sha256((Path(snapshot) / "processor_config.json").read_bytes()).hexdigest(),
    "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0), "cuda_matrix_check": "passed", "all_parameters_on_cuda": True,
    "video_environment": VIDEO_ENV, "video_processor": processor.video_processor.to_dict(),
    "model_load_seconds": round(loaded - start, 3), "total_seconds": round(time.monotonic() - start, 3),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "compile": False,
}
(OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
