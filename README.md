# Marlin-2B on AnyCloud

Turn short video clips into timestamped scene/event captions and locate a named event with Marlin-2B on cloud GPUs.

This AnyCloud-powered Job calls the creator's [Marlin helpers](https://huggingface.co/NemoStation/Marlin-2B).
Both the helper result and the unmodified model generation are retained.

## Run

First accept access on the model page and run `hf auth login`. Requires a
configured AnyCloud API, a Lambda credential named `lambda`, and an AWS storage
credential named `artifact-storage`; adjust names/region for your accounts.
Create the model-access Secret once, then run the included three-video batch:

```bash
anycloud secrets new marlin-hf "HF_TOKEN=$(hf auth token --quiet)"
export MARLIN_OUTPUT="marlin-$(date +%s)-$RANDOM"
anycloud job ghcr.io/anycloud-sh/marlin-2b:0.1.0-fd111fc-r1 \
  --credentials lambda --gpu-type a10 --gpus all --secret marlin-hf \
  --output-bucket "$MARLIN_OUTPUT" \
  --output-storage-credentials artifact-storage --output-storage-region us-east-1
```

After the Job completes, download its concrete predictions:

```bash
anycloud bucket download "$MARLIN_OUTPUT" predictions.jsonl ./predictions.jsonl \
  --credentials artifact-storage --region us-east-1
```

[Three example clips](examples/captions.jsonl) produce scenes and timestamped
events in `predictions.jsonl`, plus `summary.json` and the input manifest.
The output Bucket retains them after compute cleanup. Short batches restart
from the beginning; there are no training checkpoints.

## Locate an event in an uploaded video

From a checkout of this repository, upload the example files and event query.
Replace the video and manifest with your own inputs to use the same path:

```bash
export MARLIN_INPUT="${MARLIN_OUTPUT}-input"
anycloud bucket create "$MARLIN_INPUT" --credentials artifact-storage --region us-east-1
anycloud bucket upload "$MARLIN_INPUT" ./examples '' --recursive \
  --credentials artifact-storage --region us-east-1
anycloud job ghcr.io/anycloud-sh/marlin-2b:0.1.0-fd111fc-r1 \
  --credentials lambda --gpu-type a10 --gpus all --secret marlin-hf \
  --input-bucket "$MARLIN_INPUT" \
  --input-storage-credentials artifact-storage --input-storage-region us-east-1 \
  --output-bucket "${MARLIN_OUTPUT}-find" \
  --output-storage-credentials artifact-storage --output-storage-region us-east-1 \
  -- --input /mnt/input/find.jsonl
```

The included query returns **3.0–4.0 seconds** for the squirrel/butterfly scene.
JSONL rows need unique `id`, `video`, and `mode` (`caption` or `find`); `find`
also needs `event`. Video paths resolve beside the manifest. Videos must be
positive-length and at most 120 seconds. Invalid inputs fail before weights
load; inference/parser errors are retained per clip and fail the batch.
Find reports a predicted span and has no event-absence signal.

## Validation and requirements

Validated with AnyCloud 0.1.64 on Lambda A10 (24 GB), Linux/amd64. Default and
custom paths match upstream helper results, tokens, and preprocessing tensors
exactly; outputs survive compute cleanup. [Evidence](validation/lambda-a10/checks.json).
Measured model loading plus inference: **45.2 s** for three captions, **14.8 s**
for the custom query, **5.96 GB** peak GPU allocation. Provisioning and pulling
add time. Cold downloads: about 4.55 GB of image layers and 5.5 GB of model files.
These short clips are smoke examples, not benchmark accuracy scores.

Image digest: `sha256:4f26458f3753febf5e628bfd30b31c2100e1e6a51983fa0d78de31395281cfbd`.
Model/code revision: `fd111fca4fc7897876fb0d7e9df22ca5ac8ab965`.
Runtime: PyTorch 2.11.0 / CUDA 12.8, Transformers 5.7.0, TorchCodec 0.11.0 CPU,
BF16; exact creator preprocessing and greedy generation defaults are retained.
Weights and model code download at runtime using each operator's approved token.
Code/model: Apache-2.0; Big Buck Bunny excerpts: CC BY 3.0.
See [LICENSE](LICENSE), [NOTICE](NOTICE), and [clip provenance](examples/manifest.json).
CPU tests: `python -m unittest discover -s tests -v`.
