# Marlin-2B on AnyCloud

Turn short video clips into timestamped scene/event captions and locate a named event with Marlin-2B on cloud GPUs.

This AnyCloud-powered artifact calls the creator's `caption()` and `find()`
helpers using the [pinned Marlin release](https://huggingface.co/NemoStation/Marlin-2B).
An approved Hugging Face account token is required at runtime.

## Validation status

The unwrapped upstream examples passed on a Lambda A10. Three captions produced
parsed events within their clip durations, and `find()` located the selected
butterfly scene at 3.0–4.0 seconds. The coarse event-window smoke check passed;
this is not a benchmark accuracy claim. [Reference outputs](validation/reference.json).

Model loading plus four inference calls took 49.4 seconds and used 5.96 GB peak
allocated GPU memory. Provisioning, image pulling, and dependency installation
are additional. [Reference metadata](validation/reference-metadata.json).

The portable image is being built and validated. A release command will be
published after the exact candidate digest passes Lambda validation.

## Inputs and outputs

Inputs are JSONL rows with unique `id`, a local `video` path, and `mode` set to
`caption` or `find`. `find` also requires a non-empty `event` query. Relative
video paths resolve beside the input manifest. See [caption inputs](examples/captions.jsonl)
and [the event query](examples/find.jsonl).

Outputs are `predictions.jsonl`, `summary.json`, and `inputs.jsonl`. Each
prediction retains the upstream result, raw generation/token IDs, preprocessing
tensor hashes, and any inference/validation error. The wrapper checks parsed
intervals without clipping or repairing them. Invalid input fails before model
loading; per-clip inference failures are retained and fail the batch.

Videos must have a positive duration of at most 120 seconds. The smoke fixtures
are approximately 11 seconds; resource needs depend on the input. Upstream
defaults remain greedy decoding with 2048 tokens for caption and 64 for find.
The complete pinned local snapshot keeps the helper's lazy processor lookup on
the same model revision. No optional compilation is enabled.
Find returns a model-predicted span and has no event-absence signal.

## Provenance and development

Model revision: `fd111fca4fc7897876fb0d7e9df22ca5ac8ab965`.
Runtime: PyTorch 2.11.0 / CUDA 12.8, Transformers 5.7.0, TorchCodec 0.11.0 CPU
decoder, BF16. Dependencies are installed in an isolated virtual environment.
Weights and creator code download from the gated model repository at runtime.

Code/model: Apache-2.0. The short Big Buck Bunny trailer excerpts are CC BY 3.0;
source hashes, extraction times, and modifications are in [the manifest](examples/manifest.json).
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

CPU tests: `python -m unittest discover -s tests -v`. Hosted Linux builds the
candidate image; Lambda through AnyCloud supplies the real GPU validation.
