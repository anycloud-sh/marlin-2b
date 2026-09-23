"""Prepare isolated dependencies while reusing the image's installed PyTorch."""

import subprocess
import sys

PYTHON = "/opt/marlin-runtime/bin/python"
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(
    ["apt-get", "install", "-y", "--no-install-recommends", "ffmpeg", "python3-venv"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "venv", "--system-site-packages", "/opt/marlin-runtime"],
    check=True,
)
subprocess.run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "transformers==5.7.0",
        "accelerate==1.14.0",
        "huggingface-hub==1.7.0",
        "tokenizers==0.22.2",
        "safetensors==0.7.0",
        "qwen-vl-utils==0.0.14",
        "av==17.0.0",
        "pillow==12.3.0",
        "jinja2==3.1.6",
    ],
    check=True,
)
subprocess.run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
        "torchcodec==0.11.0",
    ],
    check=True,
)
subprocess.run(
    [
        PYTHON,
        "-c",
        "import torch, transformers, torchcodec; from transformers import Qwen3_5ForConditionalGeneration; assert torch.__version__.split('+')[0] == '2.11.0'; print('Runtime imports:', torch.__version__, transformers.__version__, torchcodec.__version__)",
    ],
    check=True,
)
if len(sys.argv) == 2:
    subprocess.run(
        [
            PYTHON,
            "-c",
            "import sys; from torchcodec.decoders import VideoDecoder; d=VideoDecoder(sys.argv[1]); print('CPU video decode:', d[0].shape, d.metadata.duration_seconds)",
            sys.argv[1],
        ],
        check=True,
    )
