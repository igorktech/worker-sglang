# CUDA 12 nightly from main — v0.5.18 has generic DSpark, not LFM2.
# Pin the digest-equivalent tag so rebuilds do not silently jump.
FROM lmsysorg/sglang:nightly-dev-cu12-20260825-1ec20fd2

# Keep handler and sglang output unbuffered so container logs appear live;
# without this a slow or failing startup produces no logs at all.
ENV PYTHONUNBUFFERED=1

# LFM2 / LFM2-MoE DSpark (https://github.com/sgl-project/sglang/pull/31041).
# Drop this layer once a tagged image includes that PR.
ARG SGLANG_DSPARK_SHA=6fa5223c0692bcc54f3521c740aa9c97c3c9ad14
RUN python3 - <<PY
import pathlib
import urllib.request

import sglang

sha = "${SGLANG_DSPARK_SHA}"
root = pathlib.Path(sglang.__file__).resolve().parent
files = (
    "kernels/ops/speculative/dspark/fused_kv_write.py",
    "srt/models/dspark.py",
    "srt/models/lfm2.py",
    "srt/models/lfm2_dspark.py",
    "srt/models/lfm2_moe.py",
)
base = f"https://raw.githubusercontent.com/tugot17/sglang/{sha}/python/sglang/"
for rel in files:
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + rel) as resp:
        dest.write_bytes(resp.read())
    print(f"overlaid {rel} -> {dest}")
PY

WORKDIR /sgl-workspace

# Install into the same interpreter CMD uses. The nightly image's `python3` is
# not the one `uv pip install --system` targets, which produced
# `ModuleNotFoundError: No module named 'runpod'` on RunPod.
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && python3 -c "import runpod"

# copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Model download script execution
# Ensure this script uses python3 and handles paths correctly relative to /app if needed
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 download_model.py; \
    fi

CMD ["python3", "handler.py"]
