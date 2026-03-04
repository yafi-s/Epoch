#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "/mnt/a/Careers and Jobs/Coding/Epoch"

echo "=== Installing Python dependencies ==="
poetry install --no-root

echo "=== Installing TensorFlow with CUDA ==="
poetry run pip install 'tensorflow[and-cuda]'

echo "=== Checking GPU detection ==="
poetry run python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF version:', tf.__version__)
print('GPUs found:', gpus)
if gpus:
    print('CUDA acceleration is ENABLED')
else:
    print('WARNING: No GPU detected')
"

echo "=== Generating Python protobuf files ==="
bash scripts/generate_protos.sh

echo "=== Building C++ scheduler ==="
mkdir -p build-wsl && cd build-wsl
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target epoch_scheduler --parallel

echo "=== Setup complete ==="
