#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PROTO_DIR="$PROJECT_ROOT/protos"
PY_OUT="$PROJECT_ROOT/generated/python"

mkdir -p "$PY_OUT"

if command -v poetry &>/dev/null; then
    PYTHON_CMD=(poetry run python)
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=(python3)
else
    PYTHON_CMD=(python)
fi

echo "Generating Python protobuf/gRPC code..."
"${PYTHON_CMD[@]}" -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$PY_OUT" \
    --grpc_python_out="$PY_OUT" \
    --pyi_out="$PY_OUT" \
    "$PROTO_DIR/epoch.proto"

# Create __init__.py so the generated package is importable
touch "$PY_OUT/__init__.py"

echo "Python proto generation complete: $PY_OUT"
echo "(C++ proto generation is handled by CMake)"
