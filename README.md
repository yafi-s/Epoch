# Epoch

Distributed hyperparameter optimization with:
- a C++ gRPC scheduler
- Python GA controller
- Python workers (local or Modal)

## Quick Start

### 1) Generate Python protobuf bindings
```bash
bash scripts/generate_protos.sh
```

### 2) Build scheduler
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target epoch_scheduler -j"$(nproc)"
```

### 3) Run tests
```bash
poetry run pytest
```

## Notes

- Secrets and local runtime outputs are intentionally gitignored.
- For distributed runs, configure worker auth via env vars or secret manager.
