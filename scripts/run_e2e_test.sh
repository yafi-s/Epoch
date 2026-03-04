#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════╗"
echo "║      Epoch End-to-End Test           ║"
echo "╚══════════════════════════════════════╝"

cleanup() {
    echo "Cleaning up..."
    kill "$SCHEDULER_PID" 2>/dev/null || true
    kill "$W0_PID" 2>/dev/null || true
    kill "$W1_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Build scheduler
echo "Building scheduler..."
cmake --build build --target epoch_scheduler -j"$(nproc)" 2>&1

# Generate Python protos
echo "Generating Python protos..."
bash scripts/generate_protos.sh

# Start scheduler
echo "Starting scheduler..."
./build/scheduler/epoch_scheduler --listen-address 0.0.0.0:50051 &
SCHEDULER_PID=$!
sleep 2

# Start 2 workers
echo "Starting workers..."
python -m worker.main --scheduler-address localhost:50051 --worker-id w0 --log-level INFO &
W0_PID=$!
python -m worker.main --scheduler-address localhost:50051 --worker-id w1 --log-level INFO &
W1_PID=$!
sleep 3

# Run GA controller with small population for quick test
echo "Running GA controller (pop=4, gen=2, epochs=1)..."
python -c "
import sys
sys.path.insert(0, '.')
from ga.controller import GAController, ControllerConfig
from ga.engine import GAConfig
from metrics.collector import MetricsStore

ga_cfg = GAConfig(population_size=4, num_generations=2, dataset='mnist', epochs=1, seed=42)
ctrl_cfg = ControllerConfig(scheduler_address='localhost:50051', ga_config=ga_cfg)
store = MetricsStore(run_name='e2e_test')
controller = GAController(ctrl_cfg, store)
final_pop = controller.run()
store.to_json('results/e2e_test.json')

assert len(store.generations) == 2, f'Expected 2 generations, got {len(store.generations)}'
assert store.generations[-1].best_fitness > 0.0, 'Best fitness should be > 0'
print()
print('═══════════════════════════')
print('  E2E TEST PASSED')
print(f'  Final best: {store.final_best_fitness:.4f}')
print('═══════════════════════════')
"

echo "End-to-end test complete!"
