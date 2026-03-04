#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RESULTS_DIR="results/scalability"
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════╗"
echo "║      Epoch Scalability Test          ║"
echo "╚══════════════════════════════════════╝"

WORKER_COUNTS=(1 2 4 8)

for NUM_WORKERS in "${WORKER_COUNTS[@]}"; do
    echo ""
    echo "═══ Testing with $NUM_WORKERS worker(s) ═══"

    # Start scheduler
    ./build/scheduler/epoch_scheduler --listen-address 0.0.0.0:50051 &
    SCHEDULER_PID=$!
    sleep 2

    # Start N workers
    WORKER_PIDS=()
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        python -m worker.main --scheduler-address localhost:50051 --worker-id "w${i}" --log-level WARNING &
        WORKER_PIDS+=($!)
    done
    sleep 3

    # Run GA with fixed config
    python -c "
import sys
sys.path.insert(0, '.')
from ga.controller import GAController, ControllerConfig
from ga.engine import GAConfig
from metrics.collector import MetricsStore

ga_cfg = GAConfig(population_size=8, num_generations=3, dataset='mnist', epochs=1, seed=42)
ctrl_cfg = ControllerConfig(scheduler_address='localhost:50051', ga_config=ga_cfg)
store = MetricsStore(run_name='scale_${NUM_WORKERS}w')
controller = GAController(ctrl_cfg, store)
controller.run()
store.to_json('${RESULTS_DIR}/scale_${NUM_WORKERS}w.json')
print(f'  Workers: ${NUM_WORKERS}  Time: {store.total_wall_clock_ms}ms  Best: {store.final_best_fitness:.4f}')
"

    # Cleanup
    for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    kill "$SCHEDULER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    sleep 2
done

# Generate speedup plot
echo ""
echo "Generating speedup plot..."
python -c "
import sys
sys.path.insert(0, '.')
from metrics.collector import MetricsStore
from metrics.dashboard import plot_speedup

stores = {}
for n in [1, 2, 4, 8]:
    try:
        stores[n] = MetricsStore.from_json(f'${RESULTS_DIR}/scale_{n}w.json')
    except FileNotFoundError:
        pass

if stores:
    plot_speedup(stores, '${RESULTS_DIR}/speedup.png')
    print('Speedup plot saved to ${RESULTS_DIR}/speedup.png')

    # Print summary
    baseline = stores[min(stores.keys())].total_wall_clock_ms
    print()
    print(f'{\"Workers\":<10} {\"Time (ms)\":<12} {\"Speedup\":<10}')
    print('-' * 32)
    for n, s in sorted(stores.items()):
        speedup = baseline / s.total_wall_clock_ms if s.total_wall_clock_ms > 0 else 0
        print(f'{n:<10} {s.total_wall_clock_ms:<12} {speedup:<10.2f}')
"

echo ""
echo "Scalability test complete! Results in $RESULTS_DIR"
