#pragma once

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace std;

namespace epoch {

/// Collects scheduler-side performance metrics.
///
/// All counters are atomic for lock-free reads from monitoring endpoints.
/// Per-generation wall clock times are protected by a mutex.
class MetricsCollector {
public:
    struct MetricSummary {
        double p50_ms = 0.0;
        double p90_ms = 0.0;
        double max_ms = 0.0;
        uint32_t samples = 0;
    };

    struct GenerationRuntimeMetrics {
        MetricSummary dispatch_latency;
        MetricSummary worker_idle_gap;
        MetricSummary queue_wait;
        double queue_wait_min_ms = 0.0;
    };

    void RecordJobDispatched();
    void RecordJobCompleted();
    void RecordJobFailed();
    void RecordDispatchLatency(int32_t generation_id, double latency_ms);
    void RecordWorkerIdleGap(int32_t generation_id, double idle_gap_ms);
    void RecordQueueWait(int32_t generation_id, double queue_wait_ms);

    /// Mark the start of a generation's evaluation.
    void StartGeneration(int32_t generation_id);

    /// Mark the end of a generation's evaluation. Returns wall-clock ms.
    int64_t EndGeneration(int32_t generation_id);

    /// Get wall-clock ms for a completed generation (0 if not found/not finished).
    int64_t GetGenerationWallClockMs(int32_t generation_id) const;
    GenerationRuntimeMetrics GetGenerationRuntimeMetrics(int32_t generation_id) const;

    uint64_t TotalDispatched() const { return dispatched_.load(); }
    uint64_t TotalCompleted() const { return completed_.load(); }
    uint64_t TotalFailed() const { return failed_.load(); }

private:
    atomic<uint64_t> dispatched_{0};
    atomic<uint64_t> completed_{0};
    atomic<uint64_t> failed_{0};

    mutable mutex gen_mu_;
    unordered_map<int32_t, chrono::steady_clock::time_point> gen_start_;
    unordered_map<int32_t, int64_t> gen_wall_clock_ms_;
    unordered_map<int32_t, vector<double>> gen_dispatch_latency_ms_;
    unordered_map<int32_t, vector<double>> gen_worker_idle_gap_ms_;
    unordered_map<int32_t, vector<double>> gen_queue_wait_ms_;

    static MetricSummary Summarize(const vector<double>& samples);
};

}  // namespace epoch
