#include "epoch/metrics_collector.h"

#include <algorithm>

using namespace std;

namespace epoch {

void MetricsCollector::RecordJobDispatched() {
    dispatched_.fetch_add(1, memory_order_relaxed);
}

void MetricsCollector::RecordJobCompleted() {
    completed_.fetch_add(1, memory_order_relaxed);
}

void MetricsCollector::RecordJobFailed() {
    failed_.fetch_add(1, memory_order_relaxed);
}

void MetricsCollector::RecordDispatchLatency(int32_t generation_id, double latency_ms) {
    lock_guard<mutex> lock(gen_mu_);
    gen_dispatch_latency_ms_[generation_id].push_back(max(0.0, latency_ms));
}

void MetricsCollector::RecordWorkerIdleGap(int32_t generation_id, double idle_gap_ms) {
    lock_guard<mutex> lock(gen_mu_);
    gen_worker_idle_gap_ms_[generation_id].push_back(max(0.0, idle_gap_ms));
}

void MetricsCollector::RecordQueueWait(int32_t generation_id, double queue_wait_ms) {
    lock_guard<mutex> lock(gen_mu_);
    gen_queue_wait_ms_[generation_id].push_back(max(0.0, queue_wait_ms));
}

void MetricsCollector::StartGeneration(int32_t generation_id) {
    lock_guard<mutex> lock(gen_mu_);
    gen_start_[generation_id] = chrono::steady_clock::now();
    gen_dispatch_latency_ms_[generation_id].clear();
    gen_worker_idle_gap_ms_[generation_id].clear();
    gen_queue_wait_ms_[generation_id].clear();
}

int64_t MetricsCollector::EndGeneration(int32_t generation_id) {
    lock_guard<mutex> lock(gen_mu_);
    auto it = gen_start_.find(generation_id);
    if (it == gen_start_.end()) {
        return 0;
    }
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(
                       chrono::steady_clock::now() - it->second)
                       .count();
    gen_wall_clock_ms_[generation_id] = elapsed;
    return elapsed;
}

int64_t MetricsCollector::GetGenerationWallClockMs(int32_t generation_id) const {
    lock_guard<mutex> lock(gen_mu_);
    auto it = gen_wall_clock_ms_.find(generation_id);
    return it != gen_wall_clock_ms_.end() ? it->second : 0;
}

MetricsCollector::GenerationRuntimeMetrics MetricsCollector::GetGenerationRuntimeMetrics(
    int32_t generation_id) const {
    lock_guard<mutex> lock(gen_mu_);
    GenerationRuntimeMetrics out;

    auto dispatch_it = gen_dispatch_latency_ms_.find(generation_id);
    if (dispatch_it != gen_dispatch_latency_ms_.end()) {
        out.dispatch_latency = Summarize(dispatch_it->second);
    }

    auto idle_it = gen_worker_idle_gap_ms_.find(generation_id);
    if (idle_it != gen_worker_idle_gap_ms_.end()) {
        out.worker_idle_gap = Summarize(idle_it->second);
    }

    auto queue_it = gen_queue_wait_ms_.find(generation_id);
    if (queue_it != gen_queue_wait_ms_.end()) {
        out.queue_wait = Summarize(queue_it->second);
    }

    return out;
}

MetricsCollector::MetricSummary MetricsCollector::Summarize(const vector<double>& samples) {
    MetricSummary out;
    out.samples = static_cast<uint32_t>(samples.size());
    if (samples.empty()) {
        return out;
    }

    vector<double> sorted = samples;
    sort(sorted.begin(), sorted.end());

    const size_t p50_idx = sorted.size() / 2;
    const size_t p90_idx = min(sorted.size() - 1, static_cast<size_t>(sorted.size() * 0.9));
    out.p50_ms = sorted[p50_idx];
    out.p90_ms = sorted[p90_idx];
    out.max_ms = sorted.back();
    return out;
}

}  // namespace epoch
