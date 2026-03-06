#include "epoch/metrics_collector.h"

#include <gtest/gtest.h>

namespace epoch {
namespace {

TEST(MetricsCollectorTest, RecordsDispatchLatencySummary) {
    MetricsCollector collector;
    collector.StartGeneration(0);

    collector.RecordDispatchLatency(0, 1.0);
    collector.RecordDispatchLatency(0, 5.0);
    collector.RecordDispatchLatency(0, 3.0);
    collector.RecordDispatchLatency(0, 9.0);
    collector.RecordDispatchLatency(0, 2.0);

    const auto metrics = collector.GetGenerationRuntimeMetrics(0);
    EXPECT_EQ(metrics.dispatch_latency.samples, 5u);
    EXPECT_DOUBLE_EQ(metrics.dispatch_latency.p50_ms, 3.0);
    EXPECT_DOUBLE_EQ(metrics.dispatch_latency.p90_ms, 9.0);
    EXPECT_DOUBLE_EQ(metrics.dispatch_latency.max_ms, 9.0);
}

TEST(MetricsCollectorTest, RecordsIdleGapSummary) {
    MetricsCollector collector;
    collector.StartGeneration(1);

    collector.RecordWorkerIdleGap(1, 4.0);
    collector.RecordWorkerIdleGap(1, 10.0);
    collector.RecordWorkerIdleGap(1, 7.0);

    const auto metrics = collector.GetGenerationRuntimeMetrics(1);
    EXPECT_EQ(metrics.worker_idle_gap.samples, 3u);
    EXPECT_DOUBLE_EQ(metrics.worker_idle_gap.p50_ms, 7.0);
    EXPECT_DOUBLE_EQ(metrics.worker_idle_gap.p90_ms, 10.0);
    EXPECT_DOUBLE_EQ(metrics.worker_idle_gap.max_ms, 10.0);
}

TEST(MetricsCollectorTest, RecordsQueueWaitSummary) {
    MetricsCollector collector;
    collector.StartGeneration(2);

    collector.RecordQueueWait(2, 15.0);
    collector.RecordQueueWait(2, 1.0);
    collector.RecordQueueWait(2, 8.0);
    collector.RecordQueueWait(2, 6.0);

    const auto metrics = collector.GetGenerationRuntimeMetrics(2);
    EXPECT_EQ(metrics.queue_wait.samples, 4u);
    EXPECT_DOUBLE_EQ(metrics.queue_wait.p50_ms, 8.0);
    EXPECT_DOUBLE_EQ(metrics.queue_wait.p90_ms, 15.0);
    EXPECT_DOUBLE_EQ(metrics.queue_wait.max_ms, 15.0);
    EXPECT_DOUBLE_EQ(metrics.queue_wait_min_ms, 1.0);
}

TEST(MetricsCollectorTest, StartGenerationClearsPriorSamples) {
    MetricsCollector collector;
    collector.StartGeneration(3);
    collector.RecordDispatchLatency(3, 11.0);
    collector.RecordWorkerIdleGap(3, 12.0);
    collector.RecordQueueWait(3, 13.0);

    collector.StartGeneration(3);
    const auto metrics = collector.GetGenerationRuntimeMetrics(3);
    EXPECT_EQ(metrics.dispatch_latency.samples, 0u);
    EXPECT_EQ(metrics.worker_idle_gap.samples, 0u);
    EXPECT_EQ(metrics.queue_wait.samples, 0u);
}

}  // namespace
}  // namespace epoch
