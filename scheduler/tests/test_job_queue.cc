#include "epoch/job_queue.h"

#include <gtest/gtest.h>

using namespace std;

namespace epoch {
namespace {

HyperparamConfig MakeConfig(double lr) {
    HyperparamConfig config;
    config.set_learning_rate(lr);
    config.set_batch_size(32);
    config.set_optimizer(OPTIMIZER_ADAM);
    config.set_epochs(5);
    return config;
}

TEST(JobQueueTest, EnqueueBatchReturnsCount) {
    JobQueue queue;
    vector<HyperparamConfig> configs = {MakeConfig(0.01), MakeConfig(0.001)};
    EXPECT_EQ(queue.EnqueueBatch(0, configs), 2);
}

TEST(JobQueueTest, PendingCountReflectsEnqueued) {
    JobQueue queue;
    vector<HyperparamConfig> configs = {MakeConfig(0.01), MakeConfig(0.001), MakeConfig(0.1)};
    queue.EnqueueBatch(0, configs);
    EXPECT_EQ(queue.PendingCount(), 3u);
}

TEST(JobQueueTest, TryDequeueReturnsPending) {
    JobQueue queue;
    queue.EnqueueBatch(0, {MakeConfig(0.01)});
    auto job = queue.TryDequeue();
    ASSERT_TRUE(job.has_value());
    EXPECT_EQ(job->generation_id, 0);
    EXPECT_DOUBLE_EQ(job->config.learning_rate(), 0.01);
    EXPECT_EQ(queue.PendingCount(), 0u);
}

TEST(JobQueueTest, TryDequeueReturnsNulloptWhenEmpty) {
    JobQueue queue;
    auto job = queue.TryDequeue();
    EXPECT_FALSE(job.has_value());
}

TEST(JobQueueTest, GenerationCompleteAfterAllMarked) {
    JobQueue queue;
    queue.EnqueueBatch(0, {MakeConfig(0.01), MakeConfig(0.001)});

    auto job1 = queue.TryDequeue();
    auto job2 = queue.TryDequeue();
    ASSERT_TRUE(job1.has_value());
    ASSERT_TRUE(job2.has_value());

    EXPECT_FALSE(queue.IsGenerationComplete(0));

    TrainingResult result;
    result.set_validation_accuracy(0.95);
    result.set_status(JOB_STATUS_COMPLETED);

    EXPECT_TRUE(queue.MarkCompleted(job1->job_id, result));
    EXPECT_FALSE(queue.IsGenerationComplete(0));

    EXPECT_TRUE(queue.MarkCompleted(job2->job_id, result));
    EXPECT_TRUE(queue.IsGenerationComplete(0));
}

TEST(JobQueueTest, FailedJobsCountAsComplete) {
    JobQueue queue;
    queue.EnqueueBatch(0, {MakeConfig(0.01)});
    auto job = queue.TryDequeue();
    ASSERT_TRUE(job.has_value());

    EXPECT_TRUE(queue.MarkFailed(job->job_id, "test error"));
    EXPECT_TRUE(queue.IsGenerationComplete(0));
}

TEST(JobQueueTest, GetGenerationResultsReturnsAllResults) {
    JobQueue queue;
    queue.EnqueueBatch(0, {MakeConfig(0.01), MakeConfig(0.001)});

    auto job1 = queue.TryDequeue();
    auto job2 = queue.TryDequeue();

    TrainingResult r1, r2;
    r1.set_validation_accuracy(0.90);
    r1.set_status(JOB_STATUS_COMPLETED);
    r2.set_validation_accuracy(0.95);
    r2.set_status(JOB_STATUS_COMPLETED);

    EXPECT_TRUE(queue.MarkCompleted(job1->job_id, r1));
    EXPECT_TRUE(queue.MarkCompleted(job2->job_id, r2));

    auto results = queue.GetGenerationResults(0);
    EXPECT_EQ(results.size(), 2u);
}

TEST(JobQueueTest, MultipleGenerationsIndependent) {
    JobQueue queue;
    queue.EnqueueBatch(0, {MakeConfig(0.01)});
    queue.EnqueueBatch(1, {MakeConfig(0.001)});

    EXPECT_EQ(queue.GenerationSize(0), 1u);
    EXPECT_EQ(queue.GenerationSize(1), 1u);

    auto job = queue.TryDequeue();
    TrainingResult r;
    r.set_status(JOB_STATUS_COMPLETED);
    EXPECT_TRUE(queue.MarkCompleted(job->job_id, r));

    EXPECT_TRUE(queue.IsGenerationComplete(0));
    EXPECT_FALSE(queue.IsGenerationComplete(1));
}

TEST(JobQueueTest, DuplicateTerminalTransitionsAreIgnored) {
    JobQueue queue;
    queue.EnqueueBatch(0, {MakeConfig(0.01)});
    auto job = queue.TryDequeue();
    ASSERT_TRUE(job.has_value());

    TrainingResult completed;
    completed.set_status(JOB_STATUS_COMPLETED);
    completed.set_validation_accuracy(0.9);

    EXPECT_TRUE(queue.MarkFailed(job->job_id, "first failure"));
    EXPECT_FALSE(queue.MarkCompleted(job->job_id, completed));
    EXPECT_FALSE(queue.MarkFailed(job->job_id, "duplicate failure"));
}

TEST(JobQueueTest, EstimatedCostDispatchPrioritizesHeavierJobs) {
    JobQueue queue;
    queue.SetDispatchStrategy(DispatchStrategy::kEstimatedCost);

    HyperparamConfig fast = MakeConfig(0.01);
    fast.set_epochs(1);
    fast.set_batch_size(256);
    fast.clear_conv_filters();
    fast.add_conv_filters(8);
    fast.add_conv_filters(16);
    fast.add_conv_filters(16);
    fast.clear_dense_units();
    fast.add_dense_units(32);

    HyperparamConfig slow = MakeConfig(0.01);
    slow.set_epochs(2);
    slow.set_batch_size(32);
    slow.clear_conv_filters();
    slow.add_conv_filters(32);
    slow.add_conv_filters(64);
    slow.add_conv_filters(64);
    slow.clear_dense_units();
    slow.add_dense_units(128);

    queue.EnqueueBatch(0, {fast, slow});

    auto first = queue.TryDequeue();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->config.batch_size(), 32);
    EXPECT_EQ(first->config.epochs(), 2);

    auto second = queue.TryDequeue();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->config.batch_size(), 256);
    EXPECT_EQ(second->config.epochs(), 1);
}

TEST(JobQueueTest, FifoDispatchPreservesInsertionOrder) {
    JobQueue queue;
    queue.SetDispatchStrategy(DispatchStrategy::kFifo);

    HyperparamConfig first_cfg = MakeConfig(0.01);
    first_cfg.set_batch_size(256);

    HyperparamConfig second_cfg = MakeConfig(0.01);
    second_cfg.set_batch_size(16);
    second_cfg.set_epochs(3);

    queue.EnqueueBatch(0, {first_cfg, second_cfg});

    auto first = queue.TryDequeue();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->config.batch_size(), 256);

    auto second = queue.TryDequeue();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->config.batch_size(), 16);
}

}  // namespace
}  // namespace epoch
