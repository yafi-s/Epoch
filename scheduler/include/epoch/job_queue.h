#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "epoch/config.h"
#include "epoch/job.h"

using namespace std;

namespace epoch {

/// Thread-safe job queue with per-generation tracking.
///
/// Jobs are enqueued in batches (one per generation), dequeued one at a time
/// by the dispatch thread, and marked completed/failed when results arrive.
class JobQueue {
public:
    /// Enqueue a batch of jobs for a generation. Returns the number of jobs enqueued.
    int EnqueueBatch(int32_t generation_id, const vector<HyperparamConfig>& configs);

    /// Try to dequeue a pending job. Returns nullopt if none available.
    optional<Job> TryDequeue();

    /// Set dispatch strategy for selecting the next pending job.
    void SetDispatchStrategy(DispatchStrategy strategy);

    /// Mark a job as completed with a training result.
    /// Returns true if state transitioned to completed.
    bool MarkCompleted(const string& job_id, const TrainingResult& result);

    /// Mark a job as failed.
    /// Returns true if state transitioned to failed.
    bool MarkFailed(const string& job_id, const string& error);

    /// Check whether all jobs for a generation have finished (completed or failed).
    bool IsGenerationComplete(int32_t generation_id) const;

    /// Get all results for a completed generation.
    vector<TrainingResult> GetGenerationResults(int32_t generation_id) const;

    /// Number of pending jobs.
    size_t PendingCount() const;

    /// Total jobs across all states for a generation.
    size_t GenerationSize(int32_t generation_id) const;

private:
    static double EstimateJobCost(const HyperparamConfig& config);

    mutable mutex mu_;
    condition_variable cv_;

    // All jobs indexed by job_id
    unordered_map<string, Job> jobs_;

    // Pending job IDs in FIFO order
    vector<string> pending_;

    DispatchStrategy dispatch_strategy_ = DispatchStrategy::kFifo;

    // Generation ID → list of job IDs
    unordered_map<int32_t, vector<string>> generations_;

    int next_job_seq_ = 0;
};

}  // namespace epoch
