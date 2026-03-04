#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "epoch.pb.h"

using namespace std;

namespace epoch {

enum class JobState {
    kPending,
    kDispatched,
    kCompleted,
    kFailed,
};

/// A single hyperparameter training job.
struct Job {
    string job_id;
    int32_t generation_id = 0;
    HyperparamConfig config;
    JobState state = JobState::kPending;
    string assigned_worker_id;
    TrainingResult result;
    chrono::steady_clock::time_point enqueued_at;
    chrono::steady_clock::time_point dequeued_at;
    chrono::steady_clock::time_point dispatched_at;
};

}  // namespace epoch
