
#pragma once
#include <chrono>
#include <mutex>
#include <string>

#include <grpcpp/grpcpp.h>

#include "epoch.grpc.pb.h"
#include "epoch/job.h"

using namespace std;

namespace epoch {

enum class WorkerState {
    kIdle,
    kBusy,
    kDead,
};

/// Represents a single connected worker and its bidirectional gRPC stream.
///
/// Each WorkerSession is created when a worker sends a Registration message
/// on the Connect stream. The scheduler writes JobAssignments to the stream
/// and reads TrainingResults back.
class WorkerSession {
public:
    WorkerSession(const string& worker_id,
                  grpc::ServerReaderWriter<SchedulerMessage, WorkerMessage>* stream);

    const string& WorkerId() const { return worker_id_; }
    WorkerState State() const;
    void SetState(WorkerState state);

    /// Send a job assignment to this worker. Returns false if the write fails.
    bool AssignJob(
        const Job& job,
        chrono::steady_clock::time_point* dispatch_completed_at = nullptr,
        double* dispatch_latency_ms = nullptr);

    /// Update the last heartbeat timestamp to now.
    void TouchHeartbeat();

    /// Check if the worker has exceeded the heartbeat timeout.
    bool IsTimedOut(uint32_t timeout_ms) const;

    /// Get the ID of the currently assigned job (empty if idle).
    string AssignedJobId() const;

    /// Clear tracked assigned job after completion/failure handling.
    void ClearAssignedJob();

    /// Mark the timestamp when a result from this worker was processed.
    void MarkResultProcessed();

    /// Compute idle gap from the last processed result to a dispatch time.
    /// Returns false if no prior result timestamp is available.
    bool IdleGapMsAtDispatch(
        chrono::steady_clock::time_point dispatch_at,
        double* idle_gap_ms) const;

private:
    string worker_id_;
    grpc::ServerReaderWriter<SchedulerMessage, WorkerMessage>* stream_;
    mutable mutex mu_;
    WorkerState state_ = WorkerState::kIdle;
    string assigned_job_id_;
    chrono::steady_clock::time_point last_heartbeat_;
    bool has_result_processed_at_ = false;
    chrono::steady_clock::time_point last_result_processed_at_;
};

}  // namespace epoch
