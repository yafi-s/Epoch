#include "epoch/worker_session.h"

#include <iostream>

using namespace std;

namespace epoch {

WorkerSession::WorkerSession(
    const string& worker_id,
    grpc::ServerReaderWriter<SchedulerMessage, WorkerMessage>* stream)
    : worker_id_(worker_id),
      stream_(stream),
      last_heartbeat_(chrono::steady_clock::now()) {}

WorkerState WorkerSession::State() const {
    lock_guard<mutex> lock(mu_);
    return state_;
}

void WorkerSession::SetState(WorkerState state) {
    lock_guard<mutex> lock(mu_);
    state_ = state;
}

bool WorkerSession::AssignJob(
    const Job& job,
    chrono::steady_clock::time_point* dispatch_completed_at,
    double* dispatch_latency_ms) {
    lock_guard<mutex> lock(mu_);

    SchedulerMessage msg;
    auto* assignment = msg.mutable_job_assignment();
    assignment->set_job_id(job.job_id);
    assignment->set_generation_id(job.generation_id);
    *assignment->mutable_config() = job.config;

    const auto write_start = chrono::steady_clock::now();
    if (!stream_->Write(msg)) {
        cerr << "[WorkerSession] Failed to write job " << job.job_id
                  << " to worker " << worker_id_ << endl;
        return false;
    }
    const auto write_end = chrono::steady_clock::now();
    if (dispatch_completed_at != nullptr) {
        *dispatch_completed_at = write_end;
    }
    if (dispatch_latency_ms != nullptr) {
        *dispatch_latency_ms = chrono::duration<double, milli>(write_end - write_start).count();
    }

    state_ = WorkerState::kBusy;
    assigned_job_id_ = job.job_id;
    return true;
}

void WorkerSession::TouchHeartbeat() {
    lock_guard<mutex> lock(mu_);
    last_heartbeat_ = chrono::steady_clock::now();
}

bool WorkerSession::IsTimedOut(uint32_t timeout_ms) const {
    lock_guard<mutex> lock(mu_);
    auto now = chrono::steady_clock::now();
    auto elapsed =
        chrono::duration_cast<chrono::milliseconds>(now - last_heartbeat_).count();
    return elapsed > static_cast<int64_t>(timeout_ms);
}

string WorkerSession::AssignedJobId() const {
    lock_guard<mutex> lock(mu_);
    return assigned_job_id_;
}

void WorkerSession::ClearAssignedJob() {
    lock_guard<mutex> lock(mu_);
    assigned_job_id_.clear();
}

void WorkerSession::MarkResultProcessed() {
    lock_guard<mutex> lock(mu_);
    last_result_processed_at_ = chrono::steady_clock::now();
    has_result_processed_at_ = true;
}

bool WorkerSession::IdleGapMsAtDispatch(
    chrono::steady_clock::time_point dispatch_at,
    double* idle_gap_ms) const {
    lock_guard<mutex> lock(mu_);
    if (!has_result_processed_at_) {
        return false;
    }
    if (idle_gap_ms != nullptr) {
        *idle_gap_ms = chrono::duration<double, milli>(dispatch_at - last_result_processed_at_)
                           .count();
    }
    return true;
}

}  // namespace epoch
