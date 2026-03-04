#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "epoch/worker_session.h"

using namespace std;

namespace epoch {

/// Manages all connected worker sessions.
///
/// Thread-safe. Used by the dispatch thread to find idle workers and by the
/// heartbeat monitor to detect dead workers.
class WorkerPool {
public:
    /// Register a new worker. Returns false if the worker_id is already registered.
    bool RegisterWorker(shared_ptr<WorkerSession> session);

    /// Remove a worker from the pool.
    void RemoveWorker(const string& worker_id);

    /// Get an idle worker session, or nullptr if none available.
    shared_ptr<WorkerSession> GetIdleWorker();

    /// Mark a worker as idle after job completion.
    void MarkWorkerIdle(const string& worker_id);

    /// Mark a worker as busy (job dispatched).
    void MarkWorkerBusy(const string& worker_id);

    /// Mark a worker as dead (heartbeat timeout or disconnect).
    void MarkWorkerDead(const string& worker_id);

    /// Check all workers for heartbeat timeouts. Returns IDs of timed-out workers.
    vector<string> CheckHeartbeats(uint32_t timeout_ms);

    /// Get currently assigned job id for a worker (empty if none/unknown).
    string GetAssignedJobId(const string& worker_id) const;

    /// Number of currently registered workers.
    size_t Size() const;

    /// Number of idle workers.
    size_t IdleCount() const;

private:
    mutable mutex mu_;
    unordered_map<string, shared_ptr<WorkerSession>> workers_;
};

}  // namespace epoch
