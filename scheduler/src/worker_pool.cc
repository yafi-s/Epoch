#include "epoch/worker_pool.h"

#include <iostream>

using namespace std;

namespace epoch {

bool WorkerPool::RegisterWorker(shared_ptr<WorkerSession> session) {
    lock_guard<mutex> lock(mu_);
    const auto& id = session->WorkerId();
    if (workers_.count(id)) {
        cerr << "[WorkerPool] Worker " << id << " already registered" << endl;
        return false;
    }
    workers_[id] = move(session);
    cout << "[WorkerPool] Registered worker " << id
              << " (total: " << workers_.size() << ")" << endl;
    return true;
}

void WorkerPool::RemoveWorker(const string& worker_id) {
    lock_guard<mutex> lock(mu_);
    workers_.erase(worker_id);
    cout << "[WorkerPool] Removed worker " << worker_id
              << " (total: " << workers_.size() << ")" << endl;
}

shared_ptr<WorkerSession> WorkerPool::GetIdleWorker() {
    lock_guard<mutex> lock(mu_);
    for (auto& [id, session] : workers_) {
        if (session->State() == WorkerState::kIdle) {
            return session;
        }
    }
    return nullptr;
}

void WorkerPool::MarkWorkerIdle(const string& worker_id) {
    lock_guard<mutex> lock(mu_);
    auto it = workers_.find(worker_id);
    if (it != workers_.end()) {
        it->second->ClearAssignedJob();
        it->second->SetState(WorkerState::kIdle);
    }
}

void WorkerPool::MarkWorkerBusy(const string& worker_id) {
    lock_guard<mutex> lock(mu_);
    auto it = workers_.find(worker_id);
    if (it != workers_.end()) {
        it->second->SetState(WorkerState::kBusy);
    }
}

void WorkerPool::MarkWorkerDead(const string& worker_id) {
    lock_guard<mutex> lock(mu_);
    auto it = workers_.find(worker_id);
    if (it != workers_.end()) {
        it->second->SetState(WorkerState::kDead);
    }
}

vector<string> WorkerPool::CheckHeartbeats(uint32_t timeout_ms) {
    lock_guard<mutex> lock(mu_);
    vector<string> timed_out;
    for (auto& [id, session] : workers_) {
        if (session->State() != WorkerState::kDead && session->IsTimedOut(timeout_ms)) {
            session->SetState(WorkerState::kDead);
            timed_out.push_back(id);
            cerr << "[WorkerPool] Worker " << id << " timed out" << endl;
        }
    }
    return timed_out;
}

string WorkerPool::GetAssignedJobId(const string& worker_id) const {
    lock_guard<mutex> lock(mu_);
    auto it = workers_.find(worker_id);
    if (it == workers_.end()) {
        return "";
    }
    return it->second->AssignedJobId();
}

size_t WorkerPool::Size() const {
    lock_guard<mutex> lock(mu_);
    return workers_.size();
}

size_t WorkerPool::IdleCount() const {
    lock_guard<mutex> lock(mu_);
    size_t count = 0;
    for (const auto& [id, session] : workers_) {
        if (session->State() == WorkerState::kIdle) {
            ++count;
        }
    }
    return count;
}

size_t WorkerPool::BusyCount() const {
    lock_guard<mutex> lock(mu_);
    size_t count = 0;
    for (const auto& [id, session] : workers_) {
        if (session->State() == WorkerState::kBusy) {
            ++count;
        }
    }
    return count;
}

}  // namespace epoch
