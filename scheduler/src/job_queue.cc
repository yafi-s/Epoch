
#include "epoch/job_queue.h"

#include <algorithm>
#include <sstream>

using namespace std;

namespace epoch {

double JobQueue::EstimateJobCost(const HyperparamConfig& config) {
    const int epochs = max(1, config.epochs());
    const int batch_size = max(1, config.batch_size());

    int conv_sum = 0;
    if (config.conv_filters_size() > 0) {
        for (const int filters : config.conv_filters()) {
            conv_sum += max(0, filters);
        }
    } else {
        conv_sum = 32 + 64 + 128;
    }

    int dense_units = 128;
    if (config.dense_units_size() > 0) {
        dense_units = max(0, config.dense_units(0));
    }

    const double model_scale = static_cast<double>(conv_sum) + (static_cast<double>(dense_units) / 4.0);
    const double batch_factor = 128.0 / static_cast<double>(batch_size);
    return static_cast<double>(epochs) * model_scale * batch_factor;
}

int JobQueue::EnqueueBatch(int32_t generation_id,
                           const vector<HyperparamConfig>& configs) {
    lock_guard<mutex> lock(mu_);

    auto& gen_jobs = generations_[generation_id];
    int count = 0;
    const auto now = chrono::steady_clock::now();

    for (const auto& config : configs) {
        ostringstream oss;
        oss << "gen" << generation_id << "_job" << next_job_seq_++;
        string job_id = oss.str();

        Job job;
        job.job_id = job_id;
        job.generation_id = generation_id;
        job.config = config;
        job.state = JobState::kPending;
        job.enqueued_at = now;

        jobs_[job_id] = move(job);
        pending_.push_back(job_id);
        gen_jobs.push_back(job_id);
        ++count;
    }

    cv_.notify_all();
    return count;
}

optional<Job> JobQueue::TryDequeue() {
    lock_guard<mutex> lock(mu_);

    if (pending_.empty()) {
        return nullopt;
    }

    size_t index = 0;
    if (dispatch_strategy_ == DispatchStrategy::kEstimatedCost && pending_.size() > 1) {
        double best_cost = -1.0;
        for (size_t i = 0; i < pending_.size(); ++i) {
            auto it = jobs_.find(pending_[i]);
            if (it == jobs_.end()) {
                continue;
            }
            const double cost = EstimateJobCost(it->second.config);
            if (cost > best_cost) {
                best_cost = cost;
                index = i;
            }
        }
    }

    const string job_id = pending_[index];
    pending_.erase(pending_.begin() + index);

    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return nullopt;
    }

    it->second.state = JobState::kDispatched;
    it->second.dequeued_at = chrono::steady_clock::now();
    return it->second;
}

void JobQueue::SetDispatchStrategy(DispatchStrategy strategy) {
    lock_guard<mutex> lock(mu_);
    dispatch_strategy_ = strategy;
}

bool JobQueue::MarkCompleted(const string& job_id, const TrainingResult& result) {
    lock_guard<mutex> lock(mu_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return false;
    }
    if (it->second.state == JobState::kCompleted || it->second.state == JobState::kFailed) {
        return false;
    }
    it->second.state = JobState::kCompleted;
    it->second.result = result;
    it->second.result.set_job_id(job_id);
    it->second.result.set_generation_id(it->second.generation_id);
    it->second.result.set_status(JOB_STATUS_COMPLETED);
    cv_.notify_all();
    return true;
}

bool JobQueue::MarkFailed(const string& job_id, const string& error) {
    lock_guard<mutex> lock(mu_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return false;
    }
    if (it->second.state == JobState::kCompleted || it->second.state == JobState::kFailed) {
        return false;
    }
    it->second.state = JobState::kFailed;
    it->second.result.set_job_id(job_id);
    it->second.result.set_generation_id(it->second.generation_id);
    it->second.result.set_validation_accuracy(0.0);
    it->second.result.set_training_loss(0.0);
    it->second.result.set_status(JOB_STATUS_FAILED);
    it->second.result.set_error_message(error);
    cv_.notify_all();
    return true;
}

bool JobQueue::IsGenerationComplete(int32_t generation_id) const {
    lock_guard<mutex> lock(mu_);
    auto gen_it = generations_.find(generation_id);
    if (gen_it == generations_.end()) {
        return false;
    }

    for (const auto& job_id : gen_it->second) {
        auto job_it = jobs_.find(job_id);
        if (job_it == jobs_.end()) continue;
        if (job_it->second.state != JobState::kCompleted &&
            job_it->second.state != JobState::kFailed) {
            return false;
        }
    }
    return true;
}

vector<TrainingResult> JobQueue::GetGenerationResults(int32_t generation_id) const {
    lock_guard<mutex> lock(mu_);
    vector<TrainingResult> results;

    auto gen_it = generations_.find(generation_id);
    if (gen_it == generations_.end()) {
        return results;
    }

    for (const auto& job_id : gen_it->second) {
        auto job_it = jobs_.find(job_id);
        if (job_it != jobs_.end()) {
            results.push_back(job_it->second.result);
        }
    }
    return results;
}

size_t JobQueue::PendingCount() const {
    lock_guard<mutex> lock(mu_);
    return pending_.size();
}

size_t JobQueue::GenerationSize(int32_t generation_id) const {
    lock_guard<mutex> lock(mu_);
    auto it = generations_.find(generation_id);
    return it != generations_.end() ? it->second.size() : 0;
}

}  // namespace epoch
