#include "epoch/service_impl.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

using namespace std;

namespace epoch {

EpochServiceImpl::EpochServiceImpl(const Config& config) : config_(config) {
    if (!config_.metrics_log_path.empty()) {
        metrics_log_stream_.open(config_.metrics_log_path, ios::out | ios::trunc);
        if (!metrics_log_stream_) {
            cerr << "[Scheduler] Failed to open metrics log at " << config_.metrics_log_path
                 << endl;
        } else {
            cout << "[Scheduler] Metrics log enabled: " << config_.metrics_log_path << endl;
        }
    }
}

EpochServiceImpl::~EpochServiceImpl() {
    Shutdown();
}

void EpochServiceImpl::Start() {
    running_.store(true);
    dispatch_thread_ = thread(&EpochServiceImpl::DispatchLoop, this);
    heartbeat_thread_ = thread(&EpochServiceImpl::HeartbeatLoop, this);
    cout << "[Scheduler] Dispatch and heartbeat threads started" << endl;
}

void EpochServiceImpl::Shutdown() {
    running_.store(false);
    if (dispatch_thread_.joinable()) dispatch_thread_.join();
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
    if (metrics_log_stream_.is_open()) {
        metrics_log_stream_.flush();
        metrics_log_stream_.close();
    }
    cout << "[Scheduler] Shutdown complete" << endl;
}

grpc::Status EpochServiceImpl::Connect(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<SchedulerMessage, WorkerMessage>* stream) {
    WorkerMessage msg;
    if (!stream->Read(&msg) || !msg.has_registration()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "First message must be a registration");
    }

    const auto& reg = msg.registration();
    if (!config_.worker_auth_key.empty() && reg.auth_key() != config_.worker_auth_key) {
        cerr << "[Scheduler] Rejected worker " << reg.worker_id()
             << " due to invalid auth key" << endl;
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Invalid worker auth key");
    }

    auto session = make_shared<WorkerSession>(reg.worker_id(), stream);

    if (!worker_pool_.RegisterWorker(session)) {
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                            "Worker already registered: " + reg.worker_id());
    }

    cout << "[Scheduler] Worker " << reg.worker_id() << " connected" << endl;

    while (stream->Read(&msg)) {
        if (msg.has_heartbeat()) {
            session->TouchHeartbeat();
        } else if (msg.has_result()) {
            const auto& result = msg.result();
            cout << "[Scheduler] Result from " << result.worker_id()
                 << " for job " << result.job_id()
                 << ": acc=" << result.validation_accuracy() << endl;

            if (result.status() == JOB_STATUS_COMPLETED) {
                if (job_queue_.MarkCompleted(result.job_id(), result)) {
                    metrics_.RecordJobCompleted();
                } else {
                    cerr << "[Scheduler] Ignoring late/duplicate completion for job "
                         << result.job_id() << endl;
                }
            } else {
                if (job_queue_.MarkFailed(result.job_id(), result.error_message())) {
                    metrics_.RecordJobFailed();
                } else {
                    cerr << "[Scheduler] Ignoring late/duplicate failure for job "
                         << result.job_id() << endl;
                }
            }

            session->MarkResultProcessed();
            worker_pool_.MarkWorkerIdle(result.worker_id());

            {
                ostringstream json;
                json << "{\"event\":\"result\",\"worker_id\":\"" << result.worker_id()
                     << "\",\"job_id\":\"" << result.job_id()
                     << "\",\"generation_id\":" << result.generation_id()
                     << ",\"training_time_ms\":" << result.training_time_ms()
                     << ",\"status\":" << result.status() << "}";
                LogMetricEvent(json.str());
            }

            if (job_queue_.IsGenerationComplete(result.generation_id())) {
                metrics_.EndGeneration(result.generation_id());
                cout << "[Scheduler] Generation " << result.generation_id()
                     << " complete (wall_clock="
                     << metrics_.GetGenerationWallClockMs(result.generation_id())
                     << "ms)" << endl;
            }
        }
    }

    cout << "[Scheduler] Worker " << reg.worker_id() << " disconnected" << endl;

    string assigned = session->AssignedJobId();
    if (!assigned.empty() && session->State() == WorkerState::kBusy) {
        if (job_queue_.MarkFailed(assigned, "Worker disconnected")) {
            metrics_.RecordJobFailed();
        }
    }

    worker_pool_.RemoveWorker(reg.worker_id());
    return grpc::Status::OK;
}

grpc::Status EpochServiceImpl::SubmitGeneration(
    grpc::ServerContext* context,
    const SubmitGenerationRequest* request,
    SubmitGenerationResponse* response) {
    vector<HyperparamConfig> configs(request->configs().begin(), request->configs().end());

    int count = job_queue_.EnqueueBatch(request->generation_id(), configs);
    metrics_.StartGeneration(request->generation_id());

    cout << "[Scheduler] Enqueued " << count << " jobs for generation "
         << request->generation_id() << endl;

    response->set_accepted(true);
    response->set_num_jobs(count);
    return grpc::Status::OK;
}

grpc::Status EpochServiceImpl::GetGenerationResults(
    grpc::ServerContext* context,
    const GetResultsRequest* request,
    GetResultsResponse* response) {
    bool complete = job_queue_.IsGenerationComplete(request->generation_id());
    response->set_complete(complete);

    if (complete) {
        auto results = job_queue_.GetGenerationResults(request->generation_id());
        for (auto& r : results) {
            *response->add_results() = move(r);
        }
        response->set_wall_clock_ms(metrics_.GetGenerationWallClockMs(request->generation_id()));

        auto runtime_metrics = metrics_.GetGenerationRuntimeMetrics(request->generation_id());
        auto* proto_metrics = response->mutable_runtime_metrics();
        proto_metrics->set_dispatch_latency_p50_ms(runtime_metrics.dispatch_latency.p50_ms);
        proto_metrics->set_dispatch_latency_p90_ms(runtime_metrics.dispatch_latency.p90_ms);
        proto_metrics->set_dispatch_latency_max_ms(runtime_metrics.dispatch_latency.max_ms);
        proto_metrics->set_worker_idle_gap_p50_ms(runtime_metrics.worker_idle_gap.p50_ms);
        proto_metrics->set_worker_idle_gap_p90_ms(runtime_metrics.worker_idle_gap.p90_ms);
        proto_metrics->set_worker_idle_gap_max_ms(runtime_metrics.worker_idle_gap.max_ms);
        proto_metrics->set_queue_wait_p50_ms(runtime_metrics.queue_wait.p50_ms);
        proto_metrics->set_queue_wait_p90_ms(runtime_metrics.queue_wait.p90_ms);
        proto_metrics->set_queue_wait_max_ms(runtime_metrics.queue_wait.max_ms);
        proto_metrics->set_dispatch_samples(runtime_metrics.dispatch_latency.samples);
        proto_metrics->set_idle_gap_samples(runtime_metrics.worker_idle_gap.samples);
        proto_metrics->set_queue_wait_samples(runtime_metrics.queue_wait.samples);
    }

    return grpc::Status::OK;
}

void EpochServiceImpl::DispatchLoop() {
    while (running_.load()) {
        while (true) {
            auto idle_worker = worker_pool_.GetIdleWorker();
            if (!idle_worker) break;

            auto maybe_job = job_queue_.TryDequeue();
            if (!maybe_job.has_value()) break;

            auto job = maybe_job.value();
            chrono::steady_clock::time_point dispatch_completed_at;
            double dispatch_latency_ms = 0.0;

            if (idle_worker->AssignJob(
                    job,
                    &dispatch_completed_at,
                    &dispatch_latency_ms)) {
                job.dispatched_at = dispatch_completed_at;
                worker_pool_.MarkWorkerBusy(idle_worker->WorkerId());
                metrics_.RecordJobDispatched();

                metrics_.RecordDispatchLatency(job.generation_id, dispatch_latency_ms);

                const double queue_wait_ms = chrono::duration<double, milli>(
                    dispatch_completed_at - job.enqueued_at).count();
                metrics_.RecordQueueWait(job.generation_id, queue_wait_ms);

                double idle_gap_ms = 0.0;
                const bool has_idle_gap =
                    idle_worker->IdleGapMsAtDispatch(dispatch_completed_at, &idle_gap_ms);
                if (has_idle_gap) {
                    metrics_.RecordWorkerIdleGap(job.generation_id, idle_gap_ms);
                }

                cout << "[Dispatch] Job " << job.job_id << " -> worker "
                     << idle_worker->WorkerId() << endl;

                ostringstream json;
                json << "{\"event\":\"dispatch\",\"worker_id\":\"" << idle_worker->WorkerId()
                     << "\",\"job_id\":\"" << job.job_id
                     << "\",\"generation_id\":" << job.generation_id
                     << ",\"dispatch_latency_ms\":" << dispatch_latency_ms
                     << ",\"queue_wait_ms\":" << queue_wait_ms;
                if (has_idle_gap) {
                    json << ",\"worker_idle_gap_ms\":" << idle_gap_ms;
                }
                json << "}";
                LogMetricEvent(json.str());
            } else {
                if (job_queue_.MarkFailed(job.job_id, "Failed to dispatch")) {
                    metrics_.RecordJobFailed();
                }
                worker_pool_.MarkWorkerDead(idle_worker->WorkerId());
            }
        }

        this_thread::sleep_for(chrono::milliseconds(config_.dispatch_interval_ms));
    }
}

void EpochServiceImpl::HeartbeatLoop() {
    while (running_.load()) {
        auto timed_out = worker_pool_.CheckHeartbeats(config_.heartbeat_timeout_ms);
        for (const auto& worker_id : timed_out) {
            const string assigned_job_id = worker_pool_.GetAssignedJobId(worker_id);
            if (!assigned_job_id.empty()) {
                if (job_queue_.MarkFailed(assigned_job_id, "Worker heartbeat timeout")) {
                    metrics_.RecordJobFailed();
                    cerr << "[Heartbeat] Marked in-flight job " << assigned_job_id
                         << " failed for timed-out worker " << worker_id << endl;
                }
            }
            cerr << "[Heartbeat] Worker " << worker_id << " timed out" << endl;
        }

        this_thread::sleep_for(chrono::milliseconds(config_.heartbeat_check_interval_ms));
    }
}

void EpochServiceImpl::LogMetricEvent(const string& json_line) {
    if (!metrics_log_stream_.is_open()) {
        return;
    }
    lock_guard<mutex> lock(metrics_log_mu_);
    metrics_log_stream_ << json_line << '\n';
    metrics_log_stream_.flush();
}

}  // namespace epoch
