#pragma once

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#include <grpcpp/grpcpp.h>

#include "epoch.grpc.pb.h"
#include "epoch/config.h"
#include "epoch/job_queue.h"
#include "epoch/metrics_collector.h"
#include "epoch/worker_pool.h"

using namespace std;

namespace epoch {

/// gRPC service implementation for both WorkerService and SchedulerControl.
///
/// Runs a dispatch loop on a dedicated thread that matches pending jobs to
/// idle workers, and a heartbeat monitor thread that detects dead workers.
class EpochServiceImpl final : public WorkerService::Service,
                               public SchedulerControl::Service {
public:
    explicit EpochServiceImpl(const Config& config);
    ~EpochServiceImpl();

    /// Start the dispatch and heartbeat threads.
    void Start();

    /// Signal all threads to stop and join them.
    void Shutdown();

    // ─── WorkerService RPCs ──────────────────────────────────────────────

    grpc::Status Connect(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<SchedulerMessage, WorkerMessage>* stream) override;

    // ─── SchedulerControl RPCs ───────────────────────────────────────────

    grpc::Status SubmitGeneration(
        grpc::ServerContext* context,
        const SubmitGenerationRequest* request,
        SubmitGenerationResponse* response) override;

    grpc::Status GetGenerationResults(
        grpc::ServerContext* context,
        const GetResultsRequest* request,
        GetResultsResponse* response) override;

    grpc::Status GetSchedulerStatus(
        grpc::ServerContext* context,
        const SchedulerStatusRequest* request,
        SchedulerStatusResponse* response) override;

private:
    void DispatchLoop();
    void HeartbeatLoop();
    void LogMetricEvent(const string& json_line);

    Config config_;
    JobQueue job_queue_;
    WorkerPool worker_pool_;
    MetricsCollector metrics_;

    atomic<bool> running_{false};
    thread dispatch_thread_;
    thread heartbeat_thread_;
    mutable mutex metrics_log_mu_;
    ofstream metrics_log_stream_;
};

}  // namespace epoch
