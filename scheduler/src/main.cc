#include <atomic>
#include <csignal>
#include <memory>
#include <thread>
#include <grpcpp/grpcpp.h>
#include <iostream>

#include "epoch/config.h"
#include "epoch/service_impl.h"

using namespace std;

namespace {
unique_ptr<grpc::Server> g_server;
atomic<bool> g_shutdown_requested{false};

void SignalHandler(int /*signal*/) {
    // Only set the flag — calling Shutdown() from a signal handler triggers
    // mutex recursion in gRPC/abseil and aborts.
    g_shutdown_requested.store(true, memory_order_release);
}
}  // namespace

int main(int argc, char* argv[]) {
    auto config = epoch::Config::FromArgs(argc, argv);

    cout << "╔══════════════════════════════════════╗\n"
              << "║          Epoch Scheduler             ║\n"
              << "╚══════════════════════════════════════╝\n"
              << "Listening on: " << config.listen_address << "\n"
              << "Dispatch interval: " << config.dispatch_interval_ms << "ms\n"
              << "Dispatch strategy: " << epoch::DispatchStrategyToString(config.dispatch_strategy)
              << "\n"
              << "Heartbeat timeout: " << config.heartbeat_timeout_ms << "ms\n"
              << "Worker auth: " << (config.worker_auth_key.empty() ? "disabled" : "enabled") << "\n"
              << endl;

    auto service = make_shared<epoch::EpochServiceImpl>(config);
    service->Start();

    grpc::ServerBuilder builder;
    builder.AddListeningPort(config.listen_address, grpc::InsecureServerCredentials());

    // Register both services (WorkerService and SchedulerControl)
    builder.RegisterService(static_cast<epoch::WorkerService::Service*>(service.get()));
    builder.RegisterService(static_cast<epoch::SchedulerControl::Service*>(service.get()));

    g_server = builder.BuildAndStart();
    if (!g_server) {
        cerr << "Failed to start server on " << config.listen_address << endl;
        return 1;
    }

    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);

    cout << "[Main] Server running. Press Ctrl+C to stop." << endl;

    // Poll for shutdown signal from a safe (non-signal-handler) context
    thread shutdown_watcher([&]() {
        while (!g_shutdown_requested.load(memory_order_acquire)) {
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        cout << "\n[Main] Shutdown requested, stopping server..." << endl;
        if (g_server) {
            g_server->Shutdown();
        }
    });

    g_server->Wait();
    shutdown_watcher.join();

    service->Shutdown();
    cout << "[Main] Server stopped." << endl;
    return 0;
}
