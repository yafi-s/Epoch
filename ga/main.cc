#include <csignal>
#include <iostream>
#include <memory>

#include <grpcpp/grpcpp.h>

#include "epoch/config.h"
#include "epoch/service_impl.h"

using namespace std;

namespace {
unique_ptr<grpc::Server> g_server;

void SignalHandler(int signal) {
    cout << "\n[Main] Caught signal " << signal << ", shutting down..." << endl;
    if (g_server) {
        g_server->Shutdown();
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    auto config = epoch::Config::FromArgs(argc, argv);

    cout << "╔══════════════════════════════════════╗\n"
              << "║          Epoch Scheduler             ║\n"
              << "╚══════════════════════════════════════╝\n"
              << "Listening on: " << config.listen_address << "\n"
              << "Dispatch interval: " << config.dispatch_interval_ms << "ms\n"
              << "Heartbeat timeout: " << config.heartbeat_timeout_ms << "ms\n"
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
    g_server->Wait();

    service->Shutdown();
    cout << "[Main] Server stopped." << endl;
    return 0;
}
