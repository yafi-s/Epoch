#include "epoch/config.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

namespace epoch {

Config Config::FromArgs(int argc, char* argv[]) {
    Config cfg;

    // Check environment variable first
    const char* env_addr = getenv("EPOCH_SCHEDULER_ADDRESS");
    if (env_addr != nullptr) {
        cfg.listen_address = env_addr;
    }
    const char* env_auth_key = getenv("EPOCH_WORKER_AUTH_KEY");
    if (env_auth_key != nullptr) {
        cfg.worker_auth_key = env_auth_key;
    }
    const char* env_metrics_log_path = getenv("EPOCH_METRICS_LOG_PATH");
    if (env_metrics_log_path != nullptr) {
        cfg.metrics_log_path = env_metrics_log_path;
    }

    // CLI flags override environment
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--listen-address") == 0 && i + 1 < argc) {
            cfg.listen_address = argv[++i];
        } else if (strcmp(argv[i], "--dispatch-interval") == 0 && i + 1 < argc) {
            cfg.dispatch_interval_ms = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--heartbeat-timeout") == 0 && i + 1 < argc) {
            cfg.heartbeat_timeout_ms = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--worker-auth-key") == 0 && i + 1 < argc) {
            cfg.worker_auth_key = argv[++i];
        } else if (strcmp(argv[i], "--metrics-log-path") == 0 && i + 1 < argc) {
            cfg.metrics_log_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            cout << "epoch_scheduler [options]\n"
                      << "  --listen-address <addr>    Listen address (default: 0.0.0.0:50051)\n"
                      << "  --dispatch-interval <ms>   Dispatch loop interval (default: 10)\n"
                      << "  --heartbeat-timeout <ms>   Worker heartbeat timeout (default: 30000)\n"
                      << "  --worker-auth-key <key>    Worker registration auth key (default: "
                      << "superkey)\n"
                      << "  --metrics-log-path <path>  Write scheduler runtime metrics JSONL "
                         "(default: disabled)\n";
            exit(0);
        }
    }

    return cfg;
}

}  // namespace epoch
