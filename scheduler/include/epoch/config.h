#pragma once

#include <cstdint>
#include <string>

using namespace std;

namespace epoch {

/// Server configuration parsed from CLI flags and environment variables.
struct Config {
    string listen_address = "0.0.0.0:50051";
    uint32_t dispatch_interval_ms = 10;
    uint32_t heartbeat_check_interval_ms = 5000;
    uint32_t heartbeat_timeout_ms = 30000;
    string worker_auth_key = "superkey";
    string metrics_log_path;

    /// Parse config from argc/argv. Recognizes:
    ///   --listen-address <addr>
    ///   --dispatch-interval <ms>
    ///   --heartbeat-timeout <ms>
    ///   --worker-auth-key <key>
    ///   --metrics-log-path <path>
    /// Falls back to EPOCH_SCHEDULER_ADDRESS env var for address.
    static Config FromArgs(int argc, char* argv[]);
};

}  // namespace epoch
