#include "epoch/service_impl.h"

#include <gtest/gtest.h>

namespace epoch {
namespace {

HyperparamConfig MakeConfig() {
    HyperparamConfig config;
    config.set_learning_rate(0.001);
    config.set_batch_size(32);
    config.set_optimizer(OPTIMIZER_ADAM);
    config.set_epochs(1);
    config.set_dataset("mnist");
    config.add_conv_filters(16);
    config.add_conv_filters(32);
    config.add_conv_filters(32);
    config.add_dense_units(64);
    config.set_kernel_size(3);
    config.set_dropout_rate(0.1);
    config.set_activation(ACTIVATION_RELU);
    return config;
}

TEST(EpochServiceImplTest, SchedulerStatusDefaultsToZero) {
    Config config;
    EpochServiceImpl service(config);

    grpc::ServerContext status_ctx;
    SchedulerStatusRequest status_req;
    SchedulerStatusResponse status_resp;
    const auto status = service.GetSchedulerStatus(&status_ctx, &status_req, &status_resp);

    ASSERT_TRUE(status.ok());
    EXPECT_EQ(status_resp.connected_workers(), 0u);
    EXPECT_EQ(status_resp.idle_workers(), 0u);
    EXPECT_EQ(status_resp.busy_workers(), 0u);
    EXPECT_EQ(status_resp.pending_jobs(), 0u);
}

TEST(EpochServiceImplTest, SchedulerStatusReportsPendingJobsAfterSubmission) {
    Config config;
    EpochServiceImpl service(config);

    grpc::ServerContext submit_ctx;
    SubmitGenerationRequest submit_req;
    submit_req.set_generation_id(0);
    *submit_req.add_configs() = MakeConfig();
    *submit_req.add_configs() = MakeConfig();
    SubmitGenerationResponse submit_resp;
    const auto submit_status = service.SubmitGeneration(&submit_ctx, &submit_req, &submit_resp);

    ASSERT_TRUE(submit_status.ok());
    ASSERT_TRUE(submit_resp.accepted());
    EXPECT_EQ(submit_resp.num_jobs(), 2);

    grpc::ServerContext status_ctx;
    SchedulerStatusRequest status_req;
    SchedulerStatusResponse status_resp;
    const auto status = service.GetSchedulerStatus(&status_ctx, &status_req, &status_resp);

    ASSERT_TRUE(status.ok());
    EXPECT_EQ(status_resp.connected_workers(), 0u);
    EXPECT_EQ(status_resp.idle_workers(), 0u);
    EXPECT_EQ(status_resp.busy_workers(), 0u);
    EXPECT_EQ(status_resp.pending_jobs(), 2u);
}

}  // namespace
}  // namespace epoch
