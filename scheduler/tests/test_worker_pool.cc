#include "epoch/worker_pool.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace epoch {
namespace {

// We cannot easily create a real gRPC stream in unit tests, so WorkerSession
// methods that touch the stream (AssignJob) won't be tested here — those are
// covered by integration tests. We test pool logic with sessions whose stream
// pointer is null (safe since we only test state management, not writing).

shared_ptr<WorkerSession> MakeSession(const string& id) {
    return make_shared<WorkerSession>(id, nullptr);
}

TEST(WorkerPoolTest, RegisterAndSize) {
    WorkerPool pool;
    EXPECT_EQ(pool.Size(), 0u);
    EXPECT_TRUE(pool.RegisterWorker(MakeSession("w0")));
    EXPECT_EQ(pool.Size(), 1u);
}

TEST(WorkerPoolTest, DuplicateRegistrationFails) {
    WorkerPool pool;
    pool.RegisterWorker(MakeSession("w0"));
    EXPECT_FALSE(pool.RegisterWorker(MakeSession("w0")));
}

TEST(WorkerPoolTest, GetIdleWorkerReturnsIdle) {
    WorkerPool pool;
    pool.RegisterWorker(MakeSession("w0"));
    auto idle = pool.GetIdleWorker();
    ASSERT_NE(idle, nullptr);
    EXPECT_EQ(idle->WorkerId(), "w0");
}

TEST(WorkerPoolTest, GetIdleWorkerReturnsNullWhenAllBusy) {
    WorkerPool pool;
    pool.RegisterWorker(MakeSession("w0"));
    pool.MarkWorkerBusy("w0");
    auto idle = pool.GetIdleWorker();
    EXPECT_EQ(idle, nullptr);
}

TEST(WorkerPoolTest, MarkIdleRestoresAvailability) {
    WorkerPool pool;
    pool.RegisterWorker(MakeSession("w0"));
    pool.MarkWorkerBusy("w0");
    EXPECT_EQ(pool.IdleCount(), 0u);

    pool.MarkWorkerIdle("w0");
    EXPECT_EQ(pool.IdleCount(), 1u);
}

TEST(WorkerPoolTest, RemoveWorkerDecreasesSize) {
    WorkerPool pool;
    pool.RegisterWorker(MakeSession("w0"));
    pool.RegisterWorker(MakeSession("w1"));
    EXPECT_EQ(pool.Size(), 2u);

    pool.RemoveWorker("w0");
    EXPECT_EQ(pool.Size(), 1u);
}

TEST(WorkerPoolTest, HeartbeatTimeoutDetection) {
    WorkerPool pool;
    auto session = MakeSession("w0");
    pool.RegisterWorker(session);

    // Should not be timed out yet (just created)
    auto timed_out = pool.CheckHeartbeats(100);
    EXPECT_TRUE(timed_out.empty());

    // Wait and check with a very short timeout
    this_thread::sleep_for(chrono::milliseconds(50));
    timed_out = pool.CheckHeartbeats(10);  // 10ms timeout
    EXPECT_THAT(timed_out, testing::Contains("w0"));
}

TEST(WorkerPoolTest, IdleCountWithMultipleWorkers) {
    WorkerPool pool;
    pool.RegisterWorker(MakeSession("w0"));
    pool.RegisterWorker(MakeSession("w1"));
    pool.RegisterWorker(MakeSession("w2"));
    EXPECT_EQ(pool.IdleCount(), 3u);

    pool.MarkWorkerBusy("w1");
    EXPECT_EQ(pool.IdleCount(), 2u);

    pool.MarkWorkerDead("w2");
    EXPECT_EQ(pool.IdleCount(), 1u);
}

}  // namespace
}  // namespace epoch
