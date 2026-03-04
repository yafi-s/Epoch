set(SCHEDULER_SOURCES
    src/config.cc
    src/job_queue.cc
    src/worker_session.cc
    src/worker_pool.cc
    src/metrics_collector.cc
    src/service_impl.cc
)

add_library(epoch_scheduler_lib ${SCHEDULER_SOURCES})
target_include_directories(epoch_scheduler_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(epoch_scheduler_lib PUBLIC epoch_proto)

add_executable(epoch_scheduler src/main.cc)
target_link_libraries(epoch_scheduler PRIVATE epoch_scheduler_lib)
