cmake_minimum_required(VERSION 3.20)
project(epoch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Fetch gRPC (brings in Protobuf + Abseil) ────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.62.1
    GIT_SHALLOW    TRUE
)
set(FETCHCONTENT_QUIET OFF)
set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
# The fetched deps are build-tree only; their install/export rules fail under
# newer CMake (protobuf exports reference absl targets with no export set).
set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gRPC)

# ─── Proto code generation ────────────────────────────────────────────────────
set(PROTO_SRC "${CMAKE_SOURCE_DIR}/protos/epoch.proto")
set(PROTO_GEN_DIR "${CMAKE_SOURCE_DIR}/generated/cpp")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)

set(PROTO_GENERATED_SRCS
    "${PROTO_GEN_DIR}/epoch.pb.cc"
    "${PROTO_GEN_DIR}/epoch.pb.h"
    "${PROTO_GEN_DIR}/epoch.grpc.pb.cc"
    "${PROTO_GEN_DIR}/epoch.grpc.pb.h"
)

add_custom_command(
    OUTPUT ${PROTO_GENERATED_SRCS}
    COMMAND ${_PROTOBUF_PROTOC}
        --proto_path=${CMAKE_SOURCE_DIR}/protos
        --cpp_out=${PROTO_GEN_DIR}
        --grpc_out=${PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN}
        ${PROTO_SRC}
    DEPENDS ${PROTO_SRC}
    COMMENT "Generating C++ protobuf/gRPC sources"
)

add_library(epoch_proto ${PROTO_GENERATED_SRCS})
target_include_directories(epoch_proto PUBLIC ${PROTO_GEN_DIR})
target_link_libraries(epoch_proto PUBLIC grpc++ grpc++_reflection libprotobuf)

# ─── Scheduler ────────────────────────────────────────────────────────────────
add_subdirectory(scheduler)

# ─── Tests ────────────────────────────────────────────────────────────────────
option(EPOCH_BUILD_TESTS "Build tests" ON)
if(EPOCH_BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(scheduler/tests)
endif()
