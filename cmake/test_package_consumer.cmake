cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED MLP_BINARY_DIR)
  message(FATAL_ERROR "MLP_BINARY_DIR is required")
endif()
if(NOT DEFINED MLP_EXPECTED_VERSION)
  message(FATAL_ERROR "MLP_EXPECTED_VERSION is required")
endif()

set(INSTALL_PREFIX "${MLP_BINARY_DIR}/test-install")
set(CONSUMER_DIR "${MLP_BINARY_DIR}/test-consumer")
set(CONSUMER_BUILD_DIR "${CONSUMER_DIR}/build")

file(REMOVE_RECURSE "${INSTALL_PREFIX}" "${CONSUMER_DIR}")
file(MAKE_DIRECTORY "${CONSUMER_DIR}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${MLP_BINARY_DIR}" --prefix "${INSTALL_PREFIX}"
  RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "Failed to install package for consumer test")
endif()

file(WRITE "${CONSUMER_DIR}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.16)
project(mlp_package_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(mlp REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE mlp::mlp_lib)
]=])

file(WRITE "${CONSUMER_DIR}/main.cpp" [=[
#include <iostream>
#include "mlp/library.hpp"
#include "mlp/version.hpp"

int main() {
  mlp::ExperimentOptions opt;
  opt.epochs = 20;
  opt.samples = 80;
  const auto rep = mlp::run_xor_experiment(opt, nullptr);
  std::cout << "version=" << MLP_VERSION_STRING << " acc=" << rep.test.metrics.accuracy << "\n";
  return 0;
}
]=])

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_DIR}" -B "${CONSUMER_BUILD_DIR}" "-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}"
  RESULT_VARIABLE cfg_result
)
if(NOT cfg_result EQUAL 0)
  message(FATAL_ERROR "Consumer configure failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD_DIR}"
  RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Consumer build failed")
endif()

execute_process(
  COMMAND "${CONSUMER_BUILD_DIR}/consumer"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_out
)
if(NOT run_result EQUAL 0)
  message(FATAL_ERROR "Consumer run failed")
endif()

string(FIND "${run_out}" "version=${MLP_EXPECTED_VERSION}" version_pos)
if(version_pos EQUAL -1)
  message(FATAL_ERROR "Consumer output missing expected version. Output: ${run_out}")
endif()

message(STATUS "Package consumer test passed: ${run_out}")
