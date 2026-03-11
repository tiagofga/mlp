cmake_minimum_required(VERSION 3.16)
project(modular_mlp VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(MLP_ENABLE_CUDA "Enable CUDA backend for dense linear algebra" OFF)
option(MLP_ENABLE_OPENMP "Enable OpenMP for CPU parallelism" OFF)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/include/mlp")
configure_file(
  cmake/version.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/include/mlp/version.hpp"
  @ONLY
)

add_executable(mlp
  src/main.cpp
)
add_executable(mlp_library_example
  examples/library_usage.cpp
)
add_executable(mlp_io_example
  examples/io_roundtrip.cpp
)
add_executable(mlp_test_experiment
  tests/test_experiment.cpp
)
add_executable(mlp_test_io_roundtrip
  tests/test_io_roundtrip.cpp
)

add_library(mlp_core
  src/dense.cpp
  src/activations.cpp
  src/loss.cpp
  src/metrics.cpp
  src/model.cpp
)

add_library(mlp_optim
  src/optimizer.cpp
)

add_library(mlp_train
  src/library.cpp
)

add_library(mlp_io
  src/io.cpp
)

# Compatibility aggregate target
add_library(mlp_lib INTERFACE)

add_library(mlp::mlp_core ALIAS mlp_core)
add_library(mlp::mlp_optim ALIAS mlp_optim)
add_library(mlp::mlp_train ALIAS mlp_train)
add_library(mlp::mlp_io ALIAS mlp_io)
add_library(mlp::mlp_lib ALIAS mlp_lib)

target_include_directories(mlp_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(mlp_optim
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(mlp_train
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(mlp_io
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(mlp_lib
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(mlp_optim PUBLIC mlp_core)
target_link_libraries(mlp_train PUBLIC mlp_core mlp_optim)
target_link_libraries(mlp_io PUBLIC mlp_core)
target_link_libraries(mlp_lib INTERFACE mlp_core mlp_optim mlp_train mlp_io)

target_link_libraries(mlp PRIVATE mlp_lib)
target_link_libraries(mlp_library_example PRIVATE mlp_lib)
target_link_libraries(mlp_io_example PRIVATE mlp_lib)
target_link_libraries(mlp_test_experiment PRIVATE mlp::mlp_train)
target_link_libraries(mlp_test_io_roundtrip PRIVATE mlp::mlp_io)

target_compile_options(mlp PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_optim PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_train PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_io PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_library_example PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_io_example PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_test_experiment PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(mlp_test_io_roundtrip PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_test(NAME mlp_test_experiment COMMAND mlp_test_experiment)
  add_test(NAME mlp_test_io_roundtrip COMMAND mlp_test_io_roundtrip)
  set_tests_properties(mlp_test_io_roundtrip PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME mlp_test_package_consumer
    COMMAND ${CMAKE_COMMAND}
      -DMLP_BINARY_DIR=${CMAKE_BINARY_DIR}
      -DMLP_EXPECTED_VERSION=${PROJECT_VERSION}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_package_consumer.cmake
  )
endif()

if(MLP_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "MLP_ENABLE_OPENMP=ON but OpenMP was not found for CXX compiler.")
  endif()
  target_link_libraries(mlp_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MLP_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "MLP_ENABLE_CUDA=ON but CUDA toolkit was not found. Set CUDAToolkit_ROOT to your CUDA install.")
  endif()
  enable_language(CUDA)

  target_sources(mlp_core PRIVATE src/cuda_ops.cu)
  target_compile_definitions(mlp_core PUBLIC MLP_USE_CUDA)
  target_link_libraries(mlp_core PUBLIC CUDA::cudart)
  set_target_properties(mlp_core PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
endif()

set(MLP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/mlp")

install(
  TARGETS mlp_core mlp_optim mlp_train mlp_io mlp_lib
  EXPORT mlpTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  DIRECTORY include/mlp/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlp"
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/include/mlp/version.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlp"
)

install(
  EXPORT mlpTargets
  FILE mlpTargets.cmake
  NAMESPACE mlp::
  DESTINATION ${MLP_INSTALL_CMAKEDIR}
)

configure_package_config_file(
  cmake/mlpConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/mlpConfig.cmake"
  INSTALL_DESTINATION ${MLP_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mlpConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mlpConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mlpConfigVersion.cmake"
  DESTINATION ${MLP_INSTALL_CMAKEDIR}
)
