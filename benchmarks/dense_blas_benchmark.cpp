#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "dense.hpp"
#include "matrix.hpp"

namespace {

double max_abs_diff(const mlp::Matrix &lhs, const mlp::Matrix &rhs) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < mlp::rows(lhs); ++i) {
    for (std::size_t j = 0; j < mlp::cols(lhs); ++j) {
      max_diff = std::max(max_diff, std::fabs(lhs[i][j] - rhs[i][j]));
    }
  }
  return max_diff;
}

double max_abs_diff(const mlp::Vector &lhs, const mlp::Vector &rhs) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

struct TimedResult {
  double seconds;
  double sink;
};

TimedResult run_dense_path(mlp::Dense &dense, const mlp::Matrix &input,
                           const mlp::Matrix &grad_output, int iterations) {
  volatile double sink = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const mlp::Matrix output = dense.forward(input);
    const mlp::Matrix grad_input = dense.backward(grad_output);
    sink += output.front().front() + grad_input.front().front();
  }
  const auto end = std::chrono::steady_clock::now();
  return {
      std::chrono::duration<double>(end - start).count(),
      sink,
  };
}

TimedResult run_naive_path(const mlp::Matrix &weights, const mlp::Vector &bias,
                           const mlp::Matrix &input, const mlp::Matrix &grad_output,
                           int iterations) {
  volatile double sink = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const mlp::Matrix output = mlp::add_row_vector(mlp::matmul(input, weights), bias);
    const mlp::Matrix grad_weights =
        mlp::scalar_multiply(mlp::matmul(mlp::transpose(input), grad_output),
                             1.0 / static_cast<double>(mlp::rows(input)));
    const mlp::Vector grad_bias = mlp::mean_rows(grad_output);
    const mlp::Matrix grad_input = mlp::matmul(grad_output, mlp::transpose(weights));
    sink += output.front().front() + grad_weights.front().front() + grad_bias.front() +
            grad_input.front().front();
  }
  const auto end = std::chrono::steady_clock::now();
  return {
      std::chrono::duration<double>(end - start).count(),
      sink,
  };
}

}  // namespace

int main(int argc, char **argv) {
  constexpr std::size_t kBatchSize = 512;
  constexpr std::size_t kInFeatures = 512;
  constexpr std::size_t kOutFeatures = 512;
  constexpr int kWarmupIterations = 3;
  constexpr int kIterations = 20;
  constexpr double kTolerance = 1e-5;
  double min_speedup = 0.0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--min-speedup") {
      if (i + 1 >= argc) {
        std::cerr << "--min-speedup requires a value\n";
        return 1;
      }
      min_speedup = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--help") {
      std::cout << "Usage: mlp_dense_benchmark [--min-speedup value]\n";
      return 0;
    }
    std::cerr << "unknown argument: " << arg << "\n";
    return 1;
  }

  std::mt19937 rng(123);
  mlp::Dense dense(kInFeatures, kOutFeatures, rng);

  const mlp::Matrix weights =
      mlp::random_matrix(kInFeatures, kOutFeatures, -0.25, 0.25, rng);
  const mlp::Vector bias(kOutFeatures, 0.05);
  const mlp::Matrix input =
      mlp::random_matrix(kBatchSize, kInFeatures, -1.0, 1.0, rng);
  const mlp::Matrix grad_output =
      mlp::random_matrix(kBatchSize, kOutFeatures, -0.5, 0.5, rng);

  dense.set_parameters(weights, bias);

  const mlp::Matrix output = dense.forward(input);
  const mlp::Matrix grad_input = dense.backward(grad_output);
  const auto matrix_params = dense.matrix_params();
  const auto vector_params = dense.vector_params();

  const mlp::Matrix expected_output = mlp::add_row_vector(mlp::matmul(input, weights), bias);
  const mlp::Matrix expected_grad_input =
      mlp::matmul(grad_output, mlp::transpose(weights));
  const mlp::Matrix expected_grad_weights = mlp::scalar_multiply(
      mlp::matmul(mlp::transpose(input), grad_output),
      1.0 / static_cast<double>(kBatchSize));
  const mlp::Vector expected_grad_bias = mlp::mean_rows(grad_output);

  const double forward_diff = max_abs_diff(output, expected_output);
  const double grad_input_diff = max_abs_diff(grad_input, expected_grad_input);
  const double grad_weights_diff =
      max_abs_diff(*matrix_params.front().grad, expected_grad_weights);
  const double grad_bias_diff =
      max_abs_diff(*vector_params.front().grad, expected_grad_bias);

  if (forward_diff > kTolerance || grad_input_diff > kTolerance ||
      grad_weights_diff > kTolerance || grad_bias_diff > kTolerance) {
    std::cerr << "numerical mismatch: forward=" << forward_diff
              << " grad_input=" << grad_input_diff
              << " grad_weights=" << grad_weights_diff
              << " grad_bias=" << grad_bias_diff << "\n";
    return 1;
  }

  static_cast<void>(run_dense_path(dense, input, grad_output, kWarmupIterations));
  static_cast<void>(run_naive_path(weights, bias, input, grad_output, kWarmupIterations));
  const TimedResult dense_result = run_dense_path(dense, input, grad_output, kIterations);
  const TimedResult naive_result =
      run_naive_path(weights, bias, input, grad_output, kIterations);
  const double speedup = naive_result.seconds / dense_result.seconds;

#ifdef MLP_USE_BLAS
  std::cout << "mode=blas\n";
#else
  std::cout << "mode=fallback\n";
#endif
  std::cout << "warmup_iterations=" << kWarmupIterations << "\n";
  std::cout << "iterations=" << kIterations << "\n";
  std::cout << "dense_seconds=" << dense_result.seconds << "\n";
  std::cout << "naive_seconds=" << naive_result.seconds << "\n";
  std::cout << "speedup=" << speedup << "\n";
  std::cout << "sink=" << (dense_result.sink + naive_result.sink) << "\n";
#ifdef MLP_USE_BLAS
  if (min_speedup > 0.0 && speedup < min_speedup) {
    std::cerr << "speedup check failed: expected at least " << min_speedup
              << ", observed " << speedup << "\n";
    return 1;
  }
#else
  if (min_speedup > 0.0) {
    std::cerr << "--min-speedup requires a BLAS-enabled build\n";
    return 1;
  }
#endif
  return 0;
}
