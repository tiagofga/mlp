#include <cmath>
#include <iostream>
#include <random>

#include "dense.hpp"
#include "matrix.hpp"
#include "test_helpers.hpp"

namespace {

bool run_dense_check(std::size_t batch_size, std::size_t in_features, std::size_t out_features) {
  std::mt19937 rng(1234);
  mlp::Dense dense(in_features, out_features, rng);

  const mlp::Matrix weights =
      mlp::random_matrix(in_features, out_features, -0.75, 0.75, rng);
  const mlp::Vector bias(out_features, 0.125);
  const mlp::Matrix input =
      mlp::random_matrix(batch_size, in_features, -1.0, 1.0, rng);
  const mlp::Matrix grad_output =
      mlp::random_matrix(batch_size, out_features, -0.5, 0.5, rng);

  dense.set_parameters(weights, bias);

  const mlp::Matrix forward = dense.forward(input);
  const mlp::Matrix expected_forward = mlp::add_row_vector(mlp::matmul(input, weights), bias);
  if (!test::matrix_allclose(forward, expected_forward, 1e-5)) {
    std::cerr << "forward mismatch\n";
    return false;
  }

  const mlp::Matrix backward = dense.backward(grad_output);
  const mlp::Matrix expected_backward =
      mlp::matmul(grad_output, mlp::transpose(weights));
  if (!test::matrix_allclose(backward, expected_backward, 1e-5)) {
    std::cerr << "backward mismatch\n";
    return false;
  }

  const auto matrix_params = dense.matrix_params();
  const auto vector_params = dense.vector_params();
  const mlp::Matrix expected_grad_weights = mlp::scalar_multiply(
      mlp::matmul(mlp::transpose(input), grad_output),
      1.0 / static_cast<double>(batch_size));
  const mlp::Vector expected_grad_bias = mlp::mean_rows(grad_output);

  if (!test::matrix_allclose(*matrix_params.front().grad, expected_grad_weights, 1e-5)) {
    std::cerr << "weight gradient mismatch\n";
    return false;
  }

  if (!test::vector_allclose(*vector_params.front().grad, expected_grad_bias, 1e-5)) {
    std::cerr << "bias gradient mismatch\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = run_dense_check(4, 3, 5) && ok;
  ok = run_dense_check(7, 11, 6) && ok;
  return ok ? 0 : 1;
}
