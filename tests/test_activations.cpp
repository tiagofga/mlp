#include <cmath>
#include <iostream>

#include "activations.hpp"

namespace {

bool matrix_allclose(const mlp::Matrix &lhs, const mlp::Matrix &rhs, double tolerance) {
  if (mlp::rows(lhs) != mlp::rows(rhs) || mlp::cols(lhs) != mlp::cols(rhs)) {
    std::cerr << "matrix shape mismatch\n";
    return false;
  }

  for (std::size_t i = 0; i < mlp::rows(lhs); ++i) {
    for (std::size_t j = 0; j < mlp::cols(lhs); ++j) {
      if (std::fabs(lhs[i][j] - rhs[i][j]) > tolerance) {
        std::cerr << "matrix mismatch at (" << i << ", " << j << "): "
                  << lhs[i][j] << " vs " << rhs[i][j] << "\n";
        return false;
      }
    }
  }

  return true;
}

bool run_relu_check() {
  mlp::ReLU relu;
  const mlp::Matrix input = {
      {-2.0, 0.0, 3.5},
      {4.2, -1.1, 0.3},
  };
  const mlp::Matrix grad_output = {
      {0.5, 1.0, -0.25},
      {-1.5, 2.0, 0.75},
  };

  const mlp::Matrix expected_forward = {
      {0.0, 0.0, 3.5},
      {4.2, 0.0, 0.3},
  };
  const mlp::Matrix expected_backward = {
      {0.0, 0.0, -0.25},
      {-1.5, 0.0, 0.75},
  };

  return matrix_allclose(relu.forward(input), expected_forward, 1e-12) &&
         matrix_allclose(relu.backward(grad_output), expected_backward, 1e-12);
}

bool run_sigmoid_check() {
  mlp::Sigmoid sigmoid;
  const mlp::Matrix input = {
      {-2.0, 0.0, 2.0},
      {1.5, -1.5, 0.5},
  };
  const mlp::Matrix grad_output = {
      {0.5, -1.0, 0.25},
      {1.0, 0.75, -0.5},
  };

  mlp::Matrix expected_forward = mlp::make_matrix(mlp::rows(input), mlp::cols(input));
  mlp::Matrix expected_backward = mlp::make_matrix(mlp::rows(input), mlp::cols(input));
  for (std::size_t i = 0; i < mlp::rows(input); ++i) {
    for (std::size_t j = 0; j < mlp::cols(input); ++j) {
      const double s = 1.0 / (1.0 + std::exp(-input[i][j]));
      expected_forward[i][j] = s;
      expected_backward[i][j] = grad_output[i][j] * s * (1.0 - s);
    }
  }

  return matrix_allclose(sigmoid.forward(input), expected_forward, 1e-12) &&
         matrix_allclose(sigmoid.backward(grad_output), expected_backward, 1e-12);
}

bool run_tanh_check() {
  mlp::Tanh tanh;
  const mlp::Matrix input = {
      {-1.0, 0.0, 1.0},
      {0.25, -0.75, 2.0},
  };
  const mlp::Matrix grad_output = {
      {1.0, -0.5, 0.25},
      {-1.25, 0.75, 0.5},
  };

  mlp::Matrix expected_forward = mlp::make_matrix(mlp::rows(input), mlp::cols(input));
  mlp::Matrix expected_backward = mlp::make_matrix(mlp::rows(input), mlp::cols(input));
  for (std::size_t i = 0; i < mlp::rows(input); ++i) {
    for (std::size_t j = 0; j < mlp::cols(input); ++j) {
      const double t = std::tanh(input[i][j]);
      expected_forward[i][j] = t;
      expected_backward[i][j] = grad_output[i][j] * (1.0 - t * t);
    }
  }

  return matrix_allclose(tanh.forward(input), expected_forward, 1e-12) &&
         matrix_allclose(tanh.backward(grad_output), expected_backward, 1e-12);
}

}  // namespace

int main() {
  bool ok = true;
  ok = run_relu_check() && ok;
  ok = run_sigmoid_check() && ok;
  ok = run_tanh_check() && ok;
  return ok ? 0 : 1;
}
