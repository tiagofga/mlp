#include <iostream>
#include <random>

#include "activations.hpp"
#include "dense.hpp"
#include "gradient_check.hpp"
#include "matrix.hpp"

namespace {

bool run_dense_gradient_check() {
  std::mt19937 rng(7);
  mlp::Dense dense(3, 2, rng);

  const mlp::Matrix weights = {
      {0.15, -0.20},
      {0.05, 0.30},
      {-0.25, 0.10},
  };
  const mlp::Vector bias = {0.12, -0.08};
  const mlp::Matrix input = {
      {0.20, -0.40, 0.10},
      {-0.30, 0.50, 0.70},
      {0.60, -0.10, -0.20},
      {0.15, 0.25, -0.35},
  };
  const mlp::Matrix grad_output = {
      {0.40, -0.30},
      {-0.10, 0.20},
      {0.35, 0.15},
      {-0.25, 0.05},
  };

  dense.set_parameters(weights, bias);

  gradient_check::Options options;
  options.parameter_objective_scale = 1.0 / static_cast<double>(mlp::rows(input));
  options.relative_tolerance = 1e-5;
  return gradient_check::check_layer_gradients(dense, input, grad_output, options);
}

bool run_tanh_gradient_check() {
  mlp::Tanh tanh;
  const mlp::Matrix input = {
      {-0.60, 0.10, 0.75},
      {0.20, -0.35, 0.40},
  };
  const mlp::Matrix grad_output = {
      {0.50, -0.20, 0.30},
      {-0.10, 0.40, -0.25},
  };

  return gradient_check::check_layer_gradients(tanh, input, grad_output, {});
}

}  // namespace

int main() {
  bool ok = true;
  ok = run_dense_gradient_check() && ok;
  ok = run_tanh_gradient_check() && ok;
  return ok ? 0 : 1;
}
