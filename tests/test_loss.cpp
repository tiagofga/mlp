#include <cmath>
#include <iostream>
#include <stdexcept>

#include "loss.hpp"

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

bool run_binary_cross_entropy_forward_backward_check() {
  mlp::BinaryCrossEntropy loss;

  const mlp::Matrix y_pred = {
      {0.9, 0.25},
      {0.6, 0.4},
  };
  const mlp::Matrix y_true = {
      {1.0, 0.0},
      {1.0, 0.0},
  };

  const double loss_value = loss.forward(y_pred, y_true);
  const double expected_loss =
      -(std::log(0.9) + std::log(0.75) + std::log(0.6) + std::log(0.6)) / 4.0;
  if (std::fabs(loss_value - expected_loss) > 1e-12) {
    std::cerr << "loss mismatch: " << loss_value << " vs " << expected_loss << "\n";
    return false;
  }

  const mlp::Matrix expected_grad = {
      {-1.0 / (0.9 * 4.0), 1.0 / (0.75 * 4.0)},
      {-1.0 / (0.6 * 4.0), 1.0 / (0.6 * 4.0)},
  };
  return matrix_allclose(loss.backward(), expected_grad, 1e-12);
}

bool run_binary_cross_entropy_shape_mismatch_check() {
  mlp::BinaryCrossEntropy loss;

  try {
    const mlp::Matrix y_pred = {
        {0.7, 0.3},
    };
    const mlp::Matrix y_true = {
        {1.0},
        {0.0},
    };
    (void)loss.forward(y_pred, y_true);
  } catch (const std::invalid_argument &) {
    return true;
  }

  std::cerr << "expected shape mismatch exception\n";
  return false;
}

bool run_binary_cross_entropy_clamp_check() {
  mlp::BinaryCrossEntropy loss;

  const mlp::Matrix y_pred = {
      {0.0},
      {1.0},
  };
  const mlp::Matrix y_true = {
      {1.0},
      {0.0},
  };

  const double loss_value = loss.forward(y_pred, y_true);
  if (!std::isfinite(loss_value)) {
    std::cerr << "expected finite loss, got " << loss_value << "\n";
    return false;
  }

  if (loss_value < 0.0) {
    std::cerr << "expected non-negative loss, got " << loss_value << "\n";
    return false;
  }

  const mlp::Matrix grad = loss.backward();
  for (std::size_t i = 0; i < mlp::rows(grad); ++i) {
    for (std::size_t j = 0; j < mlp::cols(grad); ++j) {
      if (!std::isfinite(grad[i][j])) {
        std::cerr << "expected finite gradient at (" << i << ", " << j << ")\n";
        return false;
      }
    }
  }

  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = run_binary_cross_entropy_forward_backward_check() && ok;
  ok = run_binary_cross_entropy_shape_mismatch_check() && ok;
  ok = run_binary_cross_entropy_clamp_check() && ok;
  return ok ? 0 : 1;
}
