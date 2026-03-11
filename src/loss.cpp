#include "loss.hpp"

#include <cmath>
#include <stdexcept>

#include "matrix.hpp"

namespace mlp {

double BinaryCrossEntropy::forward(const Matrix &y_pred, const Matrix &y_true) {
  if (rows(y_pred) != rows(y_true) || cols(y_pred) != cols(y_true)) {
    throw std::invalid_argument("BinaryCrossEntropy::forward shape mismatch");
  }
  y_pred_cache_ = clamp(y_pred, 1e-7, 1.0 - 1e-7);
  y_true_cache_ = y_true;

  double loss = 0.0;
  const double n = static_cast<double>(rows(y_pred) * cols(y_pred));
  for (std::size_t i = 0; i < rows(y_pred); ++i) {
    for (std::size_t j = 0; j < cols(y_pred); ++j) {
      const double p = y_pred_cache_[i][j];
      const double y = y_true_cache_[i][j];
      loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }
  }
  return loss / n;
}

Matrix BinaryCrossEntropy::backward() {
  const double n = static_cast<double>(rows(y_pred_cache_) * cols(y_pred_cache_));
  Matrix grad = make_matrix(rows(y_pred_cache_), cols(y_pred_cache_));
  for (std::size_t i = 0; i < rows(y_pred_cache_); ++i) {
    for (std::size_t j = 0; j < cols(y_pred_cache_); ++j) {
      const double p = y_pred_cache_[i][j];
      const double y = y_true_cache_[i][j];
      grad[i][j] = (p - y) / ((p * (1.0 - p)) * n);
    }
  }
  return grad;
}

}  // namespace mlp
