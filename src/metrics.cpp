#include "mlp/metrics.hpp"

#include <stdexcept>

#include "matrix.hpp"

namespace mlp {

BinaryMetrics compute_binary_metrics(const Matrix &y_pred, const Matrix &y_true, double threshold) {
  if (rows(y_pred) != rows(y_true) || cols(y_pred) != cols(y_true)) {
    throw std::invalid_argument("compute_binary_metrics shape mismatch");
  }
  if (cols(y_pred) != 1) {
    throw std::invalid_argument("compute_binary_metrics expects binary output with one column");
  }

  BinaryMetrics m;
  const double n = static_cast<double>(rows(y_pred));
  const double eps = 1e-12;

  for (std::size_t i = 0; i < rows(y_pred); ++i) {
    const int pred = y_pred[i][0] >= threshold ? 1 : 0;
    const int truth = y_true[i][0] >= 0.5 ? 1 : 0;

    if (pred == 1 && truth == 1) ++m.tp;
    if (pred == 0 && truth == 0) ++m.tn;
    if (pred == 1 && truth == 0) ++m.fp;
    if (pred == 0 && truth == 1) ++m.fn;
  }

  m.accuracy = static_cast<double>(m.tp + m.tn) / (n + eps);
  m.precision = static_cast<double>(m.tp) / (static_cast<double>(m.tp + m.fp) + eps);
  m.recall = static_cast<double>(m.tp) / (static_cast<double>(m.tp + m.fn) + eps);
  m.f1 = 2.0 * m.precision * m.recall / (m.precision + m.recall + eps);

  return m;
}

}  // namespace mlp
