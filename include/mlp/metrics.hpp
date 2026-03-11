#ifndef MLP_METRICS_HPP
#define MLP_METRICS_HPP

#include "mlp/types.hpp"

namespace mlp {

struct BinaryMetrics {
  double accuracy = 0.0;
  double precision = 0.0;
  double recall = 0.0;
  double f1 = 0.0;
  int tp = 0;
  int tn = 0;
  int fp = 0;
  int fn = 0;
};

BinaryMetrics compute_binary_metrics(const Matrix &y_pred, const Matrix &y_true, double threshold = 0.5);

}  // namespace mlp

#endif
