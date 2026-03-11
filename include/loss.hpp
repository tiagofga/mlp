#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

namespace mlp {

class Loss {
 public:
  virtual ~Loss() = default;
  virtual double forward(const Matrix &y_pred, const Matrix &y_true) = 0;
  virtual Matrix backward() = 0;
};

class BinaryCrossEntropy : public Loss {
 public:
  double forward(const Matrix &y_pred, const Matrix &y_true) override;
  Matrix backward() override;

 private:
  Matrix y_pred_cache_;
  Matrix y_true_cache_;
};

}  // namespace mlp

#endif
