#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "matrix.hpp"

namespace mlp {

struct MatrixParamRef {
  Matrix *value;
  Matrix *grad;
};

struct VectorParamRef {
  Vector *value;
  Vector *grad;
};

class Layer {
 public:
  virtual ~Layer() = default;
  virtual const char *type() const = 0;
  virtual Matrix forward(const Matrix &input) = 0;
  virtual Matrix backward(const Matrix &grad_output) = 0;
  virtual void update(double) {}
  virtual std::vector<MatrixParamRef> matrix_params() { return {}; }
  virtual std::vector<VectorParamRef> vector_params() { return {}; }
};

}  // namespace mlp

#endif
