#ifndef DENSE_HPP
#define DENSE_HPP

#include <random>

#include "layer.hpp"

namespace mlp {

class Dense : public Layer {
 public:
  Dense(std::size_t in_features, std::size_t out_features, std::mt19937 &rng);

  const char *type() const override { return "Dense"; }
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &grad_output) override;
  void update(double learning_rate) override;
  std::vector<MatrixParamRef> matrix_params() override;
  std::vector<VectorParamRef> vector_params() override;
  const Matrix &weights() const { return weights_; }
  const Vector &bias() const { return bias_; }
  void set_parameters(const Matrix &weights, const Vector &bias);

 private:
  Matrix weights_;
  Vector bias_;

  Matrix input_cache_;
  Matrix grad_weights_;
  Vector grad_bias_;
};

}  // namespace mlp

#endif
