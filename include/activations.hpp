#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "layer.hpp"

namespace mlp {

class ReLU : public Layer {
 public:
  const char *type() const override { return "ReLU"; }
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &grad_output) override;

 private:
  Matrix output_cache_;
};

class Sigmoid : public Layer {
 public:
  const char *type() const override { return "Sigmoid"; }
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &grad_output) override;

 private:
  Matrix output_cache_;
};

class Tanh : public Layer {
 public:
  const char *type() const override { return "Tanh"; }
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &grad_output) override;

 private:
  Matrix output_cache_;
};

}  // namespace mlp

#endif
