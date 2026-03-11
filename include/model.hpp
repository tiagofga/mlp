#ifndef MODEL_HPP
#define MODEL_HPP

#include <memory>
#include <utility>
#include <vector>

#include "layer.hpp"

namespace mlp {

class Sequential {
 public:
  void add(std::unique_ptr<Layer> layer);
  Matrix forward(const Matrix &input);
  Matrix backward(const Matrix &grad_output);
  void update(double learning_rate);
  std::vector<std::unique_ptr<Layer>> &layers();
  const std::vector<std::unique_ptr<Layer>> &layers() const;

 private:
  std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace mlp

#endif
