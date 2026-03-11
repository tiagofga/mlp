#include "model.hpp"

namespace mlp {

void Sequential::add(std::unique_ptr<Layer> layer) { layers_.push_back(std::move(layer)); }

Matrix Sequential::forward(const Matrix &input) {
  Matrix out = input;
  for (auto &layer : layers_) {
    out = layer->forward(out);
  }
  return out;
}

Matrix Sequential::backward(const Matrix &grad_output) {
  Matrix grad = grad_output;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    grad = (*it)->backward(grad);
  }
  return grad;
}

void Sequential::update(double learning_rate) {
  for (auto &layer : layers_) {
    layer->update(learning_rate);
  }
}

std::vector<std::unique_ptr<Layer>> &Sequential::layers() { return layers_; }

const std::vector<std::unique_ptr<Layer>> &Sequential::layers() const { return layers_; }

}  // namespace mlp
