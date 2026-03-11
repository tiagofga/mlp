#include "activations.hpp"

#include <cmath>

#include "matrix.hpp"

namespace mlp {

Matrix ReLU::forward(const Matrix &input) {
  output_cache_ = make_matrix(rows(input), cols(input));
  for (std::size_t i = 0; i < rows(input); ++i) {
    for (std::size_t j = 0; j < cols(input); ++j) {
      output_cache_[i][j] = input[i][j] > 0.0 ? input[i][j] : 0.0;
    }
  }
  return output_cache_;
}

Matrix ReLU::backward(const Matrix &grad_output) {
  Matrix grad_input = make_matrix(rows(grad_output), cols(grad_output));
  for (std::size_t i = 0; i < rows(grad_output); ++i) {
    for (std::size_t j = 0; j < cols(grad_output); ++j) {
      grad_input[i][j] = output_cache_[i][j] > 0.0 ? grad_output[i][j] : 0.0;
    }
  }
  return grad_input;
}

Matrix Sigmoid::forward(const Matrix &input) {
  output_cache_ = make_matrix(rows(input), cols(input));
  for (std::size_t i = 0; i < rows(input); ++i) {
    for (std::size_t j = 0; j < cols(input); ++j) {
      output_cache_[i][j] = 1.0 / (1.0 + std::exp(-input[i][j]));
    }
  }
  return output_cache_;
}

Matrix Sigmoid::backward(const Matrix &grad_output) {
  Matrix grad_input = make_matrix(rows(grad_output), cols(grad_output));
  for (std::size_t i = 0; i < rows(grad_output); ++i) {
    for (std::size_t j = 0; j < cols(grad_output); ++j) {
      const double s = output_cache_[i][j];
      grad_input[i][j] = grad_output[i][j] * s * (1.0 - s);
    }
  }
  return grad_input;
}

Matrix Tanh::forward(const Matrix &input) {
  output_cache_ = make_matrix(rows(input), cols(input));
  for (std::size_t i = 0; i < rows(input); ++i) {
    for (std::size_t j = 0; j < cols(input); ++j) {
      output_cache_[i][j] = std::tanh(input[i][j]);
    }
  }
  return output_cache_;
}

Matrix Tanh::backward(const Matrix &grad_output) {
  Matrix grad_input = make_matrix(rows(grad_output), cols(grad_output));
  for (std::size_t i = 0; i < rows(grad_output); ++i) {
    for (std::size_t j = 0; j < cols(grad_output); ++j) {
      const double t = output_cache_[i][j];
      grad_input[i][j] = grad_output[i][j] * (1.0 - t * t);
    }
  }
  return grad_input;
}

}  // namespace mlp
