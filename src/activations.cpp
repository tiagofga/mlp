#include "activations.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "matrix.hpp"

namespace mlp {

namespace {

void check_backward_cache(const Matrix &cache, const Matrix &grad_output,
                          const char *context) {
  check_rectangular(grad_output, context);
  if (cache.empty()) {
    throw std::logic_error(std::string(context) + " called before forward");
  }
  check_same_shape(cache, grad_output, context);
}

}  // namespace

Matrix ReLU::forward(const Matrix &input) {
  check_rectangular(input, "ReLU::forward input");
  output_cache_ = make_matrix(rows(input), cols(input));
  for (std::size_t i = 0; i < rows(input); ++i) {
    for (std::size_t j = 0; j < cols(input); ++j) {
      output_cache_[i][j] = input[i][j] > 0.0 ? input[i][j] : 0.0;
    }
  }
  return output_cache_;
}

Matrix ReLU::backward(const Matrix &grad_output) {
  check_backward_cache(output_cache_, grad_output, "ReLU::backward");
  Matrix grad_input = make_matrix(rows(grad_output), cols(grad_output));
  for (std::size_t i = 0; i < rows(grad_output); ++i) {
    for (std::size_t j = 0; j < cols(grad_output); ++j) {
      grad_input[i][j] = output_cache_[i][j] > 0.0 ? grad_output[i][j] : 0.0;
    }
  }
  return grad_input;
}

Matrix Sigmoid::forward(const Matrix &input) {
  check_rectangular(input, "Sigmoid::forward input");
  output_cache_ = make_matrix(rows(input), cols(input));
  for (std::size_t i = 0; i < rows(input); ++i) {
    for (std::size_t j = 0; j < cols(input); ++j) {
      output_cache_[i][j] = 1.0 / (1.0 + std::exp(-input[i][j]));
    }
  }
  return output_cache_;
}

Matrix Sigmoid::backward(const Matrix &grad_output) {
  check_backward_cache(output_cache_, grad_output, "Sigmoid::backward");
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
  check_rectangular(input, "Tanh::forward input");
  output_cache_ = make_matrix(rows(input), cols(input));
  for (std::size_t i = 0; i < rows(input); ++i) {
    for (std::size_t j = 0; j < cols(input); ++j) {
      output_cache_[i][j] = std::tanh(input[i][j]);
    }
  }
  return output_cache_;
}

Matrix Tanh::backward(const Matrix &grad_output) {
  check_backward_cache(output_cache_, grad_output, "Tanh::backward");
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
