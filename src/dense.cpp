#include "dense.hpp"

#include <cmath>
#include <stdexcept>

#ifdef MLP_USE_CUDA
#include "cuda_ops.hpp"
#endif
#include "matrix.hpp"

namespace mlp {

Dense::Dense(std::size_t in_features, std::size_t out_features, std::mt19937 &rng) {
  const double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
  weights_ = random_matrix(in_features, out_features, -limit, limit, rng);
  bias_ = Vector(out_features, 0.0);
  grad_weights_ = make_matrix(in_features, out_features, 0.0);
  grad_bias_ = Vector(out_features, 0.0);
}

Matrix Dense::forward(const Matrix &input) {
  if (input.empty() || cols(input) != rows(weights_)) {
    throw std::invalid_argument("Dense::forward input shape mismatch");
  }
  input_cache_ = input;
#ifdef MLP_USE_CUDA
  return cuda::add_row_vector(cuda::matmul(input, weights_), bias_);
#else
  return add_row_vector(matmul(input, weights_), bias_);
#endif
}

Matrix Dense::backward(const Matrix &grad_output) {
  if (grad_output.empty() || cols(grad_output) != cols(weights_)) {
    throw std::invalid_argument("Dense::backward grad_output shape mismatch");
  }
  const double inv_batch = 1.0 / static_cast<double>(rows(input_cache_));

  Matrix input_t;
#ifdef MLP_USE_CUDA
  input_t = cuda::transpose(input_cache_);
  grad_weights_ = scalar_multiply(cuda::matmul(input_t, grad_output), inv_batch);
#else
  input_t = transpose(input_cache_);
  grad_weights_ = scalar_multiply(matmul(input_t, grad_output), inv_batch);
#endif
  grad_bias_ = mean_rows(grad_output);

#ifdef MLP_USE_CUDA
  return cuda::matmul(grad_output, cuda::transpose(weights_));
#else
  return matmul(grad_output, transpose(weights_));
#endif
}

void Dense::update(double learning_rate) {
  for (std::size_t i = 0; i < rows(weights_); ++i) {
    for (std::size_t j = 0; j < cols(weights_); ++j) {
      weights_[i][j] -= learning_rate * grad_weights_[i][j];
    }
  }
  for (std::size_t j = 0; j < bias_.size(); ++j) {
    bias_[j] -= learning_rate * grad_bias_[j];
  }
}

std::vector<MatrixParamRef> Dense::matrix_params() { return {{&weights_, &grad_weights_}}; }

std::vector<VectorParamRef> Dense::vector_params() { return {{&bias_, &grad_bias_}}; }

void Dense::set_parameters(const Matrix &weights, const Vector &bias) {
  if (rows(weights) == 0 || cols(weights) == 0) {
    throw std::invalid_argument("Dense::set_parameters received empty weights");
  }
  if (cols(weights) != bias.size()) {
    throw std::invalid_argument("Dense::set_parameters shape mismatch between weights and bias");
  }
  weights_ = weights;
  bias_ = bias;
  grad_weights_ = make_matrix(rows(weights_), cols(weights_), 0.0);
  grad_bias_ = Vector(bias_.size(), 0.0);
}

}  // namespace mlp
