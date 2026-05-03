#include "dense.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef MLP_USE_CUDA
#include "cuda_ops.hpp"
#endif
#ifdef MLP_USE_BLAS
#ifdef MLP_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif
#include "matrix.hpp"

namespace mlp {

namespace {

enum class GemmTranspose {
  No,
  Yes,
};

#ifdef MLP_USE_BLAS
std::vector<double> flatten_row_major(const Matrix &matrix) {
  std::vector<double> flat;
  flat.reserve(rows(matrix) * cols(matrix));
  for (const auto &row : matrix) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

Matrix unflatten_row_major(const std::vector<double> &flat, std::size_t n_rows,
                           std::size_t n_cols) {
  Matrix out = make_matrix(n_rows, n_cols);
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < n_cols; ++j) {
      out[i][j] = flat[i * n_cols + j];
    }
  }
  return out;
}

CBLAS_TRANSPOSE to_cblas(GemmTranspose transpose) {
  return transpose == GemmTranspose::No ? CblasNoTrans : CblasTrans;
}

Matrix blas_matmul(const Matrix &lhs, const Matrix &rhs, GemmTranspose lhs_transpose,
                   GemmTranspose rhs_transpose) {
  const std::size_t lhs_rows =
      lhs_transpose == GemmTranspose::No ? rows(lhs) : cols(lhs);
  const std::size_t lhs_cols =
      lhs_transpose == GemmTranspose::No ? cols(lhs) : rows(lhs);
  const std::size_t rhs_rows =
      rhs_transpose == GemmTranspose::No ? rows(rhs) : cols(rhs);
  const std::size_t rhs_cols =
      rhs_transpose == GemmTranspose::No ? cols(rhs) : rows(rhs);

  if (lhs_cols != rhs_rows) {
    throw std::invalid_argument("Shape mismatch in blas_matmul");
  }

  const std::vector<double> lhs_flat = flatten_row_major(lhs);
  const std::vector<double> rhs_flat = flatten_row_major(rhs);
  std::vector<double> out(lhs_rows * rhs_cols, 0.0);

  cblas_dgemm(CblasRowMajor, to_cblas(lhs_transpose), to_cblas(rhs_transpose),
              static_cast<int>(lhs_rows), static_cast<int>(rhs_cols),
              static_cast<int>(lhs_cols), 1.0, lhs_flat.data(),
              static_cast<int>(cols(lhs)), rhs_flat.data(), static_cast<int>(cols(rhs)),
              0.0, out.data(), static_cast<int>(rhs_cols));
  return unflatten_row_major(out, lhs_rows, rhs_cols);
}
#endif

Matrix dense_matmul(const Matrix &lhs, const Matrix &rhs,
                    GemmTranspose lhs_transpose = GemmTranspose::No,
                    GemmTranspose rhs_transpose = GemmTranspose::No) {
#ifdef MLP_USE_BLAS
  return blas_matmul(lhs, rhs, lhs_transpose, rhs_transpose);
#else
  if (lhs_transpose == GemmTranspose::No && rhs_transpose == GemmTranspose::No) {
    return matmul(lhs, rhs);
  }

  const Matrix lhs_matrix =
      lhs_transpose == GemmTranspose::No ? lhs : transpose(lhs);
  const Matrix rhs_matrix =
      rhs_transpose == GemmTranspose::No ? rhs : transpose(rhs);
  return matmul(lhs_matrix, rhs_matrix);
#endif
}

}  // namespace

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
  return add_row_vector(dense_matmul(input, weights_), bias_);
#endif
}

Matrix Dense::backward(const Matrix &grad_output) {
  if (grad_output.empty() || cols(grad_output) != cols(weights_)) {
    throw std::invalid_argument("Dense::backward grad_output shape mismatch");
  }
  const double inv_batch = 1.0 / static_cast<double>(rows(input_cache_));

#ifdef MLP_USE_CUDA
  grad_weights_ =
      scalar_multiply(cuda::matmul(cuda::transpose(input_cache_), grad_output), inv_batch);
#else
  grad_weights_ = scalar_multiply(
      dense_matmul(input_cache_, grad_output, GemmTranspose::Yes, GemmTranspose::No),
      inv_batch);
#endif
  grad_bias_ = mean_rows(grad_output);

#ifdef MLP_USE_CUDA
  return cuda::matmul(grad_output, cuda::transpose(weights_));
#else
  return dense_matmul(grad_output, weights_, GemmTranspose::No, GemmTranspose::Yes);
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
