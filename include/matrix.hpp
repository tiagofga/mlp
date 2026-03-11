#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "mlp/types.hpp"

namespace mlp {

#ifdef _OPENMP
#define MLP_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define MLP_OMP_PARALLEL_FOR
#endif

inline std::size_t rows(const Matrix &m) { return m.size(); }
inline std::size_t cols(const Matrix &m) { return m.empty() ? 0 : m[0].size(); }

inline Matrix make_matrix(std::size_t n_rows, std::size_t n_cols, double value = 0.0) {
  return Matrix(n_rows, Vector(n_cols, value));
}

inline void check_same_shape(const Matrix &a, const Matrix &b, const char *op) {
  if (rows(a) != rows(b) || cols(a) != cols(b)) {
    throw std::invalid_argument(std::string("Shape mismatch in ") + op);
  }
}

inline Matrix matmul(const Matrix &a, const Matrix &b) {
  if (cols(a) != rows(b)) {
    throw std::invalid_argument("Shape mismatch in matmul");
  }
  Matrix out = make_matrix(rows(a), cols(b));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t k = 0; k < cols(a); ++k) {
      const double aik = a[i][k];
      for (std::size_t j = 0; j < cols(b); ++j) {
        out[i][j] += aik * b[k][j];
      }
    }
  }
  return out;
}

inline Matrix transpose(const Matrix &m) {
  Matrix out = make_matrix(cols(m), rows(m));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(m); ++i) {
    for (std::size_t j = 0; j < cols(m); ++j) {
      out[j][i] = m[i][j];
    }
  }
  return out;
}

inline Matrix add(const Matrix &a, const Matrix &b) {
  check_same_shape(a, b, "add");
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] + b[i][j];
    }
  }
  return out;
}

inline Matrix subtract(const Matrix &a, const Matrix &b) {
  check_same_shape(a, b, "subtract");
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] - b[i][j];
    }
  }
  return out;
}

inline Matrix multiply_elementwise(const Matrix &a, const Matrix &b) {
  check_same_shape(a, b, "multiply_elementwise");
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] * b[i][j];
    }
  }
  return out;
}

inline Matrix divide_elementwise(const Matrix &a, const Matrix &b, double eps = 1e-12) {
  check_same_shape(a, b, "divide_elementwise");
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] / (b[i][j] + eps);
    }
  }
  return out;
}

inline Matrix scalar_multiply(const Matrix &a, double scalar) {
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] * scalar;
    }
  }
  return out;
}

inline Matrix scalar_add(const Matrix &a, double scalar) {
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] + scalar;
    }
  }
  return out;
}

inline Matrix clamp(const Matrix &a, double lo, double hi) {
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      double v = a[i][j];
      if (v < lo) v = lo;
      if (v > hi) v = hi;
      out[i][j] = v;
    }
  }
  return out;
}

inline Matrix add_row_vector(const Matrix &a, const Vector &row_vec) {
  if (cols(a) != row_vec.size()) {
    throw std::invalid_argument("Shape mismatch in add_row_vector");
  }
  Matrix out = make_matrix(rows(a), cols(a));
  MLP_OMP_PARALLEL_FOR
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[i][j] = a[i][j] + row_vec[j];
    }
  }
  return out;
}

inline Vector mean_rows(const Matrix &a) {
  Vector out(cols(a), 0.0);
  if (rows(a) == 0) return out;
  for (std::size_t i = 0; i < rows(a); ++i) {
    for (std::size_t j = 0; j < cols(a); ++j) {
      out[j] += a[i][j];
    }
  }
  const double inv_n = 1.0 / static_cast<double>(rows(a));
  for (double &v : out) v *= inv_n;
  return out;
}

inline Matrix random_matrix(std::size_t n_rows, std::size_t n_cols, double min_val, double max_val,
                            std::mt19937 &rng) {
  std::uniform_real_distribution<double> dist(min_val, max_val);
  Matrix out = make_matrix(n_rows, n_cols);
  for (auto &row : out) {
    for (double &v : row) {
      v = dist(rng);
    }
  }
  return out;
}

inline Matrix zeros_like(const Matrix &a) { return make_matrix(rows(a), cols(a), 0.0); }

}  // namespace mlp

#undef MLP_OMP_PARALLEL_FOR

#endif
