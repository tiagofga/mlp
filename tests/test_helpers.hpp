#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <cmath>
#include <iostream>

#include "matrix.hpp"

namespace test {

inline bool matrix_allclose(const mlp::Matrix &lhs, const mlp::Matrix &rhs, double tolerance) {
  if (mlp::rows(lhs) != mlp::rows(rhs) || mlp::cols(lhs) != mlp::cols(rhs)) {
    std::cerr << "matrix shape mismatch\n";
    return false;
  }

  for (std::size_t i = 0; i < mlp::rows(lhs); ++i) {
    for (std::size_t j = 0; j < mlp::cols(lhs); ++j) {
      if (std::fabs(lhs[i][j] - rhs[i][j]) > tolerance) {
        std::cerr << "matrix mismatch at (" << i << ", " << j << "): "
                  << lhs[i][j] << " vs " << rhs[i][j] << "\n";
        return false;
      }
    }
  }

  return true;
}

inline bool vector_allclose(const mlp::Vector &lhs, const mlp::Vector &rhs, double tolerance) {
  if (lhs.size() != rhs.size()) {
    std::cerr << "vector size mismatch\n";
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::fabs(lhs[i] - rhs[i]) > tolerance) {
      std::cerr << "vector mismatch at " << i << ": " << lhs[i] << " vs " << rhs[i] << "\n";
      return false;
    }
  }

  return true;
}

}  // namespace test

#endif
