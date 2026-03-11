#ifndef CUDA_OPS_HPP
#define CUDA_OPS_HPP

#include "matrix.hpp"

namespace mlp {
namespace cuda {

Matrix matmul(const Matrix &a, const Matrix &b);
Matrix transpose(const Matrix &m);
Matrix add_row_vector(const Matrix &a, const Vector &row_vec);

}  // namespace cuda
}  // namespace mlp

#endif
