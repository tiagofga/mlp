#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace mlp {
namespace cuda {
namespace {

void check_cuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
  }
}

std::vector<double> flatten(const Matrix &m) {
  std::vector<double> out;
  out.reserve(rows(m) * cols(m));
  for (const auto &row : m) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

Matrix reshape(const std::vector<double> &flat, std::size_t n_rows, std::size_t n_cols) {
  Matrix out = make_matrix(n_rows, n_cols, 0.0);
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < n_cols; ++j) {
      out[i][j] = flat[i * n_cols + j];
    }
  }
  return out;
}

__global__ void matmul_kernel(const double *a, const double *b, double *c, int m, int k, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
}

__global__ void transpose_kernel(const double *in, double *out, int m, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    out[col * m + row] = in[row * n + col];
  }
}

__global__ void add_row_vector_kernel(const double *a, const double *row_vec, double *out, int m, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    out[row * n + col] = a[row * n + col] + row_vec[col];
  }
}

}  // namespace

Matrix matmul(const Matrix &a, const Matrix &b) {
  if (cols(a) != rows(b)) {
    throw std::invalid_argument("Shape mismatch in cuda::matmul");
  }

  const int m = static_cast<int>(rows(a));
  const int k = static_cast<int>(cols(a));
  const int n = static_cast<int>(cols(b));

  const std::vector<double> a_h = flatten(a);
  const std::vector<double> b_h = flatten(b);
  std::vector<double> c_h(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);

  double *a_d = nullptr;
  double *b_d = nullptr;
  double *c_d = nullptr;

  check_cuda(cudaMalloc(&a_d, a_h.size() * sizeof(double)), "cudaMalloc(a)");
  check_cuda(cudaMalloc(&b_d, b_h.size() * sizeof(double)), "cudaMalloc(b)");
  check_cuda(cudaMalloc(&c_d, c_h.size() * sizeof(double)), "cudaMalloc(c)");

  check_cuda(cudaMemcpy(a_d, a_h.data(), a_h.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(a)");
  check_cuda(cudaMemcpy(b_d, b_h.data(), b_h.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(b)");

  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  matmul_kernel<<<grid, block>>>(a_d, b_d, c_d, m, k, n);
  check_cuda(cudaGetLastError(), "matmul_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "matmul_kernel sync");

  check_cuda(cudaMemcpy(c_h.data(), c_d, c_h.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(c)");

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  return reshape(c_h, static_cast<std::size_t>(m), static_cast<std::size_t>(n));
}

Matrix transpose(const Matrix &m) {
  const int m_rows = static_cast<int>(rows(m));
  const int m_cols = static_cast<int>(cols(m));

  const std::vector<double> in_h = flatten(m);
  std::vector<double> out_h(static_cast<std::size_t>(m_rows) * static_cast<std::size_t>(m_cols), 0.0);

  double *in_d = nullptr;
  double *out_d = nullptr;

  check_cuda(cudaMalloc(&in_d, in_h.size() * sizeof(double)), "cudaMalloc(in)");
  check_cuda(cudaMalloc(&out_d, out_h.size() * sizeof(double)), "cudaMalloc(out)");

  check_cuda(cudaMemcpy(in_d, in_h.data(), in_h.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(in)");

  const dim3 block(16, 16);
  const dim3 grid((m_cols + block.x - 1) / block.x, (m_rows + block.y - 1) / block.y);
  transpose_kernel<<<grid, block>>>(in_d, out_d, m_rows, m_cols);
  check_cuda(cudaGetLastError(), "transpose_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "transpose_kernel sync");

  check_cuda(cudaMemcpy(out_h.data(), out_d, out_h.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(out)");

  cudaFree(in_d);
  cudaFree(out_d);

  return reshape(out_h, static_cast<std::size_t>(m_cols), static_cast<std::size_t>(m_rows));
}

Matrix add_row_vector(const Matrix &a, const Vector &row_vec) {
  if (cols(a) != row_vec.size()) {
    throw std::invalid_argument("Shape mismatch in cuda::add_row_vector");
  }

  const int m = static_cast<int>(rows(a));
  const int n = static_cast<int>(cols(a));

  const std::vector<double> a_h = flatten(a);
  const std::vector<double> row_h = row_vec;
  std::vector<double> out_h(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);

  double *a_d = nullptr;
  double *row_d = nullptr;
  double *out_d = nullptr;

  check_cuda(cudaMalloc(&a_d, a_h.size() * sizeof(double)), "cudaMalloc(a)");
  check_cuda(cudaMalloc(&row_d, row_h.size() * sizeof(double)), "cudaMalloc(row)");
  check_cuda(cudaMalloc(&out_d, out_h.size() * sizeof(double)), "cudaMalloc(out)");

  check_cuda(cudaMemcpy(a_d, a_h.data(), a_h.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(a)");
  check_cuda(cudaMemcpy(row_d, row_h.data(), row_h.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(row)");

  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  add_row_vector_kernel<<<grid, block>>>(a_d, row_d, out_d, m, n);
  check_cuda(cudaGetLastError(), "add_row_vector_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "add_row_vector_kernel sync");

  check_cuda(cudaMemcpy(out_h.data(), out_d, out_h.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(out)");

  cudaFree(a_d);
  cudaFree(row_d);
  cudaFree(out_d);

  return reshape(out_h, static_cast<std::size_t>(m), static_cast<std::size_t>(n));
}

}  // namespace cuda
}  // namespace mlp
