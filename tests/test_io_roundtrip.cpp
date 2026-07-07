#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "activations.hpp"
#include "dense.hpp"
#include "mlp/io.hpp"
#include "model.hpp"

int main() {
  std::mt19937 rng(123);

  mlp::Sequential model;
  model.add(std::make_unique<mlp::Dense>(2, 8, rng));
  model.add(std::make_unique<mlp::Tanh>());
  model.add(std::make_unique<mlp::Dense>(8, 1, rng));
  model.add(std::make_unique<mlp::Sigmoid>());

  const mlp::Matrix x = {
      {0.1, 0.2},
      {0.7, 0.8},
      {0.4, 0.9},
      {0.2, 0.6},
  };

  const mlp::Matrix y_before = model.forward(x);

  const std::string path = "test_model_checkpoint.txt";
  mlp::save_sequential(model, path);
  mlp::Sequential loaded = mlp::load_sequential(path);

  const mlp::Matrix y_after = loaded.forward(x);

  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < y_before.size(); ++i) {
    for (std::size_t j = 0; j < y_before[i].size(); ++j) {
      const double d = std::fabs(y_before[i][j] - y_after[i][j]);
      if (d > max_abs_diff) max_abs_diff = d;
    }
  }

  if (max_abs_diff > 1e-6) {
    std::cerr << "roundtrip mismatch too large: " << max_abs_diff << "\n";
    return 1;
  }

  const std::string malformed_path = "test_model_malformed_checkpoint.txt";
  {
    std::ofstream os(malformed_path);
    os << "MLPSEQv1\n1\nDense 2 2\n"
       << "0.1 nan\n"
       << "0.2 0.3\n"
       << "0.0 0.0\n";
  }

  try {
    (void)mlp::load_sequential(malformed_path);
    std::cerr << "expected malformed checkpoint exception\n";
    return 1;
  } catch (const std::exception &) {
  }

  const std::string oversized_path = "test_model_oversized_checkpoint.txt";
  {
    std::ofstream os(oversized_path);
    os << "MLPSEQv1\n1\nDense 100000001 1\n";
  }

  try {
    (void)mlp::load_sequential(oversized_path);
    std::cerr << "expected oversized checkpoint exception\n";
    return 1;
  } catch (const std::length_error &) {
  }

  std::cout << "ok max_abs_diff=" << max_abs_diff << "\n";
  return 0;
}
