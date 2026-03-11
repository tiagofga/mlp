#include <cmath>
#include <iostream>
#include <memory>
#include <random>

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

  mlp::Matrix x = {
      {0.1, 0.2},
      {0.7, 0.8},
      {0.4, 0.9},
  };

  const mlp::Matrix y_before = model.forward(x);

  const std::string path = "model_checkpoint.txt";
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

  std::cout << "Saved model to: " << path << "\n";
  std::cout << "Max absolute prediction difference after reload: " << max_abs_diff << "\n";

  return 0;
}
