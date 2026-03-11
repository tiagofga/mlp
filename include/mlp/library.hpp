#ifndef MLP_LIBRARY_HPP
#define MLP_LIBRARY_HPP

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "mlp/metrics.hpp"

namespace mlp {

struct ExperimentOptions {
  std::string optimizer = "sgd";
  std::vector<std::size_t> hidden = {8};
  int epochs = 5000;
  std::size_t samples = 400;
  unsigned int seed = 42;
  double learning_rate = -1.0;  // negative means optimizer default
  double train_ratio = 0.7;
  double val_ratio = 0.15;
  double threshold = 0.5;
};

struct SplitReport {
  std::size_t samples = 0;
  double loss = 0.0;
  BinaryMetrics metrics;
};

struct ExperimentReport {
  SplitReport train;
  SplitReport val;
  SplitReport test;
};

ExperimentReport run_xor_experiment(const ExperimentOptions &options, std::ostream *log_stream = nullptr,
                                    std::size_t log_every = 500);

std::string hidden_to_string(const std::vector<std::size_t> &hidden_sizes);

}  // namespace mlp

#endif
