#include <iostream>

#include "mlp/library.hpp"

int main() {
  mlp::ExperimentOptions opt;
  opt.optimizer = "adam";
  opt.hidden = {16, 16};
  opt.epochs = 1500;
  opt.samples = 800;
  opt.train_ratio = 0.7;
  opt.val_ratio = 0.15;

  const mlp::ExperimentReport report = mlp::run_xor_experiment(opt, &std::cout);

  std::cout << "\nLibrary call summary: test_loss=" << report.test.loss
            << " test_acc=" << report.test.metrics.accuracy << "\n";
  return 0;
}
