#include <cmath>
#include <iostream>

#include "mlp/library.hpp"

int main() {
  mlp::ExperimentOptions opt;
  opt.optimizer = "adam";
  opt.hidden = {16, 16};
  opt.epochs = 600;
  opt.samples = 600;
  opt.train_ratio = 0.7;
  opt.val_ratio = 0.15;

  const mlp::ExperimentReport rep = mlp::run_xor_experiment(opt, nullptr);

  if (rep.train.samples == 0 || rep.val.samples == 0 || rep.test.samples == 0) {
    std::cerr << "invalid split sizes\n";
    return 1;
  }
  if (!std::isfinite(rep.train.loss) || !std::isfinite(rep.val.loss) || !std::isfinite(rep.test.loss)) {
    std::cerr << "non-finite loss\n";
    return 1;
  }
  if (rep.test.metrics.accuracy < 0.0 || rep.test.metrics.accuracy > 1.0) {
    std::cerr << "invalid accuracy range\n";
    return 1;
  }

  // Ensure the model is learning on this synthetic problem.
  if (rep.train.loss > 0.10) {
    std::cerr << "train loss did not converge enough: " << rep.train.loss << "\n";
    return 1;
  }

  std::cout << "ok train_loss=" << rep.train.loss << " test_acc=" << rep.test.metrics.accuracy << "\n";
  return 0;
}
