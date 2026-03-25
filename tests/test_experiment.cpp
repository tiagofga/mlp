#include <cmath>
#include <iostream>

#include "mlp/library.hpp"

static bool run_test(const std::string &optimizer_name, double loss_threshold) {
  mlp::ExperimentOptions opt;
  opt.optimizer = optimizer_name;
  opt.hidden = {16, 16};
  opt.epochs = 600;
  opt.samples = 600;
  opt.train_ratio = 0.7;
  opt.val_ratio = 0.15;

  const mlp::ExperimentReport rep = mlp::run_xor_experiment(opt, nullptr);

  if (rep.train.samples == 0 || rep.val.samples == 0 || rep.test.samples == 0) {
    std::cerr << "[" << optimizer_name << "] invalid split sizes\n";
    return false;
  }
  if (!std::isfinite(rep.train.loss) || !std::isfinite(rep.val.loss) || !std::isfinite(rep.test.loss)) {
    std::cerr << "[" << optimizer_name << "] non-finite loss\n";
    return false;
  }
  if (rep.test.metrics.accuracy < 0.0 || rep.test.metrics.accuracy > 1.0) {
    std::cerr << "[" << optimizer_name << "] invalid accuracy range\n";
    return false;
  }
  if (rep.train.loss > loss_threshold) {
    std::cerr << "[" << optimizer_name << "] train loss did not converge enough: " << rep.train.loss << "\n";
    return false;
  }

  std::cout << "[" << optimizer_name << "] ok train_loss=" << rep.train.loss
            << " test_acc=" << rep.test.metrics.accuracy << "\n";
  return true;
}

int main() {
  bool ok = true;
  ok = run_test("adam", 0.10) && ok;
  ok = run_test("rmsprop", 0.10) && ok;
  return ok ? 0 : 1;
}
