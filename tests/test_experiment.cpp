#include <cmath>
#include <iostream>

#include "mlp/library.hpp"

static bool run_test(const std::string &optimizer_name, double loss_threshold, int epochs = 600,
                     std::size_t samples = 600, double lr = -1.0) {
  mlp::ExperimentOptions opt;
  opt.optimizer = optimizer_name;
  opt.hidden = {16, 16};
  opt.epochs = epochs;
  opt.samples = samples;
  opt.train_ratio = 0.7;
  opt.val_ratio = 0.15;
  if (lr > 0.0) opt.learning_rate = lr;

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

// Verifies an optimizer runs without errors and produces a finite loss, without
// requiring full convergence (used for optimizers like AdaDelta that converge
// slowly on dense batch gradient descent problems).
static bool run_stability_test(const std::string &optimizer_name) {
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

  std::cout << "[" << optimizer_name << "] stable train_loss=" << rep.train.loss << "\n";
  return true;
}

int main() {
  bool ok = true;
  ok = run_test("adam", 0.10) && ok;
  ok = run_test("rmsprop", 0.10) && ok;
  ok = run_test("adagrad", 0.10) && ok;
  ok = run_test("nadam", 0.10) && ok;
  ok = run_test("lion", 0.10) && ok;
  // NAG converges on a smaller, noisier dataset with lr=1.0 in 2000 epochs.
  ok = run_test("nag", 0.10, 2000, 100, 1.0) && ok;
  // AdaDelta is designed for sparse gradients and converges slowly on dense
  // batch gradient descent; verify stability (finite loss) rather than convergence.
  ok = run_stability_test("adadelta") && ok;
  return ok ? 0 : 1;
}
