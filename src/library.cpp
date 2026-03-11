#include "mlp/library.hpp"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include "activations.hpp"
#include "dense.hpp"
#include "loss.hpp"
#include "model.hpp"
#include "optimizer.hpp"

namespace mlp {
namespace {

struct Dataset {
  Matrix x;
  Matrix y;
};

std::unique_ptr<Optimizer> build_optimizer(const std::string &name, double learning_rate) {
  const bool has_custom_lr = learning_rate > 0.0;
  if (name == "sgd") return std::make_unique<SGD>(has_custom_lr ? learning_rate : 0.8);
  if (name == "momentum") return std::make_unique<Momentum>(has_custom_lr ? learning_rate : 0.2, 0.9);
  if (name == "adam") return std::make_unique<Adam>(has_custom_lr ? learning_rate : 0.05);
  if (name == "adamw") return std::make_unique<AdamW>(has_custom_lr ? learning_rate : 0.05, 1e-2);
  throw std::invalid_argument("Unknown optimizer: " + name);
}

Dataset make_xor_dataset(std::size_t samples, std::mt19937 &rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Dataset d;
  d.x.reserve(samples);
  d.y.reserve(samples);

  for (std::size_t i = 0; i < samples; ++i) {
    const double a = dist(rng);
    const double b = dist(rng);
    const int label = ((a > 0.5) != (b > 0.5)) ? 1 : 0;

    d.x.push_back({a, b});
    d.y.push_back({static_cast<double>(label)});
  }

  return d;
}

Dataset select_rows(const Dataset &d, std::size_t begin, std::size_t end) {
  Dataset out;
  out.x.reserve(end - begin);
  out.y.reserve(end - begin);

  for (std::size_t i = begin; i < end; ++i) {
    out.x.push_back(d.x[i]);
    out.y.push_back(d.y[i]);
  }

  return out;
}

void log_metrics(std::ostream &os, const std::string &name, const BinaryMetrics &m) {
  os << name << " metrics"
     << " | acc=" << std::fixed << std::setprecision(4) << m.accuracy << " | precision=" << m.precision
     << " | recall=" << m.recall << " | f1=" << m.f1 << " | tp=" << m.tp << " tn=" << m.tn
     << " fp=" << m.fp << " fn=" << m.fn << "\n";
}

}  // namespace

std::string hidden_to_string(const std::vector<std::size_t> &hidden_sizes) {
  if (hidden_sizes.empty()) return "(none)";
  std::ostringstream os;
  for (std::size_t i = 0; i < hidden_sizes.size(); ++i) {
    if (i > 0) os << ",";
    os << hidden_sizes[i];
  }
  return os.str();
}

ExperimentReport run_xor_experiment(const ExperimentOptions &opt, std::ostream *log_stream, std::size_t log_every) {
  if (opt.samples < 3) {
    throw std::invalid_argument("samples must be at least 3");
  }
  if (opt.hidden.empty()) {
    throw std::invalid_argument("hidden must contain at least one layer size");
  }
  if (opt.epochs <= 0) {
    throw std::invalid_argument("epochs must be positive");
  }
  if (opt.train_ratio <= 0.0 || opt.val_ratio <= 0.0 || opt.train_ratio + opt.val_ratio >= 1.0) {
    throw std::invalid_argument("ratios must satisfy train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1");
  }
  if (opt.threshold <= 0.0 || opt.threshold >= 1.0) {
    throw std::invalid_argument("threshold must be in (0,1)");
  }

  std::unique_ptr<Optimizer> optimizer = build_optimizer(opt.optimizer, opt.learning_rate);

  std::mt19937 rng(opt.seed);
  Dataset all = make_xor_dataset(opt.samples, rng);

  std::vector<std::size_t> idx(opt.samples);
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);

  Dataset shuffled;
  shuffled.x.reserve(all.x.size());
  shuffled.y.reserve(all.y.size());
  for (std::size_t i : idx) {
    shuffled.x.push_back(all.x[i]);
    shuffled.y.push_back(all.y[i]);
  }

  std::size_t train_end = static_cast<std::size_t>(static_cast<double>(opt.samples) * opt.train_ratio);
  std::size_t val_end = train_end + static_cast<std::size_t>(static_cast<double>(opt.samples) * opt.val_ratio);

  train_end = std::max<std::size_t>(1, train_end);
  if (train_end >= opt.samples - 1) train_end = opt.samples - 2;

  val_end = std::max(train_end + 1, val_end);
  if (val_end >= opt.samples) val_end = opt.samples - 1;

  const Dataset train = select_rows(shuffled, 0, train_end);
  const Dataset val = select_rows(shuffled, train_end, val_end);
  const Dataset test = select_rows(shuffled, val_end, opt.samples);

  Sequential model;
  std::size_t in_features = 2;
  for (std::size_t hidden : opt.hidden) {
    model.add(std::make_unique<Dense>(in_features, hidden, rng));
    model.add(std::make_unique<Tanh>());
    in_features = hidden;
  }
  model.add(std::make_unique<Dense>(in_features, 1, rng));
  model.add(std::make_unique<Sigmoid>());

  BinaryCrossEntropy train_loss_fn;
  BinaryCrossEntropy eval_loss_fn;

  if (log_stream != nullptr) {
    *log_stream << "Optimizer: " << opt.optimizer << "\n";
    *log_stream << "Hidden layers: " << hidden_to_string(opt.hidden) << "\n";
    *log_stream << "Samples: " << opt.samples << " | split train/val/test = " << train.x.size() << "/"
                << val.x.size() << "/" << test.x.size() << "\n";
    *log_stream << "Epochs: " << opt.epochs << " | threshold: " << opt.threshold << "\n";
  }

  for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
    const Matrix pred_train = model.forward(train.x);
    const double train_loss = train_loss_fn.forward(pred_train, train.y);

    const Matrix grad_loss = train_loss_fn.backward();
    model.backward(grad_loss);
    optimizer->step(model);

    if (log_stream != nullptr && (epoch == 1 || epoch == opt.epochs || epoch % static_cast<int>(log_every) == 0)) {
      const Matrix pred_val = model.forward(val.x);
      const double val_loss = eval_loss_fn.forward(pred_val, val.y);
      *log_stream << "Epoch " << std::setw(5) << epoch << " | train_loss=" << std::fixed
                  << std::setprecision(6) << train_loss << " | val_loss=" << val_loss << "\n";
    }
  }

  const Matrix pred_train = model.forward(train.x);
  const Matrix pred_val = model.forward(val.x);
  const Matrix pred_test = model.forward(test.x);

  ExperimentReport report;
  report.train.samples = train.x.size();
  report.val.samples = val.x.size();
  report.test.samples = test.x.size();

  report.train.loss = eval_loss_fn.forward(pred_train, train.y);
  report.val.loss = eval_loss_fn.forward(pred_val, val.y);
  report.test.loss = eval_loss_fn.forward(pred_test, test.y);

  report.train.metrics = compute_binary_metrics(pred_train, train.y, opt.threshold);
  report.val.metrics = compute_binary_metrics(pred_val, val.y, opt.threshold);
  report.test.metrics = compute_binary_metrics(pred_test, test.y, opt.threshold);

  if (log_stream != nullptr) {
    *log_stream << "\nFinal losses"
                << " | train=" << std::fixed << std::setprecision(6) << report.train.loss
                << " | val=" << report.val.loss << " | test=" << report.test.loss << "\n";
    log_metrics(*log_stream, "Train", report.train.metrics);
    log_metrics(*log_stream, "Val  ", report.val.metrics);
    log_metrics(*log_stream, "Test ", report.test.metrics);
  }

  return report;
}

}  // namespace mlp
