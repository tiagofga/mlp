#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <functional>
#include <unordered_map>

#include "model.hpp"

namespace mlp {

class Optimizer {
 public:
  virtual ~Optimizer() = default;
  virtual void step(Sequential &model) = 0;

 protected:
  std::size_t step_count_ = 0;
};

class SGD : public Optimizer {
 public:
  explicit SGD(double learning_rate) : learning_rate_(learning_rate) {}
  void step(Sequential &model) override;

 private:
  double learning_rate_;
};

class Momentum : public Optimizer {
 public:
  Momentum(double learning_rate, double beta = 0.9) : learning_rate_(learning_rate), beta_(beta) {}
  void step(Sequential &model) override;

 private:
  double learning_rate_;
  double beta_;
  std::unordered_map<const void *, Matrix> velocity_m_;
  std::unordered_map<const void *, Vector> velocity_v_;
};

class Adam : public Optimizer {
 public:
  Adam(double learning_rate = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}
  void step(Sequential &model) override;

 protected:
  double learning_rate_;
  double beta1_;
  double beta2_;
  double epsilon_;
  std::unordered_map<const void *, Matrix> first_m_m_;
  std::unordered_map<const void *, Matrix> second_m_m_;
  std::unordered_map<const void *, Vector> first_m_v_;
  std::unordered_map<const void *, Vector> second_m_v_;
};

class AdamW : public Adam {
 public:
  AdamW(double learning_rate = 1e-3, double weight_decay = 1e-2, double beta1 = 0.9, double beta2 = 0.999,
        double epsilon = 1e-8, bool decay_bias = false)
      : Adam(learning_rate, beta1, beta2, epsilon), weight_decay_(weight_decay), decay_bias_(decay_bias) {}
  void step(Sequential &model) override;

 private:
  double weight_decay_;
  bool decay_bias_;
};

class LambdaOptimizer : public Optimizer {
 public:
  using MatrixRule = std::function<void(Matrix &, const Matrix &, std::size_t)>;
  using VectorRule = std::function<void(Vector &, const Vector &, std::size_t)>;

  LambdaOptimizer(MatrixRule matrix_rule, VectorRule vector_rule = nullptr)
      : matrix_rule_(std::move(matrix_rule)), vector_rule_(std::move(vector_rule)) {}
  void step(Sequential &model) override;

 private:
  MatrixRule matrix_rule_;
  VectorRule vector_rule_;
};

}  // namespace mlp

#endif
