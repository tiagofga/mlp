#include "optimizer.hpp"

#include <cmath>
#include <stdexcept>

#include "matrix.hpp"

namespace mlp {

namespace {

void sgd_update(Matrix &param, const Matrix &grad, double lr) {
  for (std::size_t i = 0; i < rows(param); ++i) {
    for (std::size_t j = 0; j < cols(param); ++j) {
      param[i][j] -= lr * grad[i][j];
    }
  }
}

void sgd_update(Vector &param, const Vector &grad, double lr) {
  for (std::size_t i = 0; i < param.size(); ++i) {
    param[i] -= lr * grad[i];
  }
}

void apply_weight_decay(Matrix &param, double lr, double weight_decay) {
  const double scale = 1.0 - lr * weight_decay;
  for (std::size_t i = 0; i < rows(param); ++i) {
    for (std::size_t j = 0; j < cols(param); ++j) {
      param[i][j] *= scale;
    }
  }
}

void apply_weight_decay(Vector &param, double lr, double weight_decay) {
  const double scale = 1.0 - lr * weight_decay;
  for (double &v : param) {
    v *= scale;
  }
}

}  // namespace

void SGD::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      sgd_update(*param.value, *param.grad, learning_rate_);
    }
    for (auto &param : layer->vector_params()) {
      sgd_update(*param.value, *param.grad, learning_rate_);
    }
  }
}

void Momentum::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &velocity = velocity_m_[key];
      if (velocity.empty()) velocity = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          velocity[i][j] = beta_ * velocity[i][j] - learning_rate_ * (*param.grad)[i][j];
          (*param.value)[i][j] += velocity[i][j];
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &velocity = velocity_v_[key];
      if (velocity.empty()) velocity = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        velocity[i] = beta_ * velocity[i] - learning_rate_ * (*param.grad)[i];
        (*param.value)[i] += velocity[i];
      }
    }
  }
}

void Adam::step(Sequential &model) {
  ++step_count_;
  const double bias_c1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bias_c2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &m = first_m_m_[key];
      auto &v = second_m_m_[key];
      if (m.empty()) m = zeros_like(*param.value);
      if (v.empty()) v = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          m[i][j] = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          v[i][j] = beta2_ * v[i][j] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i][j] / bias_c1;
          const double v_hat = v[i][j] / bias_c2;
          (*param.value)[i][j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &m = first_m_v_[key];
      auto &v = second_m_v_[key];
      if (m.empty()) m = Vector(param.value->size(), 0.0);
      if (v.empty()) v = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        m[i] = beta1_ * m[i] + (1.0 - beta1_) * g;
        v[i] = beta2_ * v[i] + (1.0 - beta2_) * g * g;

        const double m_hat = m[i] / bias_c1;
        const double v_hat = v[i] / bias_c2;
        (*param.value)[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
    }
  }
}

void AdamW::step(Sequential &model) {  ++step_count_;
  const double bias_c1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bias_c2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      apply_weight_decay(*param.value, learning_rate_, weight_decay_);

      const void *key = static_cast<const void *>(param.value);
      auto &m = first_m_m_[key];
      auto &v = second_m_m_[key];
      if (m.empty()) m = zeros_like(*param.value);
      if (v.empty()) v = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          m[i][j] = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          v[i][j] = beta2_ * v[i][j] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i][j] / bias_c1;
          const double v_hat = v[i][j] / bias_c2;
          (*param.value)[i][j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      if (decay_bias_) apply_weight_decay(*param.value, learning_rate_, weight_decay_);

      const void *key = static_cast<const void *>(param.value);
      auto &m = first_m_v_[key];
      auto &v = second_m_v_[key];
      if (m.empty()) m = Vector(param.value->size(), 0.0);
      if (v.empty()) v = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        m[i] = beta1_ * m[i] + (1.0 - beta1_) * g;
        v[i] = beta2_ * v[i] + (1.0 - beta2_) * g * g;

        const double m_hat = m[i] / bias_c1;
        const double v_hat = v[i] / bias_c2;
        (*param.value)[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
    }
  }
}

void RMSProp::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &cache = cache_m_[key];
      if (cache.empty()) cache = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          cache[i][j] = rho_ * cache[i][j] + (1.0 - rho_) * g * g;
          (*param.value)[i][j] -= learning_rate_ * g / (std::sqrt(cache[i][j]) + epsilon_);
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &cache = cache_v_[key];
      if (cache.empty()) cache = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        cache[i] = rho_ * cache[i] + (1.0 - rho_) * g * g;
        (*param.value)[i] -= learning_rate_ * g / (std::sqrt(cache[i]) + epsilon_);
      }
    }
  }
}

void NAG::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &v = velocity_m_[key];
      if (v.empty()) v = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          v[i][j] = beta_ * v[i][j] + g;
          (*param.value)[i][j] -= learning_rate_ * (g + beta_ * v[i][j]);
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &v = velocity_v_[key];
      if (v.empty()) v = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        v[i] = beta_ * v[i] + g;
        (*param.value)[i] -= learning_rate_ * (g + beta_ * v[i]);
      }
    }
  }
}

void AdaGrad::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &cache = cache_m_[key];
      if (cache.empty()) cache = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          cache[i][j] += g * g;
          (*param.value)[i][j] -= learning_rate_ * g / (std::sqrt(cache[i][j]) + epsilon_);
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &cache = cache_v_[key];
      if (cache.empty()) cache = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        cache[i] += g * g;
        (*param.value)[i] -= learning_rate_ * g / (std::sqrt(cache[i]) + epsilon_);
      }
    }
  }
}

void Nadam::step(Sequential &model) {
  ++step_count_;
  const double bias_c1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bias_c2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &m = first_m_m_[key];
      auto &v = second_m_m_[key];
      if (m.empty()) m = zeros_like(*param.value);
      if (v.empty()) v = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          m[i][j] = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          v[i][j] = beta2_ * v[i][j] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i][j] / bias_c1;
          const double v_hat = v[i][j] / bias_c2;
          const double nesterov_m_hat = beta1_ * m_hat + (1.0 - beta1_) * g / bias_c1;
          (*param.value)[i][j] -= learning_rate_ * nesterov_m_hat / (std::sqrt(v_hat) + epsilon_);
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &m = first_m_v_[key];
      auto &v = second_m_v_[key];
      if (m.empty()) m = Vector(param.value->size(), 0.0);
      if (v.empty()) v = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        m[i] = beta1_ * m[i] + (1.0 - beta1_) * g;
        v[i] = beta2_ * v[i] + (1.0 - beta2_) * g * g;

        const double m_hat = m[i] / bias_c1;
        const double v_hat = v[i] / bias_c2;
        const double nesterov_m_hat = beta1_ * m_hat + (1.0 - beta1_) * g / bias_c1;
        (*param.value)[i] -= learning_rate_ * nesterov_m_hat / (std::sqrt(v_hat) + epsilon_);
      }
    }
  }
}

void AdaDelta::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &eg2 = eg2_m_[key];
      auto &edx2 = edx2_m_[key];
      if (eg2.empty()) eg2 = zeros_like(*param.value);
      if (edx2.empty()) edx2 = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          eg2[i][j] = rho_ * eg2[i][j] + (1.0 - rho_) * g * g;
          const double rms_g = std::sqrt(eg2[i][j] + epsilon_);
          const double rms_dx = std::sqrt(edx2[i][j] + epsilon_);
          const double dx = -(rms_dx / rms_g) * g;
          edx2[i][j] = rho_ * edx2[i][j] + (1.0 - rho_) * dx * dx;
          (*param.value)[i][j] += dx;
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &eg2 = eg2_v_[key];
      auto &edx2 = edx2_v_[key];
      if (eg2.empty()) eg2 = Vector(param.value->size(), 0.0);
      if (edx2.empty()) edx2 = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        eg2[i] = rho_ * eg2[i] + (1.0 - rho_) * g * g;
        const double rms_g = std::sqrt(eg2[i] + epsilon_);
        const double rms_dx = std::sqrt(edx2[i] + epsilon_);
        const double dx = -(rms_dx / rms_g) * g;
        edx2[i] = rho_ * edx2[i] + (1.0 - rho_) * dx * dx;
        (*param.value)[i] += dx;
      }
    }
  }
}

void Lion::step(Sequential &model) {
  ++step_count_;
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &m = momentum_m_[key];
      if (m.empty()) m = zeros_like(*param.value);

      for (std::size_t i = 0; i < rows(*param.value); ++i) {
        for (std::size_t j = 0; j < cols(*param.value); ++j) {
          const double g = (*param.grad)[i][j];
          const double update = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          const double sign_update = (update > 0.0) - (update < 0.0);
          m[i][j] = beta2_ * m[i][j] + (1.0 - beta2_) * g;
          (*param.value)[i][j] -= learning_rate_ * sign_update;
        }
      }
    }

    for (auto &param : layer->vector_params()) {
      const void *key = static_cast<const void *>(param.value);
      auto &m = momentum_v_[key];
      if (m.empty()) m = Vector(param.value->size(), 0.0);

      for (std::size_t i = 0; i < param.value->size(); ++i) {
        const double g = (*param.grad)[i];
        const double update = beta1_ * m[i] + (1.0 - beta1_) * g;
        const double sign_update = (update > 0.0) - (update < 0.0);
        m[i] = beta2_ * m[i] + (1.0 - beta2_) * g;
        (*param.value)[i] -= learning_rate_ * sign_update;
      }
    }
  }
}

void LambdaOptimizer::step(Sequential &model) {
  ++step_count_;
  if (!matrix_rule_) {
    throw std::invalid_argument("LambdaOptimizer requires a matrix update rule");
  }

  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      matrix_rule_(*param.value, *param.grad, step_count_);
    }

    for (auto &param : layer->vector_params()) {
      if (vector_rule_) vector_rule_(*param.value, *param.grad, step_count_);
    }
  }
}

}  // namespace mlp
