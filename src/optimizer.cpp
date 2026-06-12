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

template <typename MatrixRule, typename VectorRule>
void for_each_parameter(Sequential &model, MatrixRule matrix_rule, VectorRule vector_rule) {
  for (auto &layer : model.layers()) {
    for (auto &param : layer->matrix_params()) {
      matrix_rule(param);
    }

    for (auto &param : layer->vector_params()) {
      vector_rule(param);
    }
  }
}

Matrix &state_for(std::unordered_map<const void *, Matrix> &states, Matrix &param) {
  auto &state = states[static_cast<const void *>(&param)];
  if (state.empty()) state = zeros_like(param);
  return state;
}

Vector &state_for(std::unordered_map<const void *, Vector> &states, Vector &param) {
  auto &state = states[static_cast<const void *>(&param)];
  if (state.empty()) state = Vector(param.size(), 0.0);
  return state;
}

template <typename Rule>
void update_matrix(MatrixParamRef param, Rule rule) {
  for (std::size_t i = 0; i < rows(*param.value); ++i) {
    for (std::size_t j = 0; j < cols(*param.value); ++j) {
      rule((*param.value)[i][j], (*param.grad)[i][j], i, j);
    }
  }
}

template <typename Rule>
void update_vector(VectorParamRef param, Rule rule) {
  for (std::size_t i = 0; i < param.value->size(); ++i) {
    rule((*param.value)[i], (*param.grad)[i], i);
  }
}

}  // namespace

void SGD::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        sgd_update(*param.value, *param.grad, learning_rate_);
      },
      [this](VectorParamRef param) {
        sgd_update(*param.value, *param.grad, learning_rate_);
      });
}

void Momentum::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        Matrix &velocity = state_for(velocity_m_, *param.value);
        update_matrix(param, [this, &velocity](double &value, double grad, std::size_t i, std::size_t j) {
          velocity[i][j] = beta_ * velocity[i][j] - learning_rate_ * grad;
          value += velocity[i][j];
        });
      },
      [this](VectorParamRef param) {
        Vector &velocity = state_for(velocity_v_, *param.value);
        update_vector(param, [this, &velocity](double &value, double grad, std::size_t i) {
          velocity[i] = beta_ * velocity[i] - learning_rate_ * grad;
          value += velocity[i];
        });
      });
}

void Adam::step(Sequential &model) {
  ++step_count_;
  const double bias_c1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bias_c2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for_each_parameter(
      model,
      [this, bias_c1, bias_c2](MatrixParamRef param) {
        Matrix &m = state_for(first_m_m_, *param.value);
        Matrix &v = state_for(second_m_m_, *param.value);
        update_matrix(param, [this, bias_c1, bias_c2, &m, &v](double &value, double g, std::size_t i, std::size_t j) {
          m[i][j] = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          v[i][j] = beta2_ * v[i][j] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i][j] / bias_c1;
          const double v_hat = v[i][j] / bias_c2;
          value -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        });
      },
      [this, bias_c1, bias_c2](VectorParamRef param) {
        Vector &m = state_for(first_m_v_, *param.value);
        Vector &v = state_for(second_m_v_, *param.value);
        update_vector(param, [this, bias_c1, bias_c2, &m, &v](double &value, double g, std::size_t i) {
          m[i] = beta1_ * m[i] + (1.0 - beta1_) * g;
          v[i] = beta2_ * v[i] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i] / bias_c1;
          const double v_hat = v[i] / bias_c2;
          value -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        });
      });
}

void AdamW::step(Sequential &model) {
  ++step_count_;
  const double bias_c1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bias_c2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for_each_parameter(
      model,
      [this, bias_c1, bias_c2](MatrixParamRef param) {
        apply_weight_decay(*param.value, learning_rate_, weight_decay_);
        Matrix &m = state_for(first_m_m_, *param.value);
        Matrix &v = state_for(second_m_m_, *param.value);
        update_matrix(param, [this, bias_c1, bias_c2, &m, &v](double &value, double g, std::size_t i, std::size_t j) {
          m[i][j] = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          v[i][j] = beta2_ * v[i][j] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i][j] / bias_c1;
          const double v_hat = v[i][j] / bias_c2;
          value -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        });
      },
      [this, bias_c1, bias_c2](VectorParamRef param) {
        if (decay_bias_) apply_weight_decay(*param.value, learning_rate_, weight_decay_);
        Vector &m = state_for(first_m_v_, *param.value);
        Vector &v = state_for(second_m_v_, *param.value);
        update_vector(param, [this, bias_c1, bias_c2, &m, &v](double &value, double g, std::size_t i) {
          m[i] = beta1_ * m[i] + (1.0 - beta1_) * g;
          v[i] = beta2_ * v[i] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i] / bias_c1;
          const double v_hat = v[i] / bias_c2;
          value -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        });
      });
}

void RMSProp::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        Matrix &cache = state_for(cache_m_, *param.value);
        update_matrix(param, [this, &cache](double &value, double g, std::size_t i, std::size_t j) {
          cache[i][j] = rho_ * cache[i][j] + (1.0 - rho_) * g * g;
          value -= learning_rate_ * g / (std::sqrt(cache[i][j]) + epsilon_);
        });
      },
      [this](VectorParamRef param) {
        Vector &cache = state_for(cache_v_, *param.value);
        update_vector(param, [this, &cache](double &value, double g, std::size_t i) {
          cache[i] = rho_ * cache[i] + (1.0 - rho_) * g * g;
          value -= learning_rate_ * g / (std::sqrt(cache[i]) + epsilon_);
        });
      });
}

void NAG::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        Matrix &v = state_for(velocity_m_, *param.value);
        update_matrix(param, [this, &v](double &value, double g, std::size_t i, std::size_t j) {
          v[i][j] = beta_ * v[i][j] + g;
          value -= learning_rate_ * (g + beta_ * v[i][j]);
        });
      },
      [this](VectorParamRef param) {
        Vector &v = state_for(velocity_v_, *param.value);
        update_vector(param, [this, &v](double &value, double g, std::size_t i) {
          v[i] = beta_ * v[i] + g;
          value -= learning_rate_ * (g + beta_ * v[i]);
        });
      });
}

void AdaGrad::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        Matrix &cache = state_for(cache_m_, *param.value);
        update_matrix(param, [this, &cache](double &value, double g, std::size_t i, std::size_t j) {
          cache[i][j] += g * g;
          value -= learning_rate_ * g / (std::sqrt(cache[i][j]) + epsilon_);
        });
      },
      [this](VectorParamRef param) {
        Vector &cache = state_for(cache_v_, *param.value);
        update_vector(param, [this, &cache](double &value, double g, std::size_t i) {
          cache[i] += g * g;
          value -= learning_rate_ * g / (std::sqrt(cache[i]) + epsilon_);
        });
      });
}

void Nadam::step(Sequential &model) {
  ++step_count_;
  const double bias_c1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bias_c2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for_each_parameter(
      model,
      [this, bias_c1, bias_c2](MatrixParamRef param) {
        Matrix &m = state_for(first_m_m_, *param.value);
        Matrix &v = state_for(second_m_m_, *param.value);
        update_matrix(param, [this, bias_c1, bias_c2, &m, &v](double &value, double g, std::size_t i, std::size_t j) {
          m[i][j] = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          v[i][j] = beta2_ * v[i][j] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i][j] / bias_c1;
          const double v_hat = v[i][j] / bias_c2;
          const double nesterov_m_hat = beta1_ * m_hat + (1.0 - beta1_) * g / bias_c1;
          value -= learning_rate_ * nesterov_m_hat / (std::sqrt(v_hat) + epsilon_);
        });
      },
      [this, bias_c1, bias_c2](VectorParamRef param) {
        Vector &m = state_for(first_m_v_, *param.value);
        Vector &v = state_for(second_m_v_, *param.value);
        update_vector(param, [this, bias_c1, bias_c2, &m, &v](double &value, double g, std::size_t i) {
          m[i] = beta1_ * m[i] + (1.0 - beta1_) * g;
          v[i] = beta2_ * v[i] + (1.0 - beta2_) * g * g;

          const double m_hat = m[i] / bias_c1;
          const double v_hat = v[i] / bias_c2;
          const double nesterov_m_hat = beta1_ * m_hat + (1.0 - beta1_) * g / bias_c1;
          value -= learning_rate_ * nesterov_m_hat / (std::sqrt(v_hat) + epsilon_);
        });
      });
}

void AdaDelta::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        Matrix &eg2 = state_for(eg2_m_, *param.value);
        Matrix &edx2 = state_for(edx2_m_, *param.value);
        update_matrix(param, [this, &eg2, &edx2](double &value, double g, std::size_t i, std::size_t j) {
          eg2[i][j] = rho_ * eg2[i][j] + (1.0 - rho_) * g * g;
          const double rms_g = std::sqrt(eg2[i][j] + epsilon_);
          const double rms_dx = std::sqrt(edx2[i][j] + epsilon_);
          const double dx = -(rms_dx / rms_g) * g;
          edx2[i][j] = rho_ * edx2[i][j] + (1.0 - rho_) * dx * dx;
          value += dx;
        });
      },
      [this](VectorParamRef param) {
        Vector &eg2 = state_for(eg2_v_, *param.value);
        Vector &edx2 = state_for(edx2_v_, *param.value);
        update_vector(param, [this, &eg2, &edx2](double &value, double g, std::size_t i) {
          eg2[i] = rho_ * eg2[i] + (1.0 - rho_) * g * g;
          const double rms_g = std::sqrt(eg2[i] + epsilon_);
          const double rms_dx = std::sqrt(edx2[i] + epsilon_);
          const double dx = -(rms_dx / rms_g) * g;
          edx2[i] = rho_ * edx2[i] + (1.0 - rho_) * dx * dx;
          value += dx;
        });
      });
}

void Lion::step(Sequential &model) {
  ++step_count_;
  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        Matrix &m = state_for(momentum_m_, *param.value);
        update_matrix(param, [this, &m](double &value, double g, std::size_t i, std::size_t j) {
          const double update = beta1_ * m[i][j] + (1.0 - beta1_) * g;
          const double sign_update = (update > 0.0) - (update < 0.0);
          m[i][j] = beta2_ * m[i][j] + (1.0 - beta2_) * g;
          value -= learning_rate_ * sign_update;
        });
      },
      [this](VectorParamRef param) {
        Vector &m = state_for(momentum_v_, *param.value);
        update_vector(param, [this, &m](double &value, double g, std::size_t i) {
          const double update = beta1_ * m[i] + (1.0 - beta1_) * g;
          const double sign_update = (update > 0.0) - (update < 0.0);
          m[i] = beta2_ * m[i] + (1.0 - beta2_) * g;
          value -= learning_rate_ * sign_update;
        });
      });
}

void LambdaOptimizer::step(Sequential &model) {
  ++step_count_;
  if (!matrix_rule_) {
    throw std::invalid_argument("LambdaOptimizer requires a matrix update rule");
  }

  for_each_parameter(
      model,
      [this](MatrixParamRef param) {
        matrix_rule_(*param.value, *param.grad, step_count_);
      },
      [this](VectorParamRef param) {
        if (vector_rule_) vector_rule_(*param.value, *param.grad, step_count_);
      });
}

}  // namespace mlp
