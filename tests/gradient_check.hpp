#ifndef TESTS_GRADIENT_CHECK_HPP
#define TESTS_GRADIENT_CHECK_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "layer.hpp"

namespace gradient_check {

struct Options {
  double epsilon = 1e-6;
  double relative_tolerance = 1e-5;
  double parameter_objective_scale = 1.0;
};

inline double contract(const mlp::Matrix &lhs, const mlp::Matrix &rhs) {
  mlp::check_same_shape(lhs, rhs, "gradient_check::contract");
  double total = 0.0;
  for (std::size_t i = 0; i < mlp::rows(lhs); ++i) {
    for (std::size_t j = 0; j < mlp::cols(lhs); ++j) {
      total += lhs[i][j] * rhs[i][j];
    }
  }
  return total;
}

inline double objective(mlp::Layer &layer, const mlp::Matrix &input, const mlp::Matrix &grad_output,
                        double scale = 1.0) {
  return scale * contract(layer.forward(input), grad_output);
}

inline double relative_error(double analytical, double numerical) {
  return std::fabs(analytical - numerical) /
         std::max({1.0, std::fabs(analytical), std::fabs(numerical)});
}

inline double max_relative_error(const mlp::Matrix &analytical, const mlp::Matrix &numerical) {
  mlp::check_same_shape(analytical, numerical, "gradient_check::max_relative_error");
  double max_error = 0.0;
  for (std::size_t i = 0; i < mlp::rows(analytical); ++i) {
    for (std::size_t j = 0; j < mlp::cols(analytical); ++j) {
      max_error = std::max(max_error, relative_error(analytical[i][j], numerical[i][j]));
    }
  }
  return max_error;
}

inline double max_relative_error(const mlp::Vector &analytical, const mlp::Vector &numerical) {
  if (analytical.size() != numerical.size()) {
    throw std::invalid_argument("Shape mismatch in gradient_check::max_relative_error(vector)");
  }

  double max_error = 0.0;
  for (std::size_t i = 0; i < analytical.size(); ++i) {
    max_error = std::max(max_error, relative_error(analytical[i], numerical[i]));
  }
  return max_error;
}

inline mlp::Matrix numerical_input_gradient(mlp::Layer &layer, mlp::Matrix input,
                                            const mlp::Matrix &grad_output, double epsilon) {
  mlp::Matrix gradient = mlp::zeros_like(input);
  for (std::size_t i = 0; i < mlp::rows(input); ++i) {
    for (std::size_t j = 0; j < mlp::cols(input); ++j) {
      input[i][j] += epsilon;
      const double positive = objective(layer, input, grad_output);
      input[i][j] -= 2.0 * epsilon;
      const double negative = objective(layer, input, grad_output);
      input[i][j] += epsilon;
      gradient[i][j] = (positive - negative) / (2.0 * epsilon);
    }
  }
  return gradient;
}

inline mlp::Matrix numerical_matrix_gradient(mlp::Layer &layer, mlp::Matrix &parameter,
                                             const mlp::Matrix &input,
                                             const mlp::Matrix &grad_output, double scale,
                                             double epsilon) {
  mlp::Matrix gradient = mlp::zeros_like(parameter);
  for (std::size_t i = 0; i < mlp::rows(parameter); ++i) {
    for (std::size_t j = 0; j < mlp::cols(parameter); ++j) {
      parameter[i][j] += epsilon;
      const double positive = objective(layer, input, grad_output, scale);
      parameter[i][j] -= 2.0 * epsilon;
      const double negative = objective(layer, input, grad_output, scale);
      parameter[i][j] += epsilon;
      gradient[i][j] = (positive - negative) / (2.0 * epsilon);
    }
  }
  return gradient;
}

inline mlp::Vector numerical_vector_gradient(mlp::Layer &layer, mlp::Vector &parameter,
                                             const mlp::Matrix &input,
                                             const mlp::Matrix &grad_output, double scale,
                                             double epsilon) {
  mlp::Vector gradient(parameter.size(), 0.0);
  for (std::size_t i = 0; i < parameter.size(); ++i) {
    parameter[i] += epsilon;
    const double positive = objective(layer, input, grad_output, scale);
    parameter[i] -= 2.0 * epsilon;
    const double negative = objective(layer, input, grad_output, scale);
    parameter[i] += epsilon;
    gradient[i] = (positive - negative) / (2.0 * epsilon);
  }
  return gradient;
}

inline bool check_layer_gradients(mlp::Layer &layer, const mlp::Matrix &input,
                                  const mlp::Matrix &grad_output, const Options &options,
                                  std::ostream &stream = std::cerr) {
  layer.forward(input);
  const mlp::Matrix analytical_input_gradient = layer.backward(grad_output);
  const auto matrix_params = layer.matrix_params();
  const auto vector_params = layer.vector_params();

  const mlp::Matrix numerical_input =
      numerical_input_gradient(layer, input, grad_output, options.epsilon);
  const double input_relative_error =
      max_relative_error(analytical_input_gradient, numerical_input);
  if (input_relative_error > options.relative_tolerance) {
    stream << "input gradient relative error " << input_relative_error
           << " exceeded tolerance " << options.relative_tolerance << "\n";
    return false;
  }

  for (std::size_t index = 0; index < matrix_params.size(); ++index) {
    const mlp::Matrix numerical = numerical_matrix_gradient(
        layer, *matrix_params[index].value, input, grad_output,
        options.parameter_objective_scale, options.epsilon);
    const double parameter_relative_error =
        max_relative_error(*matrix_params[index].grad, numerical);
    if (parameter_relative_error > options.relative_tolerance) {
      stream << "matrix parameter gradient " << index << " relative error "
             << parameter_relative_error << " exceeded tolerance "
             << options.relative_tolerance << "\n";
      return false;
    }
  }

  for (std::size_t index = 0; index < vector_params.size(); ++index) {
    const mlp::Vector numerical = numerical_vector_gradient(
        layer, *vector_params[index].value, input, grad_output,
        options.parameter_objective_scale, options.epsilon);
    const double parameter_relative_error =
        max_relative_error(*vector_params[index].grad, numerical);
    if (parameter_relative_error > options.relative_tolerance) {
      stream << "vector parameter gradient " << index << " relative error "
             << parameter_relative_error << " exceeded tolerance "
             << options.relative_tolerance << "\n";
      return false;
    }
  }

  return true;
}

}  // namespace gradient_check

#endif
