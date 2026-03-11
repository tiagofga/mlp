#include "mlp/io.hpp"

#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "activations.hpp"
#include "dense.hpp"

namespace mlp {
namespace {

constexpr const char *kMagic = "MLPSEQv1";

void check_stream(const std::ios &stream, const std::string &context) {
  if (!stream) {
    throw std::runtime_error("I/O error while " + context);
  }
}

void write_dense(std::ostream &os, const Dense &dense) {
  const Matrix &w = dense.weights();
  const Vector &b = dense.bias();
  os << "Dense " << rows(w) << " " << cols(w) << "\n";
  for (std::size_t i = 0; i < rows(w); ++i) {
    for (std::size_t j = 0; j < cols(w); ++j) {
      os << w[i][j] << (j + 1 < cols(w) ? ' ' : '\n');
    }
  }
  for (std::size_t j = 0; j < b.size(); ++j) {
    os << b[j] << (j + 1 < b.size() ? ' ' : '\n');
  }
}

std::unique_ptr<Layer> read_layer(std::istream &is) {
  std::string layer_type;
  is >> layer_type;
  check_stream(is, "reading layer type");

  if (layer_type == "Dense") {
    std::size_t in_features = 0;
    std::size_t out_features = 0;
    is >> in_features >> out_features;
    check_stream(is, "reading dense shape");

    Matrix w = make_matrix(in_features, out_features, 0.0);
    for (std::size_t i = 0; i < in_features; ++i) {
      for (std::size_t j = 0; j < out_features; ++j) {
        is >> w[i][j];
        check_stream(is, "reading dense weights");
      }
    }

    Vector b(out_features, 0.0);
    for (std::size_t j = 0; j < out_features; ++j) {
      is >> b[j];
      check_stream(is, "reading dense bias");
    }

    std::mt19937 rng(0);
    auto dense = std::make_unique<Dense>(in_features, out_features, rng);
    dense->set_parameters(w, b);
    return dense;
  }

  if (layer_type == "ReLU") return std::make_unique<ReLU>();
  if (layer_type == "Sigmoid") return std::make_unique<Sigmoid>();
  if (layer_type == "Tanh") return std::make_unique<Tanh>();

  throw std::invalid_argument("Unknown layer type in model file: " + layer_type);
}

}  // namespace

void save_sequential(const Sequential &model, const std::string &path) {
  std::ofstream os(path);
  if (!os.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + path);
  }

  const auto &layers = model.layers();
  os << kMagic << "\n";
  os << layers.size() << "\n";

  for (const auto &layer : layers) {
    const std::string type = layer->type();
    if (type == "Dense") {
      const Dense *dense = dynamic_cast<const Dense *>(layer.get());
      if (dense == nullptr) {
        throw std::runtime_error("Internal type mismatch while saving Dense layer");
      }
      write_dense(os, *dense);
    } else if (type == "ReLU" || type == "Sigmoid" || type == "Tanh") {
      os << type << "\n";
    } else {
      throw std::invalid_argument("Cannot serialize unsupported layer type: " + type);
    }

    check_stream(os, "writing model file");
  }
}

Sequential load_sequential(const std::string &path) {
  std::ifstream is(path);
  if (!is.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + path);
  }

  std::string magic;
  is >> magic;
  check_stream(is, "reading file header");
  if (magic != kMagic) {
    throw std::invalid_argument("Invalid model file format: " + path);
  }

  std::size_t layer_count = 0;
  is >> layer_count;
  check_stream(is, "reading layer count");

  Sequential model;
  for (std::size_t i = 0; i < layer_count; ++i) {
    model.add(read_layer(is));
  }

  return model;
}

}  // namespace mlp
