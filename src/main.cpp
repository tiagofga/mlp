#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mlp/library.hpp"

namespace {

void print_usage(const char *program) {
  std::cout << "Usage: " << program
            << " [--optimizer sgd|momentum|nag|adam|adamw|nadam|rmsprop|adagrad|adadelta|lion]"
            << " [--hidden 16,16,8]"
            << " [--epochs N]"
            << " [--lr VALUE]"
            << " [--samples N]"
            << " [--seed N]"
            << " [--train-ratio VALUE]"
            << " [--val-ratio VALUE]"
            << " [--threshold VALUE]\n";
}

bool parse_hidden_sizes(const std::string &raw, std::vector<std::size_t> &hidden_sizes, std::string &error) {
  hidden_sizes.clear();
  std::stringstream ss(raw);
  std::string token;

  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      error = "Empty hidden size value in --hidden list";
      return false;
    }

    std::size_t idx = 0;
    std::size_t value = 0;
    try {
      value = std::stoul(token, &idx);
    } catch (const std::exception &) {
      error = "Invalid hidden size value: " + token;
      return false;
    }

    if (idx != token.size() || value == 0) {
      error = "Invalid hidden size value: " + token;
      return false;
    }

    hidden_sizes.push_back(value);
  }

  if (hidden_sizes.empty()) {
    error = "Provide at least one hidden size, for example: --hidden 16,16,8";
    return false;
  }
  return true;
}

bool parse_positive_int(const std::string &raw, int &value, std::string &error) {
  std::size_t idx = 0;
  long parsed = 0;
  try {
    parsed = std::stol(raw, &idx);
  } catch (const std::exception &) {
    error = "Invalid integer value: " + raw;
    return false;
  }
  if (idx != raw.size() || parsed <= 0) {
    error = "Value must be a positive integer: " + raw;
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

bool parse_positive_size(const std::string &raw, std::size_t &value, std::string &error) {
  std::size_t idx = 0;
  std::size_t parsed = 0;
  try {
    parsed = std::stoul(raw, &idx);
  } catch (const std::exception &) {
    error = "Invalid size value: " + raw;
    return false;
  }
  if (idx != raw.size() || parsed == 0) {
    error = "Value must be a positive integer: " + raw;
    return false;
  }
  value = parsed;
  return true;
}

bool parse_positive_double(const std::string &raw, double &value, std::string &error) {
  std::size_t idx = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(raw, &idx);
  } catch (const std::exception &) {
    error = "Invalid numeric value: " + raw;
    return false;
  }
  if (idx != raw.size() || parsed <= 0.0) {
    error = "Value must be > 0: " + raw;
    return false;
  }
  value = parsed;
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  mlp::ExperimentOptions opt;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }

    if (arg == "--optimizer") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --optimizer\n";
        print_usage(argv[0]);
        return 1;
      }
      opt.optimizer = argv[++i];
      continue;
    }

    if (arg == "--hidden") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --hidden\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_hidden_sizes(argv[++i], opt.hidden, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    if (arg == "--epochs") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --epochs\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_positive_int(argv[++i], opt.epochs, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    if (arg == "--lr") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --lr\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_positive_double(argv[++i], opt.learning_rate, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    if (arg == "--samples") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --samples\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_positive_size(argv[++i], opt.samples, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    if (arg == "--seed") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --seed\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      std::size_t parsed = 0;
      if (!parse_positive_size(argv[++i], parsed, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      opt.seed = static_cast<unsigned int>(parsed);
      continue;
    }

    if (arg == "--train-ratio") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --train-ratio\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_positive_double(argv[++i], opt.train_ratio, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    if (arg == "--val-ratio") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --val-ratio\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_positive_double(argv[++i], opt.val_ratio, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    if (arg == "--threshold") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --threshold\n";
        print_usage(argv[0]);
        return 1;
      }
      std::string err;
      if (!parse_positive_double(argv[++i], opt.threshold, err)) {
        std::cerr << err << "\n";
        print_usage(argv[0]);
        return 1;
      }
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage(argv[0]);
    return 1;
  }

  try {
    (void)mlp::run_xor_experiment(opt, &std::cout);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  return 0;
}
