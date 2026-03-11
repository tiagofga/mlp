#ifndef MLP_IO_HPP
#define MLP_IO_HPP

#include <string>

#include "model.hpp"

namespace mlp {

void save_sequential(const Sequential &model, const std::string &path);
Sequential load_sequential(const std::string &path);

}  // namespace mlp

#endif
