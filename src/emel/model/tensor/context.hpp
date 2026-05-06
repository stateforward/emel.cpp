#pragma once

#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/model/tensor/detail.hpp"

namespace emel::io::mmap {
struct sm;
} // namespace emel::io::mmap

namespace emel::io::read {
struct sm;
} // namespace emel::io::read

namespace emel::model::tensor::action {

struct context {
  detail::tensor_storage tensors = {};
  emel::io::mmap::sm *io_mmap = nullptr;
  emel::io::read::sm *io_read = nullptr;
};

} // namespace emel::model::tensor::action
