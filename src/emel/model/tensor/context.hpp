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

namespace emel::io::staged_read {
struct sm;
} // namespace emel::io::staged_read

namespace emel::io::async {
struct sm;
} // namespace emel::io::async

namespace emel::model::tensor::action {

struct context {
  detail::tensor_storage tensors = {};
  emel::io::mmap::sm *io_mmap = nullptr;
  emel::io::read::sm *io_read = nullptr;
  emel::io::staged_read::sm *io_staged_read = nullptr;
  emel::io::async::sm *io_async = nullptr;
};

} // namespace emel::model::tensor::action
