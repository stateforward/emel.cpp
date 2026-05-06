#pragma once

namespace emel::io::read {
struct sm;
} // namespace emel::io::read

namespace emel::io::loader::action {

struct context {
  emel::io::read::sm *io_read = nullptr;
};

} // namespace emel::io::loader::action
