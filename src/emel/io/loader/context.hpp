#pragma once

namespace emel::io::async {
struct sm;
} // namespace emel::io::async

namespace emel::io::read {
struct sm;
} // namespace emel::io::read

namespace emel::io::staged_read {
struct sm;
} // namespace emel::io::staged_read

namespace emel::io::loader::action {

struct context {
  emel::io::read::sm *io_read = nullptr;
  emel::io::staged_read::sm *io_staged_read = nullptr;
  emel::io::async::sm *io_async = nullptr;
};

} // namespace emel::io::loader::action
