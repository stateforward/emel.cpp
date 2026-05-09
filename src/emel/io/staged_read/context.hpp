#pragma once

#include "emel/io/staged_read/errors.hpp"

namespace emel::io::staged_read::action {

// Empty persistent actor context. Per AGENTS.md, dispatch-local payloads are
// not mirrored here.

struct context {};

} // namespace emel::io::staged_read::action
