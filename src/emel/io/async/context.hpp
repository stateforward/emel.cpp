#pragma once

namespace emel::io::async::action {

// Empty persistent actor context. Resumable ownership is caller-provided and
// must not be mirrored here.
struct context {};

} // namespace emel::io::async::action
