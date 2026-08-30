#pragma once

namespace emel::model::needle::action {

// The needle binder holds no persistent actor state: each bind request
// carries its geometry, tensor table, and output contract, and every
// per-dispatch value crosses phases via the typed runtime event.
struct context {};

} // namespace emel::model::needle::action
