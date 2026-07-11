#pragma once

#include "emel/speech/predictor/moshi/sm.hpp"

namespace emel::speech::predictor::moshi {

using policies = action::policies;

template <class graph_actor_type>
using dependencies = action::dependencies<graph_actor_type>;

} // namespace emel::speech::predictor::moshi
