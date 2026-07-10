#pragma once

#include "emel/speech/generator/sm.hpp"

namespace emel::speech::generator {

template <class dependencies_type>
using SpeechGenerator = sm<dependencies_type>;

} // namespace emel::speech::generator
