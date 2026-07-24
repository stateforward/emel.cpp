#pragma once

#include "emel/speech/generator/frame/sm.hpp"

namespace emel {

template <class dependencies_type>
using SpeechGeneratorFrame = speech::generator::frame::sm<dependencies_type>;

} // namespace emel
