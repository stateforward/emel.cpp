#pragma once

// benchmark: scaffold

#include "emel/text/encoders/bpe/sm.hpp"
#include "emel/text/encoders/fallback/sm.hpp"
#include "emel/text/encoders/plamo2/sm.hpp"
#include "emel/text/encoders/rwkv/sm.hpp"
#include "emel/text/encoders/spm/sm.hpp"
#include "emel/text/encoders/ugm/sm.hpp"
#include "emel/text/encoders/wpm/sm.hpp"

namespace emel::text::encoders {

using sm = emel::text::encoders::bpe::sm;

}  // namespace emel::text::encoders
