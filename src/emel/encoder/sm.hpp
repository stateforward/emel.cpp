#pragma once

#include "emel/encoder/bpe/sm.hpp"
#include "emel/encoder/fallback/sm.hpp"
#include "emel/encoder/plamo2/sm.hpp"
#include "emel/encoder/rwkv/sm.hpp"
#include "emel/encoder/spm/sm.hpp"
#include "emel/encoder/ugm/sm.hpp"
#include "emel/encoder/wpm/sm.hpp"

namespace emel::encoder {

using sm = emel::encoder::bpe::sm;

}  // namespace emel::encoder
