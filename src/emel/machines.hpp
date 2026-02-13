#pragma once

#include "emel/buffer_allocator/sm.hpp"
#include "emel/buffer_planner/sm.hpp"
#include "emel/decoder/sm.hpp"
#include "emel/encoder/sm.hpp"
#include "emel/generator/sm.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/tokenizer/sm.hpp"

namespace emel {

using BufferAllocator = emel::buffer_allocator::sm;
using BufferPlanner = emel::buffer_planner::sm;
using Decoder = emel::decoder::sm;
using Encoder = emel::encoder::sm;
using Generator = emel::generator::sm;
using ModelLoader = emel::model::loader::sm;
using Model = emel::model::loader::sm;
using Parser = emel::model::parser::sm;
using Tokenizer = emel::tokenizer::sm;
using WeightLoader = emel::model::weight_loader::sm;

}  // namespace emel
