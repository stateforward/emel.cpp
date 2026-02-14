#pragma once

#include "emel/buffer/allocator/sm.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/decoder/sm.hpp"
#include "emel/encoder/sm.hpp"
#include "emel/generator/sm.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/tensor/allocator/sm.hpp"
#include "emel/tensor/lifetime_analyzer/sm.hpp"
#include "emel/tokenizer/sm.hpp"

namespace emel {

using BufferAllocator = emel::buffer::allocator::sm;
using BufferChunkAllocator = emel::buffer::chunk_allocator::sm;
using BufferPlanner = emel::buffer::planner::sm;
using BufferReallocAnalyzer = emel::buffer::realloc_analyzer::sm;
using Decoder = emel::decoder::sm;
using Encoder = emel::encoder::sm;
using Generator = emel::generator::sm;
using ModelLoader = emel::model::loader::sm;
using Model = emel::model::loader::sm;
using Parser = emel::model::parser::sm;
using TensorAllocator = emel::tensor::allocator::sm;
using TensorLifetimeAnalyzer = emel::tensor::lifetime_analyzer::sm;
using Tokenizer = emel::tokenizer::sm;
using WeightLoader = emel::model::weight_loader::sm;

}  // namespace emel
