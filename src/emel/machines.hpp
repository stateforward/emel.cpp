#pragma once

#include "emel/batch/planner/sm.hpp"
#include "emel/graph/processor/sm.hpp"
#include "emel/text/encoders/any.hpp"
#include "emel/text/encoders/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/detokenizer/sm.hpp"
#include "emel/generator/sm.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/text/renderer/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace emel {

using ComputeExecutor = emel::graph::processor::sm;
using Conditioner = emel::text::conditioner::sm;
using Detokenizer = emel::text::detokenizer::sm;
using EncoderAny = emel::text::encoders::any;
using EncoderBpe = emel::text::encoders::bpe::sm;
using EncoderSpm = emel::text::encoders::spm::sm;
using EncoderWpm = emel::text::encoders::wpm::sm;
using EncoderUgm = emel::text::encoders::ugm::sm;
using EncoderRwkv = emel::text::encoders::rwkv::sm;
using EncoderPlamo2 = emel::text::encoders::plamo2::sm;
using EncoderFallback = emel::text::encoders::fallback::sm;
using Generator = emel::generator::sm;
using MemoryHybrid = emel::memory::hybrid::sm;
using KvCache = emel::memory::kv::sm;
using MemoryRecurrent = emel::memory::recurrent::sm;
using ModelLoader = emel::model::loader::sm;
using Model = emel::model::loader::sm;
using Parser = emel::gguf::loader::sm;
using Renderer = emel::text::renderer::sm;
using Tokenizer = emel::text::tokenizer::sm;
using WeightLoader = emel::model::weight_loader::sm;

}  // namespace emel
