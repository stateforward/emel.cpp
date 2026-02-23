#pragma once

#include "emel/buffer/allocator/sm.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/batch/planner/sm.hpp"
#include "emel/graph/processor/sm.hpp"
#include "emel/decoder/sm.hpp"
#include "emel/text/encoders/any.hpp"
#include "emel/text/encoders/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/detokenizer/sm.hpp"
#include "emel/generator/sm.hpp"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"
#include "emel/memory/coordinator/kv/sm.hpp"
#include "emel/memory/coordinator/hybrid/sm.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/telemetry/exporter/sm.hpp"
#include "emel/telemetry/provider/sm.hpp"
#include "emel/tensor/allocator/sm.hpp"
#include "emel/tensor/lifetime_analyzer/sm.hpp"
#include "emel/text/renderer/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace emel {

using BufferAllocator = emel::buffer::allocator::sm;
using BufferChunkAllocator = emel::buffer::chunk_allocator::sm;
using BufferPlanner = emel::buffer::planner::sm;
using BufferReallocAnalyzer = emel::buffer::realloc_analyzer::sm;
using BatchSplitter = emel::batch::planner::sm;
using ComputeExecutor = emel::graph::processor::sm;
using Decoder = emel::decoder::sm;
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
using KvCache = emel::memory::kv::sm;
using RecurrentMemory = emel::memory::recurrent::sm;
using HybridMemory = emel::memory::hybrid::sm;
using MemoryCoordinator = emel::memory::coordinator::sm;
using MemoryCoordinatorRecurrent = emel::memory::coordinator::recurrent::sm;
using MemoryCoordinatorKv = emel::memory::coordinator::kv::sm;
using MemoryCoordinatorHybrid = emel::memory::coordinator::hybrid::sm;
using ModelLoader = emel::model::loader::sm;
using Model = emel::model::loader::sm;
using Parser = emel::parser::gguf::sm;
using TelemetryExporter = emel::telemetry::exporter::sm;
using TelemetryProvider = emel::telemetry::provider::sm;
using TensorAllocator = emel::tensor::allocator::sm;
using TensorLifetimeAnalyzer = emel::tensor::lifetime_analyzer::sm;
using Renderer = emel::text::renderer::sm;
using Tokenizer = emel::text::tokenizer::sm;
using WeightLoader = emel::model::weight_loader::sm;

}  // namespace emel
