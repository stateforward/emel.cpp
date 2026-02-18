#pragma once

#include "emel/buffer/allocator/sm.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/decoder/compute_executor/sm.hpp"
#include "emel/decoder/sm.hpp"
#include "emel/encoder/sm.hpp"
#include "emel/generator/sm.hpp"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/telemetry/exporter/sm.hpp"
#include "emel/telemetry/provider/sm.hpp"
#include "emel/tensor/allocator/sm.hpp"
#include "emel/tensor/lifetime_analyzer/sm.hpp"
#include "emel/tokenizer/sm.hpp"

namespace emel {

using BufferAllocator = emel::buffer::allocator::sm;
using BufferChunkAllocator = emel::buffer::chunk_allocator::sm;
using BufferPlanner = emel::buffer::planner::sm;
using BufferReallocAnalyzer = emel::buffer::realloc_analyzer::sm;
using BatchSplitter = emel::batch::splitter::sm;
using ComputeExecutor = emel::decoder::compute_executor::sm;
using Decoder = emel::decoder::sm;
using EncoderBpe = emel::encoder::bpe::sm;
using EncoderSpm = emel::encoder::spm::sm;
using EncoderWpm = emel::encoder::wpm::sm;
using EncoderUgm = emel::encoder::ugm::sm;
using EncoderRwkv = emel::encoder::rwkv::sm;
using EncoderPlamo2 = emel::encoder::plamo2::sm;
using EncoderFallback = emel::encoder::fallback::sm;
using Generator = emel::generator::sm;
using KvCache = emel::kv::cache::sm;
using MemoryCoordinator = emel::memory::coordinator::sm;
using ModelLoader = emel::model::loader::sm;
using Model = emel::model::loader::sm;
using Parser = emel::model::parser::sm;
using TelemetryExporter = emel::telemetry::exporter::sm;
using TelemetryProvider = emel::telemetry::provider::sm;
using TensorAllocator = emel::tensor::allocator::sm;
using TensorLifetimeAnalyzer = emel::tensor::lifetime_analyzer::sm;
using Tokenizer = emel::tokenizer::sm;
using WeightLoader = emel::model::weight_loader::sm;

}  // namespace emel
