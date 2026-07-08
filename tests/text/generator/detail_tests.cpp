#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string_view>

#include <doctest/doctest.h>

// Component-private numeric and binding regression tests.
// Maintained generator behavior proof lives in lifecycle, parity, and benchmark
// tests that drive public generator events. This file intentionally covers
// private detail kernels/helpers until the corresponding code is extracted to a
// kernel-owned surface.

#include "../../kernel/test_helpers.hpp"
#include "emel/graph/processor/guards.hpp"
#include "emel/memory/events.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/model/transformer/any.hpp"
#include "emel/text/generator/detail.hpp"
#include "emel/text/generator/guards.hpp"
#include "emel/text/generator/layer/sm.hpp"
#include "generator_test_policies.hpp"

namespace {

using emel::kernel::test::flash_attn_reference_online_softmax_f16_values;
using emel::kernel::test::k_flash_online_f16_abs_tolerance;
using emel::kernel::test::within_flash_online_f16_tolerance;
using emel::text::generator::detail::quant::block_q2_k;
using emel::text::generator::detail::quant::block_q3_k;
using emel::text::generator::detail::quant::block_q4_0;
using emel::text::generator::detail::quant::block_q4_k;
using emel::text::generator::detail::quant::block_q6_k;
using emel::text::generator::detail::quant::block_q8_0;
using emel::text::generator::detail::quant::Q4_K_X8_ROWS;
using emel::text::generator::detail::quant::Q6_K_X8_ROWS;
using emel::text::generator::detail::quant::QK4_0;
using emel::text::generator::detail::quant::QK8_0;
using emel::text::generator::detail::quant::QK_K;

struct matmul_actor_fixture {
  emel::text::generator::matmul::lane_pool<7u, 128u, 1048576u>
      parallel_matmul_lanes = {};
  emel::text::generator::matmul::execution_policy policy =
      emel::text::generator::matmul::make_auto_execution_policy(
          parallel_matmul_lanes);
  emel::text::generator::matmul::sm actor{policy};
};

void bind_test_matmul_actor(
    emel::text::generator::detail::native_backend &backend,
    matmul_actor_fixture &matmul) {
  backend.matmul_actor = &matmul.actor;
  matmul.actor.process_event(
      emel::text::generator::matmul::event::configure_kernel_kind{
          backend.kernel_kind});
}

uint16_t fp16_bits(const float value) {
  return emel::text::generator::detail::quant::fp32_to_fp16(value);
}

TEST_CASE("generator detects only maintained host kernel kinds") {
  const emel::kernel::kernel_kind kind = emel::kernel::detect_host_kind();
#if defined(__aarch64__) || defined(_M_ARM64)
  CHECK(kind == emel::kernel::kernel_kind::aarch64);
#else
  CHECK(kind == emel::kernel::kernel_kind::x86_64);
#endif
}

std::array<block_q6_k, Q6_K_X8_ROWS> make_q6_rows() {
  std::array<block_q6_k, Q6_K_X8_ROWS> rows = {};
  for (size_t row = 0; row < rows.size(); ++row) {
    rows[row].d = 0x3c00u;
    for (size_t idx = 0; idx < rows[row].scales.size(); ++idx) {
      rows[row].scales[idx] =
          static_cast<int8_t>((static_cast<int32_t>((row + idx) % 13u)) - 6);
    }
    for (size_t idx = 0; idx < rows[row].ql.size(); ++idx) {
      rows[row].ql[idx] =
          static_cast<uint8_t>(((row + 1u) * 17u + idx * 7u) & 0xffu);
    }
    for (size_t idx = 0; idx < rows[row].qh.size(); ++idx) {
      rows[row].qh[idx] =
          static_cast<uint8_t>(((row + 3u) * 11u + idx * 5u) & 0xffu);
    }
  }
  return rows;
}

std::array<block_q4_k, Q4_K_X8_ROWS> make_q4_rows() {
  std::array<block_q4_k, Q4_K_X8_ROWS> rows = {};
  for (size_t row = 0; row < rows.size(); ++row) {
    rows[row].d = 0x3800u;
    rows[row].dmin = 0x3400u;
    for (size_t idx = 0; idx < rows[row].scales.size(); ++idx) {
      rows[row].scales[idx] =
          static_cast<uint8_t>(((row + 5u) * 23u + idx * 11u) & 0xffu);
    }
    for (size_t idx = 0; idx < rows[row].qs.size(); ++idx) {
      rows[row].qs[idx] =
          static_cast<uint8_t>(((row + 1u) * 29u + idx * 3u) & 0xffu);
    }
  }
  return rows;
}

emel::model::data::tensor_record make_tensor_record(void *data,
                                                    const int32_t type,
                                                    const int32_t cols,
                                                    const int32_t rows) {
  emel::model::data::tensor_record tensor = {};
  tensor.data = data;
  tensor.type = type;
  tensor.n_dims = 2;
  tensor.dims[0] = static_cast<uint64_t>(cols);
  tensor.dims[1] = static_cast<uint64_t>(rows);
  tensor.data_size =
      emel::text::generator::detail::row_storage_bytes(tensor, cols) *
      static_cast<uint64_t>(rows);
  return tensor;
}

struct runtime_request_fixture {
  emel::model::data model = {};
  emel::model::transformer::execution_view execution = {};
  emel::model::transformer::topology topology = {};
  emel::model::transformer::step_plan plan = {};
  emel::text::generator::detail::native_backend backend = {};
  std::array<int32_t, 1> token_ids = {0};
  std::array<int32_t, 1> positions = {0};
  std::array<float, 6> logits = {};
  int32_t selected_token = -1;
  float selected_score = 0.0f;
  emel::text::generator::compute_io io = {};
  emel::memory::view::snapshot memory_snapshot = {};
  std::array<int32_t, 1> seq_primary_ids = {0};
  emel::graph::processor::event::execution_output output = {};
  emel::graph::processor::event::lifecycle_tensor_binding lifecycle_tensor = {};
  emel::graph::processor::event::lifecycle_phase lifecycle_phase = {};
  emel::graph::processor::event::lifecycle_manifest lifecycle = {};
  emel::graph::processor::event::execute request = {};

  static bool on_execution_done(
      void *, const emel::graph::processor::events::execution_done &) noexcept {
    return true;
  }

  static bool on_execution_error(
      void *,
      const emel::graph::processor::events::execution_error &) noexcept {
    return true;
  }

  explicit runtime_request_fixture(
      const emel::model::transformer::step_kind kind =
          emel::model::transformer::step_kind::prefill) {
    topology.execution = &execution;
    plan.graph = &topology;
    plan.kind = kind;
    plan.expected_outputs = 1;

    backend.model = &model;
    backend.n_embd = 4;
    backend.n_head = 1;
    backend.n_head_kv = 1;
    backend.n_layer = 1;
    backend.n_vocab = 4;
    backend.n_ctx = 8;
    backend.kv_block_tokens = 8;
    backend.kv_positions_capacity = 8;
    backend.head_dim = 4;
    backend.head_dim_kv = 4;
    backend.blocks.resize(1u);
    backend.bound_tokens.resize(1u);
    backend.bound_positions.resize(1u);
    backend.bound_logits = {0.25f, 0.5f, 0.75f, 1.0f};

    io.backend_ctx = &backend;
    io.token_ids = token_ids.data();
    io.token_count = static_cast<int32_t>(token_ids.size());
    io.logits = logits.data();
    io.logits_capacity = static_cast<int32_t>(logits.size());
    io.selected_token_out = &selected_token;
    io.selected_score_out = &selected_score;

    // Coherent captured snapshot for the request contract: one 8-token block
    // covering the single-token workloads these fixtures drive.
    memory_snapshot.max_sequences = 1;
    memory_snapshot.block_tokens = 8;
    memory_snapshot.sequence_active[0] = 1;
    memory_snapshot.sequence_length_values[0] = 1;
    memory_snapshot.sequence_kv_block_count[0] = 1;
    memory_snapshot.sequence_kv_blocks[0][0] = 0;
    lifecycle_tensor.tensor_id = 0;
    lifecycle_tensor.buffer = logits.data();
    lifecycle_tensor.buffer_bytes = sizeof(float) * logits.size();
    lifecycle_tensor.consumer_refs = 1;
    lifecycle_tensor.is_leaf = true;
    lifecycle.tensors = &lifecycle_tensor;
    lifecycle.tensor_count = 1;
    lifecycle.phase = &lifecycle_phase;

    request.step_plan = &plan;
    request.output_out = &output;
    request.lifecycle = &lifecycle;
    request.tensor_machine = reinterpret_cast<emel::graph::tensor::sm *>(this);
    request.step_index = 0;
    request.step_size = 1;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = static_cast<int32_t>(positions.size());
    request.kv_tokens = 0;
    request.memory_view = &memory_snapshot;
    request.seq_primary_ids = seq_primary_ids.data();
    request.seq_primary_ids_count =
        static_cast<int32_t>(seq_primary_ids.size());
    request.dispatch_done = {this, on_execution_done};
    request.dispatch_error = {this, on_execution_error};
  }
};

void bind_neox_rope_pairing(emel::model::data &model) {
  model.params.rope_pair_x0_stride = 1;
  model.params.rope_pair_x1_stride = 1;
  model.params.rope_pair_x1_offset = 0;
  model.params.rope_pair_x1_half_rot_offset = 1;
}

struct qwen3_runtime_fixture {
  emel::model::data model = {};
  std::vector<std::vector<float>> tensor_storage = {};

  qwen3_runtime_fixture() {
    tensor_storage.reserve(14);
    model.params.n_vocab = 2;
    model.params.n_embd = 4;
    model.params.n_head = 2;
    model.params.n_head_kv = 2;
    model.params.n_ctx = 8;
    model.params.n_rot = 2;
    model.params.n_layer = 1;
    model.params.attention_layer_norm_rms_epsilon = 1.0e-5f;
    model.params.rope_freq_base = 10000.0f;
    bind_neox_rope_pairing(model);
    model.n_layers = 1;
    model.weights_data = model.tensors.data();
    model.weights_size = 1u;
    std::memcpy(model.architecture_name.data(), "qwen3", 5u);

    uint32_t tensor_index = 0u;
    const auto add_name = [&](emel::model::data::tensor_record &tensor,
                              const std::string_view name) {
      tensor.name_offset = model.name_bytes_used;
      tensor.name_length = static_cast<uint32_t>(name.size());
      std::memcpy(model.name_storage.data() + model.name_bytes_used,
                  name.data(), name.size());
      model.name_bytes_used += static_cast<uint32_t>(name.size());
    };
    const auto add_vector = [&](const std::string_view name,
                                const std::vector<float> &values) {
      auto &tensor = model.tensors[tensor_index++];
      add_name(tensor, name);
      tensor_storage.push_back(values);
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
      tensor.n_dims = 1;
      tensor.dims[0] = static_cast<uint64_t>(values.size());
      tensor.data = tensor_storage.back().data();
      tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
    };
    const auto add_matrix = [&](const std::string_view name, const int32_t rows,
                                const int32_t cols,
                                const std::vector<float> &values) {
      auto &tensor = model.tensors[tensor_index++];
      add_name(tensor, name);
      tensor_storage.push_back(values);
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
      tensor.n_dims = 2;
      tensor.dims[0] = static_cast<uint64_t>(cols);
      tensor.dims[1] = static_cast<uint64_t>(rows);
      tensor.data = tensor_storage.back().data();
      tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
    };

    const std::vector<float> identity = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    add_matrix("token_embd.weight", 2, 4,
               {
                   1.0f,
                   2.0f,
                   3.0f,
                   4.0f,
                   0.0f,
                   0.0f,
                   0.0f,
                   0.0f,
               });
    add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix("output.weight", 2, 4, std::vector<float>(8, 0.0f));
    add_vector("blk.0.attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix("blk.0.attn_q.weight", 4, 4, identity);
    add_matrix("blk.0.attn_k.weight", 4, 4, identity);
    add_matrix("blk.0.attn_v.weight", 4, 4, identity);
    add_vector("blk.0.attn_q_norm.weight", {2.0f, 0.5f});
    add_vector("blk.0.attn_k_norm.weight", {1.5f, 0.25f});
    add_matrix("blk.0.attn_output.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_vector("blk.0.ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix("blk.0.ffn_gate.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_matrix("blk.0.ffn_down.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_matrix("blk.0.ffn_up.weight", 4, 4, std::vector<float>(16, 0.0f));
    model.n_tensors = tensor_index;
  }
};

struct prepared_qwen3_backend_fixture {
  qwen3_runtime_fixture model_fixture = {};
  matmul_actor_fixture matmul = {};
  emel::text::generator::detail::native_backend backend = {};
  emel::graph::processor::event::execute request = {};
  bool ready = false;

  prepared_qwen3_backend_fixture() {
    const auto runtime_policy =
        emel::text::generator::test::make_auto_runtime_policy(
            model_fixture.model);
    ready = emel::text::generator::detail::prepare(
                backend, model_fixture.model, matmul.actor, runtime_policy) ==
            emel::error::cast(emel::model::loader::error::none);
    backend.bound_tokens = {0};
    backend.bound_positions = {0};
    backend.bound_token_count = 1;
    backend.bound_position_count = 1;
    request.kv_tokens = 0;
  }
};

struct gemma4_runtime_fixture {
  emel::model::data model = {};
  std::vector<std::vector<float>> tensor_storage = {};

  gemma4_runtime_fixture() {
    tensor_storage.reserve(13);
    model.params.n_vocab = 2;
    model.params.n_embd = 4;
    model.params.n_embd_out = 4;
    model.params.n_ff = 4;
    model.params.n_head = 2;
    model.params.n_head_kv = 1;
    model.params.n_ctx = 8;
    model.params.n_rot = 2;
    model.params.n_layer = 1;
    model.params.attention_key_length = 2;
    model.params.attention_value_length = 2;
    model.params.attention_shared_kv_layers = 1;
    model.params.attention_layer_norm_rms_epsilon = 1.0e-6f;
    model.params.rope_freq_base = 10000.0f;
    model.params.tie_word_embeddings = true;
    bind_neox_rope_pairing(model);
    model.n_layers = 1;
    model.weights_data = model.tensors.data();
    model.weights_size = 1u;
    std::memcpy(model.architecture_name.data(), "gemma4", 6u);

    uint32_t tensor_index = 0u;
    const auto add_name = [&](emel::model::data::tensor_record &tensor,
                              const std::string_view name) {
      tensor.name_offset = model.name_bytes_used;
      tensor.name_length = static_cast<uint32_t>(name.size());
      std::memcpy(model.name_storage.data() + model.name_bytes_used,
                  name.data(), name.size());
      model.name_bytes_used += static_cast<uint32_t>(name.size());
    };
    const auto add_vector = [&](const std::string_view name,
                                const std::vector<float> &values) {
      auto &tensor = model.tensors[tensor_index++];
      add_name(tensor, name);
      tensor_storage.push_back(values);
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
      tensor.n_dims = 1;
      tensor.dims[0] = static_cast<uint64_t>(values.size());
      tensor.data = tensor_storage.back().data();
      tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
    };
    const auto add_matrix = [&](const std::string_view name, const int32_t rows,
                                const int32_t cols,
                                const std::vector<float> &values) {
      auto &tensor = model.tensors[tensor_index++];
      add_name(tensor, name);
      tensor_storage.push_back(values);
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
      tensor.n_dims = 2;
      tensor.dims[0] = static_cast<uint64_t>(cols);
      tensor.dims[1] = static_cast<uint64_t>(rows);
      tensor.data = tensor_storage.back().data();
      tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
    };

    const std::vector<float> q_identity = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::vector<float> kv_identity = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    };

    add_matrix("token_embd.weight", 2, 4,
               {
                   1.0f,
                   2.0f,
                   3.0f,
                   4.0f,
                   0.0f,
                   0.0f,
                   0.0f,
                   0.0f,
               });
    add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_vector("blk.0.attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix("blk.0.attn_q.weight", 4, 4, q_identity);
    add_matrix("blk.0.attn_k.weight", 2, 4, kv_identity);
    add_matrix("blk.0.attn_v.weight", 2, 4, kv_identity);
    add_vector("blk.0.attn_q_norm.weight", {2.0f, 0.5f});
    add_vector("blk.0.attn_k_norm.weight", {1.5f, 0.25f});
    add_matrix("blk.0.attn_output.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_vector("blk.0.ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix("blk.0.ffn_gate.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_matrix("blk.0.ffn_down.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_matrix("blk.0.ffn_up.weight", 4, 4, std::vector<float>(16, 0.0f));
    model.n_tensors = tensor_index;
  }
};

struct chunk4_prefill_runtime_fixture {
  static constexpr int32_t k_vocab = static_cast<int32_t>(QK8_0);
  static constexpr int32_t k_embd = static_cast<int32_t>(QK8_0);
  static constexpr int32_t k_ctx = 8;
  static constexpr int32_t k_prompt_tokens = 4;

  emel::model::data model = {};
  matmul_actor_fixture matmul = {};
  emel::text::generator::detail::native_backend backend = {};
  std::vector<float> token_embedding_storage = {};
  std::vector<float> output_argmax_storage = {};
  std::vector<float> output_norm_storage = {};
  std::vector<float> attention_norm_storage = {};
  std::vector<float> ffn_norm_storage = {};
  std::vector<block_q8_0> zero_rows = {};
  std::vector<uint8_t> packed_storage = {};
  emel::model::data::tensor_record token_embedding_tensor = {};
  emel::model::data::tensor_record packed_tensor = {};
  emel::model::data::tensor_record output_argmax_tensor = {};

  emel::model::transformer::execution_view execution = {};
  emel::model::transformer::topology topology = {};
  emel::model::transformer::step_plan plan = {};
  std::array<int32_t, k_prompt_tokens> token_ids = {0, 1, 2, 3};
  std::array<int32_t, k_prompt_tokens> positions = {0, 1, 2, 3};
  std::vector<float> logits = {};
  int32_t selected_token = -1;
  float selected_score = -1.0f;
  emel::text::generator::compute_io io = {};
  emel::memory::view::snapshot memory_snapshot = {};
  std::array<int32_t, 1> seq_primary_ids = {0};
  emel::graph::processor::event::execute request = {};
  bool ready = false;

  chunk4_prefill_runtime_fixture() {
    token_embedding_storage.resize(
        static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
    for (int32_t token = 0; token < k_prompt_tokens; ++token) {
      token_embedding_storage[static_cast<size_t>(token) *
                                  static_cast<size_t>(k_embd) +
                              static_cast<size_t>(token)] = 1.0f;
    }
    output_argmax_storage.resize(static_cast<size_t>(k_vocab * k_embd), 0.0f);
    output_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    ffn_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);

    const size_t block_count =
        static_cast<size_t>(k_embd / static_cast<int32_t>(QK8_0));
    zero_rows.resize(static_cast<size_t>(k_embd) * block_count);
    packed_storage.resize(
        sizeof(emel::kernel::detail::quant::block_q8_0x4) *
        emel::kernel::detail::quant::packed_q8_0_x4_group_count(
            static_cast<uint64_t>(k_embd)) *
        block_count);

    token_embedding_tensor = make_tensor_record(token_embedding_storage.data(),
                                                emel::kernel::detail::dtype_f32,
                                                k_embd, k_prompt_tokens);
    packed_tensor = make_tensor_record(packed_storage.data(),
                                       emel::kernel::detail::dtype_q8_0_x4_bl8,
                                       k_embd, k_embd);
    output_argmax_tensor =
        make_tensor_record(output_argmax_storage.data(),
                           emel::kernel::detail::dtype_f32, k_embd, k_vocab);

    backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    backend.matmul_actor = &matmul.actor;
    matmul.actor.process_event(
        emel::text::generator::matmul::event::configure_kernel_kind{
            backend.kernel_kind});
    backend.model = &model;
    backend.n_vocab = k_vocab;
    backend.n_embd = k_embd;
    backend.n_head = 1;
    backend.n_head_kv = 1;
    backend.n_layer = 1;
    backend.n_ctx = k_ctx;
    backend.kv_block_tokens = k_ctx;
    backend.kv_positions_capacity = k_ctx;
    backend.n_rot = k_embd;
    backend.head_dim = k_embd;
    backend.head_dim_kv = k_embd;
    backend.n_rep = 1;
    backend.rms_epsilon = 1.0e-5f;
    backend.rope_freq_base = 10000.0f;

    backend.token_embedding.tensor = &token_embedding_tensor;
    backend.token_embedding.rows = k_prompt_tokens;
    backend.token_embedding.cols = k_embd;
    backend.output_norm = output_norm_storage;
    backend.output.tensor = &packed_tensor;
    backend.output.rows = k_vocab;
    backend.output.cols = k_embd;
    backend.output_argmax.tensor = &output_argmax_tensor;
    backend.output_argmax.rows = k_vocab;
    backend.output_argmax.cols = k_embd;

    backend.blocks.resize(1u);
    auto &block = backend.blocks.front();
    block.attention_norm = attention_norm_storage;
    block.attention_q.tensor = &packed_tensor;
    block.attention_q.rows = k_embd;
    block.attention_q.cols = k_embd;
    block.attention_k = block.attention_q;
    block.attention_v = block.attention_q;
    block.attention_output = block.attention_q;
    block.feed_forward_norm = ffn_norm_storage;
    block.feed_forward_gate = block.attention_q;
    block.feed_forward_down = block.attention_q;
    block.feed_forward_up = block.attention_q;

    backend.bound_tokens.resize(k_prompt_tokens);
    backend.bound_positions.resize(k_prompt_tokens);
    backend.bound_logits.resize(static_cast<size_t>(k_vocab), -1.0f);
    backend.hidden.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.norm.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.q.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.q_attn.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.k.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.v.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.attn_scores.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_probs.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_probs_rounded.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_value_column.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_ctx.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.projected.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.gate.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.up.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.ffn_hidden.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.key_cache.resize(static_cast<size_t>(k_ctx * k_embd), 0u);
    backend.value_cache.resize(static_cast<size_t>(k_ctx * k_embd), 0u);
    backend.flash_key_cache.resize(static_cast<size_t>(k_ctx * k_embd), 0u);
    backend.flash_value_cache.resize(static_cast<size_t>(k_ctx * k_embd), 0u);
    backend.hidden_chunk4.resize(static_cast<size_t>(k_prompt_tokens * k_embd),
                                 0.0f);
    backend.norm_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.q_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.k_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.v_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.attn_ctx_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.projected_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.gate_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.up_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.ffn_hidden_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);

    logits.resize(static_cast<size_t>(k_vocab), -1.0f);
    topology.execution = &execution;
    plan.graph = &topology;
    plan.kind = emel::model::transformer::step_kind::prefill;
    plan.expected_outputs = 1;
    io.backend_ctx = &backend;
    io.token_ids = token_ids.data();
    io.token_count = k_prompt_tokens;
    io.logits = logits.data();
    io.logits_capacity = k_vocab;
    io.selected_token_out = &selected_token;
    io.selected_score_out = &selected_score;
    memory_snapshot.max_sequences = 1;
    memory_snapshot.block_tokens = k_ctx;
    memory_snapshot.sequence_active[0] = 1;
    memory_snapshot.sequence_length_values[0] = k_prompt_tokens;
    memory_snapshot.sequence_kv_block_count[0] = 1;
    memory_snapshot.sequence_kv_blocks[0][0] = 0;

    request.step_plan = &plan;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = k_prompt_tokens;
    request.kv_tokens = 0;
    request.memory_view = &memory_snapshot;
    request.seq_primary_ids = seq_primary_ids.data();
    request.seq_primary_ids_count =
        static_cast<int32_t>(seq_primary_ids.size());

    ready = emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
                zero_rows.data(), static_cast<uint64_t>(k_embd),
                static_cast<uint64_t>(k_embd), packed_storage.data()) &&
            emel::text::generator::detail::prepare_packed_q8_0_input_workspace(
                backend) &&
            emel::text::generator::detail::
                prepare_packed_q8_0_chunk4_input_workspace(backend);
  }
};

template <int32_t prompt_tokens, int32_t ctx_tokens = 8>
struct hybrid_chunked_q8_runtime_fixture {
  static constexpr int32_t k_vocab = static_cast<int32_t>(QK_K);
  static constexpr int32_t k_embd = static_cast<int32_t>(QK_K);
  static constexpr int32_t k_ctx = ctx_tokens;
  static constexpr int32_t k_prompt_tokens = prompt_tokens;
  static constexpr int32_t k_shortconv_kernel_size = 3;
  // Chunk work buffers are sized per gemm chunk (as prepare() does), not per
  // prompt: the chunked prepare/matmul helpers require the exact chunk-row
  // extent, and prompts longer than one chunk reuse the same buffers.
  static constexpr int32_t k_chunk4_rows = 4;
  static constexpr int32_t k_chunk8_rows = 8;

  emel::model::data model = {};
  matmul_actor_fixture matmul = {};
  emel::text::generator::detail::native_backend backend = {};
  std::vector<float> token_embedding_storage = {};
  std::vector<float> output_argmax_storage = {};
  std::vector<float> output_norm_storage = {};
  std::vector<float> attention_norm_storage = {};
  std::vector<float> ffn_norm_storage = {};
  std::vector<float> attention_q_norm_storage = {};
  std::vector<float> attention_k_norm_storage = {};
  std::vector<float> shortconv_conv_storage = {};
  std::vector<block_q4_k> square_rows = {};
  std::vector<block_q4_k> shortconv_in_rows = {};
  std::vector<uint8_t> packed_square_storage = {};
  std::vector<uint8_t> packed_shortconv_in_storage = {};
  emel::model::data::tensor_record token_embedding_tensor = {};
  emel::model::data::tensor_record packed_square_tensor = {};
  emel::model::data::tensor_record packed_shortconv_in_tensor = {};
  emel::model::data::tensor_record output_argmax_tensor = {};

  emel::model::transformer::execution_view execution = {};
  emel::model::transformer::topology topology = {};
  emel::model::transformer::step_plan plan = {};
  std::array<int32_t, k_prompt_tokens> token_ids = {};
  std::array<int32_t, k_prompt_tokens> positions = {};
  std::vector<float> logits = {};
  emel::text::generator::compute_io io = {};
  emel::memory::view::snapshot memory_snapshot = {};
  std::array<int32_t, 1> seq_primary_ids = {0};
  emel::graph::processor::event::execute request = {};
  bool ready = false;

  hybrid_chunked_q8_runtime_fixture() {
    std::memcpy(model.architecture_name.data(), "lfm2", 4u);
    bind_neox_rope_pairing(model);
    for (int32_t token = 0; token < k_prompt_tokens; ++token) {
      token_ids[static_cast<size_t>(token)] = token;
      positions[static_cast<size_t>(token)] = token;
    }
    token_embedding_storage.resize(
        static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
    for (int32_t token = 0; token < k_prompt_tokens; ++token) {
      token_embedding_storage[static_cast<size_t>(token) *
                                  static_cast<size_t>(k_embd) +
                              static_cast<size_t>(token)] = 1.0f;
    }
    output_argmax_storage.resize(static_cast<size_t>(k_vocab * k_embd), 0.0f);
    output_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    ffn_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_q_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_k_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    shortconv_conv_storage.assign(static_cast<size_t>(k_shortconv_kernel_size) *
                                      static_cast<size_t>(k_embd),
                                  0.0f);

    const size_t block_count =
        static_cast<size_t>(k_embd / static_cast<int32_t>(QK_K));
    square_rows.resize(static_cast<size_t>(k_embd) * block_count);
    shortconv_in_rows.resize(static_cast<size_t>(3 * k_embd) * block_count);
    packed_square_storage.resize(
        sizeof(emel::kernel::detail::quant::block_q4_kx8) *
        emel::kernel::detail::quant::packed_q4_k_x8_group_count(
            static_cast<uint64_t>(k_embd)) *
        block_count);
    packed_shortconv_in_storage.resize(
        sizeof(emel::kernel::detail::quant::block_q4_kx8) *
        emel::kernel::detail::quant::packed_q4_k_x8_group_count(
            static_cast<uint64_t>(3 * k_embd)) *
        block_count);

    token_embedding_tensor = make_tensor_record(token_embedding_storage.data(),
                                                emel::kernel::detail::dtype_f32,
                                                k_embd, k_prompt_tokens);
    packed_square_tensor = make_tensor_record(
        packed_square_storage.data(), emel::kernel::detail::dtype_q4_k_x8_bl8,
        k_embd, k_embd);
    packed_shortconv_in_tensor = make_tensor_record(
        packed_shortconv_in_storage.data(),
        emel::kernel::detail::dtype_q4_k_x8_bl8, k_embd, 3 * k_embd);
    output_argmax_tensor =
        make_tensor_record(output_argmax_storage.data(),
                           emel::kernel::detail::dtype_f32, k_embd, k_vocab);

    backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    backend.matmul_actor = &matmul.actor;
    matmul.actor.process_event(
        emel::text::generator::matmul::event::configure_kernel_kind{
            backend.kernel_kind});
    backend.model = &model;
    backend.n_vocab = k_vocab;
    backend.n_embd = k_embd;
    backend.n_head = 1;
    backend.n_head_kv = 1;
    backend.n_layer = 2;
    backend.n_ctx = k_ctx;
    backend.kv_block_tokens = k_ctx;
    backend.kv_positions_capacity = k_ctx;
    backend.n_rot = k_embd;
    backend.head_dim = k_embd;
    backend.head_dim_kv = k_embd;
    backend.n_rep = 1;
    backend.shortconv_kernel_size = k_shortconv_kernel_size;
    backend.shortconv_state_size = k_shortconv_kernel_size - 1;
    backend.rms_epsilon = 1.0e-5f;
    backend.rope_freq_base = 10000.0f;

    backend.token_embedding.tensor = &token_embedding_tensor;
    backend.token_embedding.rows = k_prompt_tokens;
    backend.token_embedding.cols = k_embd;
    backend.output_norm = output_norm_storage;
    backend.output.tensor = &packed_square_tensor;
    backend.output.rows = k_vocab;
    backend.output.cols = k_embd;
    backend.output_argmax.tensor = &output_argmax_tensor;
    backend.output_argmax.rows = k_vocab;
    backend.output_argmax.cols = k_embd;

    backend.blocks.resize(2u);
    auto &shortconv_block = backend.blocks[0];
    shortconv_block.residual_route =
        emel::model::transformer::generation_residual_route::shortconv;
    shortconv_block.attention_norm = attention_norm_storage;
    shortconv_block.shortconv_conv = shortconv_conv_storage;
    shortconv_block.shortconv_in_proj.tensor = &packed_shortconv_in_tensor;
    shortconv_block.shortconv_in_proj.rows = 3 * k_embd;
    shortconv_block.shortconv_in_proj.cols = k_embd;
    shortconv_block.shortconv_out_proj.tensor = &packed_square_tensor;
    shortconv_block.shortconv_out_proj.rows = k_embd;
    shortconv_block.shortconv_out_proj.cols = k_embd;
    shortconv_block.feed_forward_norm = ffn_norm_storage;
    shortconv_block.feed_forward_gate.tensor = &packed_square_tensor;
    shortconv_block.feed_forward_gate.rows = k_embd;
    shortconv_block.feed_forward_gate.cols = k_embd;
    shortconv_block.feed_forward_down = shortconv_block.feed_forward_gate;
    shortconv_block.feed_forward_up = shortconv_block.feed_forward_gate;

    auto &attention_block = backend.blocks[1];
    attention_block.attention_norm = attention_norm_storage;
    attention_block.attention_q.tensor = &packed_square_tensor;
    attention_block.attention_q.rows = k_embd;
    attention_block.attention_q.cols = k_embd;
    attention_block.attention_k = attention_block.attention_q;
    attention_block.attention_v = attention_block.attention_q;
    attention_block.attention_output = attention_block.attention_q;
    attention_block.attention_q_norm = attention_q_norm_storage;
    attention_block.attention_k_norm = attention_k_norm_storage;
    attention_block.attention_rope_pairing =
        emel::text::generator::detail::neox_rope_pairing();
    attention_block.feed_forward_norm = ffn_norm_storage;
    attention_block.feed_forward_gate = shortconv_block.feed_forward_gate;
    attention_block.feed_forward_down = shortconv_block.feed_forward_gate;
    attention_block.feed_forward_up = shortconv_block.feed_forward_gate;

    backend.bound_tokens.resize(k_prompt_tokens);
    backend.bound_positions.resize(k_prompt_tokens);
    backend.bound_logits.resize(static_cast<size_t>(k_vocab), -1.0f);
    backend.hidden.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.norm.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.shortconv_bcx.resize(static_cast<size_t>(3 * k_embd), 0.0f);
    backend.shortconv_bx.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.shortconv_conv_out.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.shortconv_bcx_chunk4.resize(
        static_cast<size_t>(k_chunk4_rows * 3 * k_embd), 0.0f);
    backend.shortconv_conv_out_chunk4.resize(
        static_cast<size_t>(k_chunk4_rows * k_embd), 0.0f);
    backend.q.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.q_attn.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.k.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.v.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.attn_scores.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_probs.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_probs_rounded.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_value_column.resize(static_cast<size_t>(k_ctx), 0.0f);
    backend.attn_ctx.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.projected.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.gate.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.up.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.ffn_hidden.resize(static_cast<size_t>(k_embd), 0.0f);
    backend.key_cache.resize(
        static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.value_cache.resize(
        static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.flash_key_cache.resize(
        static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.flash_value_cache.resize(
        static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.recurrent_shortconv_cache.resize(
        static_cast<size_t>(backend.n_layer * backend.shortconv_state_size *
                            k_embd),
        0.0f);
    backend.hidden_chunk4.resize(static_cast<size_t>(k_chunk4_rows * k_embd),
                                 0.0f);
    backend.hidden_chunk8.resize(static_cast<size_t>(k_chunk8_rows * k_embd),
                                 0.0f);
    backend.norm_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.norm_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.q_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.q_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.k_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.k_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.v_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.v_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.attn_ctx_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.attn_ctx_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.projected_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.projected_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.gate_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.gate_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.up_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.up_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.ffn_hidden_chunk4.resize(backend.hidden_chunk4.size(), 0.0f);
    backend.ffn_hidden_chunk8.resize(backend.hidden_chunk8.size(), 0.0f);
    backend.shortconv_bcx_chunk8.resize(
        static_cast<size_t>(k_chunk8_rows * 3 * k_embd), 0.0f);
    backend.shortconv_conv_out_chunk8.resize(
        static_cast<size_t>(k_chunk8_rows * k_embd), 0.0f);

    logits.resize(static_cast<size_t>(k_vocab), -1.0f);
    topology.execution = &execution;
    plan.graph = &topology;
    plan.kind = emel::model::transformer::step_kind::prefill;
    plan.expected_outputs = 1;
    io.backend_ctx = &backend;
    io.token_ids = token_ids.data();
    io.token_count = k_prompt_tokens;
    io.logits = logits.data();
    io.logits_capacity = k_vocab;
    memory_snapshot.max_sequences = 1;
    memory_snapshot.block_tokens = k_ctx;
    memory_snapshot.sequence_active[0] = 1;
    memory_snapshot.sequence_length_values[0] = k_prompt_tokens;
    memory_snapshot.sequence_kv_block_count[0] = 1;
    memory_snapshot.sequence_kv_blocks[0][0] = 0;

    request.step_plan = &plan;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = k_prompt_tokens;
    request.kv_tokens = 0;
    request.memory_view = &memory_snapshot;
    request.seq_primary_ids = seq_primary_ids.data();
    request.seq_primary_ids_count =
        static_cast<int32_t>(seq_primary_ids.size());

    ready =
        emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
            square_rows.data(), static_cast<uint64_t>(k_embd),
            static_cast<uint64_t>(k_embd), packed_square_storage.data()) &&
        emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
            shortconv_in_rows.data(), static_cast<uint64_t>(3 * k_embd),
            static_cast<uint64_t>(k_embd),
            packed_shortconv_in_storage.data()) &&
        emel::text::generator::detail::prepare_q8_input_workspace(backend) &&
        emel::text::generator::detail::prepare_q8_input_chunk4_workspace(
            backend) &&
        emel::text::generator::detail::prepare_q8_input_chunk8_workspace(
            backend);
  }
};

using hybrid_chunk4_q8_runtime_fixture = hybrid_chunked_q8_runtime_fixture<4>;
using hybrid_chunk8_q8_runtime_fixture = hybrid_chunked_q8_runtime_fixture<8>;

void fill_nonzero_q4_rows(std::span<block_q4_k> rows) {
  for (size_t row = 0; row < rows.size(); ++row) {
    auto &q4 = rows[row];
    q4.d = emel::kernel::detail::quant::fp32_to_fp16(
        0.0009765625f * static_cast<float>((row % 31u) + 1u));
    q4.dmin = emel::kernel::detail::quant::fp32_to_fp16(
        0.000244140625f * static_cast<float>(((row + 7u) % 17u) + 1u));
    for (size_t idx = 0; idx < q4.scales.size(); ++idx) {
      q4.scales[idx] =
          static_cast<uint8_t>(((row + 3u) * 13u + idx * 19u + 0x35u) & 0xffu);
    }
    for (size_t idx = 0; idx < q4.qs.size(); ++idx) {
      q4.qs[idx] =
          static_cast<uint8_t>(((row + 11u) * 17u + idx * 7u + 0x91u) & 0xffu);
    }
  }
}

template <class fixture_type>
bool seed_nonzero_hybrid_fixture_weights(fixture_type &fixture) {
  fill_nonzero_q4_rows(std::span<block_q4_k>(fixture.square_rows));
  fill_nonzero_q4_rows(std::span<block_q4_k>(fixture.shortconv_in_rows));
  return emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
             fixture.square_rows.data(), static_cast<uint64_t>(fixture.k_embd),
             static_cast<uint64_t>(fixture.k_embd),
             fixture.packed_square_storage.data()) &&
         emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
             fixture.shortconv_in_rows.data(),
             static_cast<uint64_t>(3 * fixture.k_embd),
             static_cast<uint64_t>(fixture.k_embd),
             fixture.packed_shortconv_in_storage.data());
}

float round_fp16_value(const float value) {
  return emel::text::generator::detail::quant::fp16_to_fp32(
      emel::text::generator::detail::quant::fp32_to_fp16(value));
}

void apply_qwen3_headwise_rms_norm(std::span<float> vector,
                                   std::span<const float> weights,
                                   const int32_t head_count,
                                   const int32_t head_dim,
                                   const float epsilon) {
  REQUIRE(head_count > 0);
  REQUIRE(head_dim > 0);
  REQUIRE(weights.size() == static_cast<size_t>(head_dim));
  for (int32_t head = 0; head < head_count; ++head) {
    const size_t head_offset =
        static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    float square_sum = 0.0f;
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      const float value = vector[head_offset + static_cast<size_t>(dim)];
      square_sum += value * value;
    }
    const float inv_rms =
        1.0f / std::sqrt(square_sum / static_cast<float>(head_dim) + epsilon);
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      vector[head_offset + static_cast<size_t>(dim)] *=
          inv_rms * weights[static_cast<size_t>(dim)];
    }
  }
}

void apply_rope_reference(std::span<float> vector, const int32_t head_count,
                          const int32_t head_dim, const int32_t n_rot,
                          const int32_t position, const float rope_freq_base) {
  const int32_t rot_dim = std::min(n_rot, head_dim);
  if (head_count <= 0 || head_dim <= 1 || rot_dim <= 1) {
    return;
  }

  const float theta_scale =
      ::powf(rope_freq_base, -2.0f / static_cast<float>(rot_dim));
  for (int32_t head = 0; head < head_count; ++head) {
    float *head_ptr = vector.data() + (static_cast<size_t>(head) *
                                       static_cast<size_t>(head_dim));
    float theta = static_cast<float>(position);
    for (int32_t dim = 0; dim + 1 < rot_dim; dim += 2) {
      const float cos_theta = ::cosf(theta);
      const float sin_theta = ::sinf(theta);
      const float x0 = head_ptr[dim];
      const float x1 = head_ptr[dim + 1];
      head_ptr[dim] = x0 * cos_theta - x1 * sin_theta;
      head_ptr[dim + 1] = x0 * sin_theta + x1 * cos_theta;
      theta *= theta_scale;
    }
  }
}

void apply_rope_neox_reference(std::span<float> vector,
                               const int32_t head_count, const int32_t head_dim,
                               const int32_t n_rot, const int32_t position,
                               const float rope_freq_base) {
  const int32_t rot_dim = std::min(n_rot, head_dim);
  if (head_count <= 0 || head_dim <= 1 || rot_dim <= 1) {
    return;
  }

  const float theta_scale =
      ::powf(rope_freq_base, -2.0f / static_cast<float>(rot_dim));
  const int32_t pair_stride = rot_dim / 2;
  for (int32_t head = 0; head < head_count; ++head) {
    float *head_ptr = vector.data() + (static_cast<size_t>(head) *
                                       static_cast<size_t>(head_dim));
    float theta = static_cast<float>(position);
    for (int32_t dim = 0; dim < pair_stride; ++dim) {
      const float cos_theta = ::cosf(theta);
      const float sin_theta = ::sinf(theta);
      const int32_t dim1 = dim + pair_stride;
      const float x0 = head_ptr[dim];
      const float x1 = head_ptr[dim1];
      head_ptr[dim] = x0 * cos_theta - x1 * sin_theta;
      head_ptr[dim1] = x0 * sin_theta + x1 * cos_theta;
      theta *= theta_scale;
    }
  }
}

std::vector<float> flash_attention_online_reference(
    const emel::text::generator::detail::native_backend &backend,
    const int32_t layer_index, const int32_t position,
    const std::span<const float> q_vector) {
  const int32_t head_count = backend.n_head;
  const auto &block = backend.blocks[static_cast<size_t>(layer_index)];
  const int32_t head_dim = block.attention_head_dim;
  const uint64_t kv_tokens = static_cast<uint64_t>(position + 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> out(
      static_cast<size_t>(head_count) * static_cast<size_t>(head_dim), 0.0f);
  std::vector<uint16_t> head_k(
      static_cast<size_t>(head_dim) * static_cast<size_t>(kv_tokens), 0u);
  std::vector<uint16_t> head_v(
      static_cast<size_t>(head_dim) * static_cast<size_t>(kv_tokens), 0u);

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset =
        static_cast<size_t>(head) * static_cast<size_t>(head_dim);

    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const size_t src_offset =
          emel::text::generator::detail::flash_layer_cache_head_position_offset(
              backend, block, layer_index, kv_head,
              static_cast<int32_t>(token));
      const size_t dst_offset =
          static_cast<size_t>(token) * static_cast<size_t>(head_dim);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const size_t dim_offset = static_cast<size_t>(dim);
        head_k[dst_offset + dim_offset] =
            backend.flash_key_cache[src_offset + dim_offset];
        head_v[dst_offset + dim_offset] =
            backend.flash_value_cache[src_offset + dim_offset];
      }
    }

    const std::vector<float> expected_head =
        flash_attn_reference_online_softmax_f16_values(
            q_vector.subspan(q_offset, static_cast<size_t>(head_dim)),
            std::span<const uint16_t>(head_k.data(), head_k.size()),
            std::span<const uint16_t>(head_v.data(), head_v.size()),
            static_cast<uint64_t>(head_dim), kv_tokens, scale);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      out[q_offset + static_cast<size_t>(dim)] =
          expected_head[static_cast<size_t>(dim)];
    }
  }

  return out;
}

} // namespace

TEST_CASE("generator_detail_fp16_to_fp32_handles_normal_special_and_subnormal_"
          "values") {
  CHECK(emel::text::generator::detail::quant::fp16_to_fp32(0x3c00u) ==
        doctest::Approx(1.0f));
  CHECK(emel::text::generator::detail::quant::fp16_to_fp32(0x3800u) ==
        doctest::Approx(0.5f));
  CHECK(
      std::isinf(emel::text::generator::detail::quant::fp16_to_fp32(0x7c00u)));
  CHECK(emel::text::generator::detail::quant::fp16_to_fp32(0x0001u) > 0.0f);
}

TEST_CASE("generator_detail_fp16_conversion_matches_native_arm_fp16_rounding") {
#if defined(__ARM_NEON) &&                                                     \
    !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) &&                    \
    !defined(__MUSACC__)
  constexpr std::array<float, 8> samples = {
      0.0f,      0.5f,     -0.934325f,     0.0345459f,
      -36.4516f, 65504.0f, 6.1035156e-05f, -6.1035156e-05f,
  };

  for (const float sample : samples) {
    uint16_t native_bits = 0u;
    const __fp16 native_value = sample;
    std::memcpy(&native_bits, &native_value, sizeof(native_bits));

    CHECK(emel::text::generator::detail::quant::fp32_to_fp16(sample) ==
          native_bits);
    CHECK(emel::text::generator::detail::quant::fp16_to_fp32(native_bits) ==
          doctest::Approx(static_cast<float>(native_value)));
  }
#else
  CHECK(true);
#endif
}

TEST_CASE("generator_detail_apply_rope_matches_ggml_float_recurrence") {
  std::array<float, 64> actual = {};
  std::array<float, 64> reference = {};
  for (size_t idx = 0; idx < actual.size(); ++idx) {
    actual[idx] = std::sin(static_cast<float>(idx) * 0.03125f) * 3.0f;
  }
  reference = actual;

  emel::text::generator::detail::apply_rope(actual, 1, 64, 64, 103, 10000.0f);
  apply_rope_reference(reference, 1, 64, 64, 103, 10000.0f);

  for (size_t idx = 0; idx < actual.size(); ++idx) {
    CHECK(actual[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-8));
  }
}

TEST_CASE("generator_detail_lfm2_attention_uses_neox_rope_layout") {
  constexpr int32_t k_embd = 64;
  constexpr int32_t k_position = 3;
  auto model = std::make_unique<emel::model::data>();
  std::memcpy(model->architecture_name.data(), "lfm2", 4u);

  std::vector<float> identity(
      static_cast<size_t>(k_embd) * static_cast<size_t>(k_embd), 0.0f);
  std::vector<float> zero_matrix(
      static_cast<size_t>(k_embd) * static_cast<size_t>(k_embd), 0.0f);
  for (int32_t idx = 0; idx < k_embd; ++idx) {
    identity[static_cast<size_t>(idx) * static_cast<size_t>(k_embd) +
             static_cast<size_t>(idx)] = 1.0f;
  }

  auto identity_tensor = make_tensor_record(
      identity.data(), emel::kernel::detail::dtype_f32, k_embd, k_embd);
  auto zero_tensor = make_tensor_record(
      zero_matrix.data(), emel::kernel::detail::dtype_f32, k_embd, k_embd);

  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  backend->model = model.get();
  backend->kernel_kind = emel::kernel::kernel_kind::x86_64;
  bind_test_matmul_actor(*backend, matmul);
  backend->n_embd = k_embd;
  backend->n_head = 1;
  backend->n_head_kv = 1;
  backend->n_layer = 1;
  backend->n_ctx = 8;
  backend->kv_block_tokens = 8;
  backend->kv_positions_capacity = 8;
  backend->n_rot = k_embd;
  backend->head_dim = k_embd;
  backend->head_dim_kv = k_embd;
  backend->max_q_dim = k_embd;
  backend->max_kv_dim = k_embd;
  backend->n_rep = 1;
  backend->rms_epsilon = 1.0e-6f;
  backend->rope_freq_base = 1000000.0f;
  backend->blocks.resize(1u);
  backend->layer_cache_offsets = {0u};
  backend->flash_layer_cache_offsets = {0u};
  backend->hidden.resize(k_embd);
  backend->norm.resize(k_embd);
  backend->q.resize(k_embd);
  backend->q_attn.resize(k_embd);
  backend->k.resize(k_embd);
  backend->v.resize(k_embd);
  backend->attn_scores.resize(backend->n_ctx);
  backend->attn_probs.resize(backend->n_ctx);
  backend->attn_probs_rounded.resize(backend->n_ctx);
  backend->attn_value_column.resize(backend->n_ctx);
  backend->attn_ctx.resize(k_embd);
  backend->projected.resize(k_embd);
  backend->gate.resize(k_embd);
  backend->up.resize(k_embd);
  backend->ffn_hidden.resize(k_embd);
  backend->key_cache.resize(static_cast<size_t>(backend->n_ctx) *
                            static_cast<size_t>(k_embd));
  backend->value_cache.resize(static_cast<size_t>(backend->n_ctx) *
                              static_cast<size_t>(k_embd));
  backend->flash_key_cache.resize(static_cast<size_t>(backend->n_ctx) *
                                  static_cast<size_t>(k_embd));
  backend->flash_value_cache.resize(static_cast<size_t>(backend->n_ctx) *
                                    static_cast<size_t>(k_embd));

  for (int32_t idx = 0; idx < k_embd; ++idx) {
    backend->hidden[static_cast<size_t>(idx)] =
        std::sin(static_cast<float>(idx + 1) * 0.03125f);
  }

  auto &block = backend->blocks.front();
  block.residual_route =
      emel::model::transformer::generation_residual_route::attention;
  block.attention_norm.assign(k_embd, 1.0f);
  block.attention_q.tensor = &identity_tensor;
  block.attention_q.rows = k_embd;
  block.attention_q.cols = k_embd;
  block.attention_k = block.attention_q;
  block.attention_v = block.attention_q;
  block.attention_output = block.attention_q;
  block.attention_q_norm.assign(k_embd, 1.0f);
  block.attention_k_norm.assign(k_embd, 1.0f);
  block.feed_forward_norm.assign(k_embd, 1.0f);
  block.feed_forward_gate.tensor = &zero_tensor;
  block.feed_forward_gate.rows = k_embd;
  block.feed_forward_gate.cols = k_embd;
  block.feed_forward_down = block.feed_forward_gate;
  block.feed_forward_up = block.feed_forward_gate;
  block.attention_q_dim = k_embd;
  block.attention_kv_dim = k_embd;
  block.attention_head_dim = k_embd;
  block.attention_head_dim_kv = k_embd;
  block.attention_rope_dim = k_embd;
  block.attention_rope_freq_base = backend->rope_freq_base;
  block.attention_rope_pairing =
      emel::text::generator::detail::neox_rope_pairing();

  std::array<float, k_embd> expected_k = {};
  REQUIRE(emel::text::generator::detail::rms_norm(
      backend->hidden, block.attention_norm, backend->rms_epsilon,
      std::span<float>(expected_k.data(), expected_k.size())));
  apply_qwen3_headwise_rms_norm(expected_k, block.attention_k_norm, 1, k_embd,
                                backend->rms_epsilon);
  apply_rope_neox_reference(expected_k, 1, k_embd, k_embd, k_position,
                            backend->rope_freq_base);

  REQUIRE(emel::text::generator::layer::run_layer_nonflash(*backend, 0,
                                                           k_position));
  const size_t cache_offset = emel::text::generator::detail::layer_cache_offset(
      *backend, block, 0, k_position);
  for (size_t idx = 0; idx < expected_k.size(); ++idx) {
    CHECK(backend->k[idx] == doctest::Approx(expected_k[idx]).epsilon(1.0e-5));
    CHECK(emel::text::generator::detail::quant::fp16_to_fp32(
              backend->key_cache[cache_offset + idx]) ==
          doctest::Approx(round_fp16_value(expected_k[idx])).epsilon(1.0e-5));
  }
}

TEST_CASE("generator_detail_dequantizes_q2_k_blocks") {
  block_q2_k block = {};
  block.d = 0x3c00u;
  block.dmin = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(),
            static_cast<uint8_t>(0x11u));
  std::fill(block.qs.begin(), block.qs.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::text::generator::detail::quant::dequantize_row_q2_k(&block, out.data(),
                                                            QK_K);

  CHECK(out.front() == doctest::Approx(-1.0f));
  CHECK(out[127] == doctest::Approx(-1.0f));
  CHECK(out.back() == doctest::Approx(-1.0f));
}

TEST_CASE("generator_detail_dequantizes_q3_k_blocks_without_unsigned_wrap") {
  block_q3_k block = {};
  block.d = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(),
            static_cast<uint8_t>(0x00u));
  std::fill(block.hmask.begin(), block.hmask.end(),
            static_cast<uint8_t>(0x00u));
  std::fill(block.qs.begin(), block.qs.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::text::generator::detail::quant::dequantize_row_q3_k(&block, out.data(),
                                                            QK_K);

  CHECK(std::isfinite(out.front()));
  CHECK(out.front() == doctest::Approx(128.0f));
  CHECK(out[127] == doctest::Approx(128.0f));
  CHECK(out.back() == doctest::Approx(128.0f));
}

TEST_CASE("generator_detail_dequantizes_q6_k_blocks") {
  block_q6_k block = {};
  block.d = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(), static_cast<int8_t>(1));
  std::fill(block.ql.begin(), block.ql.end(), static_cast<uint8_t>(0x00u));
  std::fill(block.qh.begin(), block.qh.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::text::generator::detail::quant::dequantize_row_q6_k(&block, out.data(),
                                                            QK_K);

  CHECK(out.front() == doctest::Approx(-32.0f));
  CHECK(out[127] == doctest::Approx(-32.0f));
  CHECK(out.back() == doctest::Approx(-32.0f));
}

TEST_CASE("generator_detail_dequantizes_q8_0_blocks") {
  block_q8_0 block = {};
  block.d = emel::text::generator::detail::quant::fp32_to_fp16(0.5f);
  for (size_t idx = 0; idx < block.qs.size(); ++idx) {
    block.qs[idx] = static_cast<int8_t>((static_cast<int32_t>(idx % 7u)) - 3);
  }

  std::array<float, QK8_0> out = {};
  emel::text::generator::detail::quant::dequantize_row_q8_0(&block, out.data(),
                                                            QK8_0);

  CHECK(out.front() == doctest::Approx(-1.5f));
  CHECK(out[1] == doctest::Approx(-1.0f));
  CHECK(out[2] == doctest::Approx(-0.5f));
  CHECK(out.back() == doctest::Approx(0.0f));
}

TEST_CASE("generator_detail_copies_q4_0_tensor_rows") {
  std::array<block_q4_0, 2> rows = {};
  rows[0].d = emel::text::generator::detail::quant::fp32_to_fp16(0.25f);
  rows[1].d = emel::text::generator::detail::quant::fp32_to_fp16(0.5f);
  for (size_t idx = 0; idx < rows[1].qs.size(); ++idx) {
    rows[1].qs[idx] =
        static_cast<uint8_t>((idx & 0x0fu) | (((idx + 1u) & 0x0fu) << 4u));
  }

  auto tensor =
      make_tensor_record(rows.data(), emel::kernel::detail::dtype_q4_0,
                         static_cast<int32_t>(QK4_0), 2);
  std::array<float, QK4_0> expected = {};
  std::array<float, QK4_0> copied = {};
  emel::text::generator::detail::quant::dequantize_row_q4_0(
      rows.data() + 1u, expected.data(), static_cast<int64_t>(expected.size()));

  REQUIRE(emel::text::generator::detail::copy_tensor_row(tensor, 1, copied));
  for (size_t idx = 0; idx < copied.size(); ++idx) {
    CHECK(copied[idx] == doctest::Approx(expected[idx]));
  }
}

TEST_CASE(
    "generator_detail_prepares_explicit_logits_routes_and_support_predicates") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::text::generator::detail::native_backend backend{};
  matmul_actor_fixture matmul = {};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(backend, matmul);
  backend.n_embd = static_cast<int32_t>(QK_K);
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_DOTPROD)
  CHECK(
      emel::text::generator::detail::packed_q6_k_x8_logits_supported(backend));
#else
  CHECK_FALSE(
      emel::text::generator::detail::packed_q6_k_x8_logits_supported(backend));
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(emel::text::generator::detail::prepared_q6_k_x8_q8_logits_supported(
      backend));
#endif
  CHECK_FALSE(emel::text::generator::detail::q8_input_workspace_candidate({}));
  CHECK_FALSE(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));

  REQUIRE(emel::text::generator::detail::prepare_output_logits(backend));
  REQUIRE(emel::text::generator::detail::prepare_q8_input_workspace(backend));
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    (defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_DOTPROD))
  CHECK(backend.q8_input_storage.size() == 1u);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, backend.output));
  CHECK(emel::text::generator::guard::detail::q8_input_argmax_path_supported(
      backend, backend.output_argmax));
#else
  CHECK(backend.q8_input_storage.empty());
  CHECK_FALSE(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, backend.output));
  CHECK_FALSE(
      emel::text::generator::guard::detail::q8_input_argmax_path_supported(
          backend, backend.output_argmax));
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared);
  CHECK(emel::text::generator::detail::row_storage_bytes(
            *backend.output.tensor, static_cast<int32_t>(QK_K)) ==
        emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(
            QK_K));
  CHECK(emel::text::generator::detail::row_storage_bytes(
            *backend.output_argmax.tensor, static_cast<int32_t>(QK_K)) ==
        emel::kernel::detail::quant::
            argmax_prepared_q6_k_x8_q8_group_storage_bytes(QK_K));
  CHECK(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));
#elif defined(__aarch64__) && defined(__ARM_NEON) &&                           \
    defined(__ARM_FEATURE_DOTPROD)
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));
#else
  CHECK(backend.output.tensor == &q6_tensor);
  CHECK(backend.output_argmax.tensor == &q6_tensor);
  CHECK_FALSE(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));
#endif
}

TEST_CASE("generator_detail_explicit_logits_routes_cover_packed_and_"
          "passthrough_helpers") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::text::generator::detail::native_backend backend{};
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  emel::text::generator::detail::reset_output_logits(backend);
  CHECK(backend.output.tensor == &q6_tensor);
  CHECK(backend.output_argmax.tensor == &q6_tensor);
  CHECK(backend.output_packed_storage.empty());
  CHECK(backend.output_prepared_storage.empty());
  CHECK(backend.output_argmax_prepared_storage.empty());

  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  REQUIRE(emel::text::generator::detail::prepare_packed_output_logits(backend));
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::text::generator::detail::matrix_buffer_bytes(backend.output) ==
        emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(QK_K));

  std::array<float, 8> f32_rows = {0.25f, 0.5f, 0.75f, 1.0f,
                                   1.25f, 1.5f, 1.75f, 2.0f};
  auto f32_tensor = make_tensor_record(f32_rows.data(),
                                       emel::kernel::detail::dtype_f32, 4, 2);
  emel::text::generator::detail::native_backend passthrough{};
  passthrough.output_native.tensor = &f32_tensor;
  passthrough.output_native.cols = 4;
  passthrough.output_native.rows = 2;
  passthrough.output = passthrough.output_native;
  passthrough.output_argmax = passthrough.output_native;
  REQUIRE(emel::text::generator::detail::prepare_output_logits(passthrough));
  CHECK(passthrough.output.tensor == &f32_tensor);
  CHECK(passthrough.output_argmax.tensor == &f32_tensor);
}

TEST_CASE("generator_detail_routes_static_q6_block_matrices_through_generic_q8_"
          "input") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::text::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(
      emel::text::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));

  emel::text::generator::detail::native_backend backend{};
  matmul_actor_fixture matmul = {};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(backend, matmul);
  backend.blocks.resize(1u);
  auto &block = backend.blocks.front();
  block.residual_route =
      emel::model::transformer::generation_residual_route::attention;
  block.attention_q = q6_matrix;
  block.attention_k = q6_matrix;
  block.attention_v = q6_matrix;
  block.attention_output = q6_matrix;
  block.feed_forward_gate = q6_matrix;
  block.feed_forward_down = q6_matrix;
  block.feed_forward_up = q6_matrix;

  REQUIRE(
      emel::text::generator::detail::prepare_block_native_matrices(backend));
  REQUIRE(emel::text::generator::detail::prepare_q8_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);

  std::array<float, QK_K> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    input[idx] =
        static_cast<float>(static_cast<int32_t>((idx * 7u) % 19u) - 9) * 0.125f;
  }

  emel::text::generator::detail::native_backend reference_backend{};
  matmul_actor_fixture reference_matmul = {};
  reference_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(reference_backend, reference_matmul);
  std::vector<float> reference(Q6_K_X8_ROWS, 0.0f);
  REQUIRE(emel::text::generator::detail::matmul_vector(
      reference_backend, q6_matrix,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(reference.data(), reference.size())));

  std::vector<float> output(Q6_K_X8_ROWS, 0.0f);
  REQUIRE(emel::text::generator::detail::matmul_vector_q8_k(
      backend, block.feed_forward_down,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));

  for (size_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    CHECK(output[row] == doctest::Approx(reference[row]).epsilon(1.0e-5f));
  }
#elif defined(__aarch64__) && defined(__ARM_NEON) &&                           \
    defined(__ARM_FEATURE_DOTPROD)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);
#else
  CHECK(block.feed_forward_down.tensor == &q6_tensor);
  CHECK_FALSE(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.empty());
#endif
}

TEST_CASE("generator_detail_routes_static_q4_block_matrices_through_generic_q8_"
          "input") {
  auto q4_rows = make_q4_rows();
  auto q4_tensor = make_tensor_record(
      q4_rows.data(), emel::kernel::detail::dtype_q4_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q4_K_X8_ROWS));

  emel::text::generator::detail::tensor_matrix q4_matrix = {};
  REQUIRE(
      emel::text::generator::detail::bind_tensor_rows(q4_tensor, q4_matrix));

  emel::text::generator::detail::native_backend backend{};
  matmul_actor_fixture matmul = {};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(backend, matmul);
  backend.blocks.resize(1u);
  auto &block = backend.blocks.front();
  block.residual_route =
      emel::model::transformer::generation_residual_route::attention;
  block.attention_q = q4_matrix;
  block.attention_k = q4_matrix;
  block.attention_v = q4_matrix;
  block.attention_output = q4_matrix;
  block.feed_forward_gate = q4_matrix;
  block.feed_forward_down = q4_matrix;
  block.feed_forward_up = q4_matrix;

  REQUIRE(
      emel::text::generator::detail::prepare_block_native_matrices(backend));
  REQUIRE(emel::text::generator::detail::prepare_q8_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q4_k_x8_bl8);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);

  std::array<float, QK_K> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    input[idx] =
        static_cast<float>(static_cast<int32_t>((idx * 5u) % 21u) - 10) *
        0.125f;
  }

  emel::text::generator::detail::native_backend reference_backend{};
  matmul_actor_fixture reference_matmul = {};
  reference_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(reference_backend, reference_matmul);
  std::vector<float> reference(Q4_K_X8_ROWS, 0.0f);
  REQUIRE(emel::text::generator::detail::matmul_vector(
      reference_backend, q4_matrix,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(reference.data(), reference.size())));

  std::vector<float> output(Q4_K_X8_ROWS, 0.0f);
  REQUIRE(emel::text::generator::detail::matmul_vector_q8_k(
      backend, block.feed_forward_down,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));

  for (size_t row = 0; row < Q4_K_X8_ROWS; ++row) {
    CHECK(output[row] == doctest::Approx(reference[row]).epsilon(1.0e-5f));
  }
#elif defined(__aarch64__) && defined(__ARM_NEON) &&                           \
    defined(__ARM_FEATURE_DOTPROD)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q4_k_x8_bl4);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);
#else
  CHECK(block.feed_forward_down.tensor == &q4_tensor);
  CHECK_FALSE(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.empty());
#endif
}

TEST_CASE("generator_detail_q6_logits_paths_slice_oversized_q8_workspace") {
#if !(defined(__aarch64__) && defined(__ARM_NEON))
  CHECK(true);
#else
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::text::generator::detail::native_backend backend{};
  matmul_actor_fixture matmul = {};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(backend, matmul);
  backend.n_embd = static_cast<int32_t>(QK_K);
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;
  backend.hidden.resize(QK_K, 0.0f);
  backend.norm.resize(QK_K, 0.0f);
  backend.output_norm.resize(QK_K, 1.0f);
  backend.bound_logits.resize(Q6_K_X8_ROWS, 0.0f);
  for (size_t idx = 0; idx < backend.hidden.size(); ++idx) {
    backend.hidden[idx] =
        static_cast<float>(static_cast<int32_t>((idx * 9u) % 23u) - 11) *
        0.0625f;
  }

  REQUIRE(emel::text::generator::detail::prepare_output_logits(backend));
  backend.q8_input_storage.resize(2u);
  REQUIRE(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, backend.output));
  REQUIRE(emel::text::generator::guard::detail::q8_input_argmax_path_supported(
      backend, backend.output_argmax));
  REQUIRE(emel::text::generator::detail::compute_logits<
          emel::text::generator::detail::scalar_matmul_route::q8_k>(backend));
  std::fill(backend.bound_logits.begin(), backend.bound_logits.end(), 0.0f);
  REQUIRE(emel::text::generator::detail::compute_logits<
          emel::text::generator::detail::scalar_matmul_route::
              native_quantized_q8_k_logits>(backend));

  int32_t selected_index = -1;
  float selected_score = 0.0f;
  REQUIRE(emel::text::generator::detail::compute_logits_preselected_argmax<
          emel::text::generator::detail::scalar_argmax_route::q8_k>(
      backend, selected_index, selected_score));
  CHECK(selected_index >= 0);
  CHECK(selected_index < static_cast<int32_t>(Q6_K_X8_ROWS));
#endif
}

TEST_CASE("generator_detail_run_kernel_callbacks_do_not_write_error_channel") {
  namespace gen_detail = emel::text::generator::detail;

  auto fixture = std::make_unique<runtime_request_fixture>();
  int32_t err = -123;

  REQUIRE(gen_detail::bind_guarded_inputs(fixture->request, nullptr));
  const auto expect_rejected_without_error_write = [&](auto fn) {
    err = -123;
    CHECK_FALSE(fn(fixture->request, &err));
    CHECK(err == -123);
  };

  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_native_quantized);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_native_quantized_q8_k_logits);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_scalar_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_scalar_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_scalar_native_quantized);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_nonflash_prefill_scalar_native_quantized_q8_k_logits);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_scalar_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_decode_packed_q8_0);
  expect_rejected_without_error_write(gen_detail::run_kernel_flash_decode_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_decode_native_quantized);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_decode_native_quantized_q8_k_logits);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_decode_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_native_quantized);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_native_quantized_q8_k_logits);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_chunk8_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_chunk8_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_chunk4_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_chunk4_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_chunk4_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_chunk4_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_flash_prefill_scalar_preselected_argmax_native_quantized_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_flash_prefill_scalar_preselected_argmax_native_quantized_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_scalar_preselected_argmax_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_scalar_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_nonflash_prefill_scalar_preselected_argmax_native_quantized_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_nonflash_prefill_scalar_preselected_argmax_native_quantized_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_scalar_preselected_argmax_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_decode_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_flash_decode_preselected_argmax_native_quantized_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_flash_decode_preselected_argmax_native_quantized_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_decode_preselected_argmax_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_nonflash_decode_preselected_argmax_native_quantized_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_nonflash_decode_preselected_argmax_native_quantized_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_decode_preselected_argmax_kernel);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_chunk8_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_chunk8_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_flash_prefill_chunk4_preselected_argmax_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::
          run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_flash_prefill_chunk4_preselected_argmax_q8_k);
  expect_rejected_without_error_write(
      gen_detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_q8_k);
}

TEST_CASE("generator_detail_decode_preconditions_reject_malformed_requests") {
  namespace gen_detail = emel::text::generator::detail;

  gen_detail::native_backend backend{};
  backend.n_ctx = 8;
  backend.kv_block_tokens = 8;
  backend.kv_positions_capacity = 8;
  backend.token_embedding.rows = 4;
  backend.bound_tokens.resize(1u);
  backend.bound_positions.resize(1u);
  emel::graph::processor::event::execute request{};
  request.kv_tokens = 0;
  const auto decode_rejected = [&]() {
    return gen_detail::run_decode<
        emel::text::generator::attention_mode::nonflash,
        gen_detail::scalar_matmul_route::kernel>(backend, request);
  };

  CHECK_FALSE(decode_rejected());
  backend.bound_token_count = 1;
  CHECK_FALSE(decode_rejected());
  backend.bound_position_count = 1;
  request.kv_tokens = -1;
  CHECK_FALSE(decode_rejected());
  request.kv_tokens = 0;
  backend.kv_cache_tokens = 1;
  CHECK_FALSE(decode_rejected());
  backend.kv_cache_tokens = 0;
  backend.bound_tokens[0] = -1;
  CHECK_FALSE(decode_rejected());
  backend.bound_tokens[0] = backend.token_embedding.rows;
  CHECK_FALSE(decode_rejected());
  backend.bound_tokens[0] = 0;
  backend.bound_positions[0] = -1;
  CHECK_FALSE(decode_rejected());
  backend.bound_positions[0] = backend.n_ctx;
  CHECK_FALSE(decode_rejected());
  backend.bound_positions[0] = 0;
  emel::model::data::tensor_record token_embedding{};
  backend.token_embedding.tensor = &token_embedding;
  backend.hidden.resize(4u);
  CHECK_FALSE(decode_rejected());

  const auto flash_decode_rejected = [&]() {
    return gen_detail::run_decode<emel::text::generator::attention_mode::flash,
                                  gen_detail::scalar_matmul_route::kernel>(
        backend, request);
  };
  backend.bound_token_count = 0;
  CHECK_FALSE(flash_decode_rejected());
  backend.bound_token_count = 1;
  backend.bound_position_count = 0;
  CHECK_FALSE(flash_decode_rejected());
  backend.bound_position_count = 1;
  request.kv_tokens = -1;
  CHECK_FALSE(flash_decode_rejected());
  request.kv_tokens = 0;
  backend.kv_cache_tokens = 1;
  CHECK_FALSE(flash_decode_rejected());
  backend.kv_cache_tokens = 0;

  int32_t selected_index = -1;
  float selected_score = 0.0f;
  const auto preselected_rejected = [&]() {
    return gen_detail::run_decode_preselected_argmax<
        emel::text::generator::attention_mode::nonflash,
        gen_detail::scalar_matmul_route::kernel,
        gen_detail::scalar_argmax_route::kernel>(
        backend, request, selected_index, selected_score);
  };

  backend.bound_token_count = 0;
  CHECK_FALSE(preselected_rejected());
  backend.bound_token_count = 1;
  backend.bound_position_count = 0;
  CHECK_FALSE(preselected_rejected());
  backend.bound_position_count = 1;
  request.kv_tokens = -1;
  CHECK_FALSE(preselected_rejected());
  request.kv_tokens = 0;
  backend.kv_cache_tokens = 1;
  CHECK_FALSE(preselected_rejected());
  backend.kv_cache_tokens = 0;
  backend.bound_tokens[0] = -1;
  CHECK_FALSE(preselected_rejected());
  backend.bound_tokens[0] = backend.token_embedding.rows;
  CHECK_FALSE(preselected_rejected());
  backend.bound_tokens[0] = 0;
  backend.bound_positions[0] = -1;
  CHECK_FALSE(preselected_rejected());
  backend.bound_positions[0] = backend.n_ctx;
  CHECK_FALSE(preselected_rejected());
  backend.bound_positions[0] = 0;
  CHECK_FALSE(preselected_rejected());
}

TEST_CASE("generator_detail_route_templates_reject_unprepared_inputs") {
  using emel::text::generator::attention_mode;
  using emel::text::generator::detail::scalar_argmax_route;
  using emel::text::generator::detail::scalar_matmul_route;

  emel::text::generator::detail::native_backend backend{};
  matmul_actor_fixture matmul = {};
  bind_test_matmul_actor(backend, matmul);
  backend.n_embd = 4;
  backend.n_head = 1;
  backend.n_head_kv = 1;
  backend.n_layer = 1;
  backend.n_vocab = 4;
  backend.n_ctx = 4;
  backend.kv_block_tokens = 4;
  backend.kv_positions_capacity = 4;
  backend.head_dim = 4;
  backend.head_dim_kv = 4;
  backend.blocks.resize(1u);

  CHECK_FALSE(
      emel::text::generator::layer::run_layer<attention_mode::flash,
                                              scalar_matmul_route::packed_q8_0>(
          backend, 0, 0));
  CHECK_FALSE(emel::text::generator::layer::run_layer<
              attention_mode::flash, scalar_matmul_route::q8_k>(backend, 0, 0));
  CHECK_FALSE(emel::text::generator::layer::run_layer<
              attention_mode::flash, scalar_matmul_route::native_quantized>(
      backend, 0, 0));
  CHECK_FALSE(
      emel::text::generator::layer::run_layer<
          attention_mode::flash,
          scalar_matmul_route::native_quantized_q8_k_logits>(backend, 0, 0));
  CHECK_FALSE(
      emel::text::generator::layer::run_layer<attention_mode::flash,
                                              scalar_matmul_route::kernel>(
          backend, 0, 0));
  CHECK_FALSE(
      emel::text::generator::layer::run_layer<attention_mode::nonflash,
                                              scalar_matmul_route::packed_q8_0>(
          backend, 0, 0));
  CHECK_FALSE(
      emel::text::generator::layer::run_layer<attention_mode::nonflash,
                                              scalar_matmul_route::q8_k>(
          backend, 0, 0));
  CHECK_FALSE(emel::text::generator::layer::run_layer<
              attention_mode::nonflash, scalar_matmul_route::native_quantized>(
      backend, 0, 0));
  CHECK_FALSE(
      emel::text::generator::layer::run_layer<
          attention_mode::nonflash,
          scalar_matmul_route::native_quantized_q8_k_logits>(backend, 0, 0));
  CHECK_FALSE(
      emel::text::generator::layer::run_layer<attention_mode::nonflash,
                                              scalar_matmul_route::kernel>(
          backend, 0, 0));

  CHECK_FALSE(emel::text::generator::detail::compute_logits<
              scalar_matmul_route::packed_q8_0>(backend));
  CHECK_FALSE(
      emel::text::generator::detail::compute_logits<scalar_matmul_route::q8_k>(
          backend));
  CHECK_FALSE(emel::text::generator::detail::compute_logits<
              scalar_matmul_route::native_quantized>(backend));
  CHECK_FALSE(emel::text::generator::detail::compute_logits<
              scalar_matmul_route::native_quantized_q8_k_logits>(backend));
  CHECK_FALSE(emel::text::generator::detail::compute_logits<
              scalar_matmul_route::kernel>(backend));

  int32_t selected_index = -1;
  float selected_score = 0.0f;
  CHECK_FALSE(
      emel::text::generator::detail::compute_logits_preselected_argmax<
          scalar_argmax_route::q8_k>(backend, selected_index, selected_score));
  CHECK_FALSE(
      emel::text::generator::detail::compute_logits_preselected_argmax<
          scalar_argmax_route::q8_k>(backend, selected_index, selected_score));
  CHECK_FALSE(emel::text::generator::detail::compute_logits_preselected_argmax<
              scalar_argmax_route::kernel>(backend, selected_index,
                                           selected_score));
  CHECK_FALSE(emel::text::generator::detail::compute_logits_preselected_argmax<
              scalar_argmax_route::kernel>(backend, selected_index,
                                           selected_score));

  emel::graph::processor::event::execute request{};
  request.kv_tokens = -1;
  CHECK_FALSE(emel::text::generator::detail::run_decode<
              attention_mode::flash, scalar_matmul_route::packed_q8_0>(
      backend, request));
  CHECK_FALSE(
      emel::text::generator::detail::run_decode<attention_mode::flash,
                                                scalar_matmul_route::q8_k>(
          backend, request));
  CHECK_FALSE(emel::text::generator::detail::run_decode<
              attention_mode::flash, scalar_matmul_route::native_quantized>(
      backend, request));
  CHECK_FALSE(
      emel::text::generator::detail::run_decode<
          attention_mode::flash,
          scalar_matmul_route::native_quantized_q8_k_logits>(backend, request));
  CHECK_FALSE(
      emel::text::generator::detail::run_decode<attention_mode::flash,
                                                scalar_matmul_route::kernel>(
          backend, request));
  CHECK_FALSE(emel::text::generator::detail::run_decode<
              attention_mode::nonflash, scalar_matmul_route::packed_q8_0>(
      backend, request));
  CHECK_FALSE(
      emel::text::generator::detail::run_decode<attention_mode::nonflash,
                                                scalar_matmul_route::q8_k>(
          backend, request));
  CHECK_FALSE(emel::text::generator::detail::run_decode<
              attention_mode::nonflash, scalar_matmul_route::native_quantized>(
      backend, request));
  CHECK_FALSE(
      emel::text::generator::detail::run_decode<
          attention_mode::nonflash,
          scalar_matmul_route::native_quantized_q8_k_logits>(backend, request));
  CHECK_FALSE(
      emel::text::generator::detail::run_decode<attention_mode::nonflash,
                                                scalar_matmul_route::kernel>(
          backend, request));

  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::flash, scalar_matmul_route::q8_k,
              scalar_argmax_route::q8_k>(backend, request, selected_index,
                                         selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::flash, scalar_matmul_route::native_quantized,
              scalar_argmax_route::q8_k>(backend, request, selected_index,
                                         selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::flash, scalar_matmul_route::native_quantized,
              scalar_argmax_route::kernel>(backend, request, selected_index,
                                           selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::flash, scalar_matmul_route::kernel,
              scalar_argmax_route::kernel>(backend, request, selected_index,
                                           selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::nonflash, scalar_matmul_route::q8_k,
              scalar_argmax_route::q8_k>(backend, request, selected_index,
                                         selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::nonflash, scalar_matmul_route::native_quantized,
              scalar_argmax_route::q8_k>(backend, request, selected_index,
                                         selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::nonflash, scalar_matmul_route::native_quantized,
              scalar_argmax_route::kernel>(backend, request, selected_index,
                                           selected_score));
  CHECK_FALSE(emel::text::generator::detail::run_decode_preselected_argmax<
              attention_mode::nonflash, scalar_matmul_route::kernel,
              scalar_argmax_route::kernel>(backend, request, selected_index,
                                           selected_score));

  backend.bound_tokens = {-1};
  backend.bound_positions = {0};
  backend.bound_token_count = 1;
  backend.bound_position_count = 1;
  CHECK_FALSE(emel::text::generator::detail::run_prefill_scalar_tokens<
              attention_mode::flash, scalar_matmul_route::packed_q8_0>(backend,
                                                                       0u, 1u));
  CHECK_FALSE(
      emel::text::generator::detail::run_prefill_scalar_tokens<
          attention_mode::flash, scalar_matmul_route::q8_k>(backend, 0u, 1u));
  CHECK_FALSE(emel::text::generator::detail::run_prefill_scalar_tokens<
              attention_mode::flash, scalar_matmul_route::native_quantized>(
      backend, 0u, 1u));
  CHECK_FALSE(
      emel::text::generator::detail::run_prefill_scalar_tokens<
          attention_mode::flash,
          scalar_matmul_route::native_quantized_q8_k_logits>(backend, 0u, 1u));
  CHECK_FALSE(
      emel::text::generator::detail::run_prefill_scalar_tokens<
          attention_mode::flash, scalar_matmul_route::kernel>(backend, 0u, 1u));
  CHECK_FALSE(emel::text::generator::detail::run_prefill_scalar_tokens<
              attention_mode::nonflash, scalar_matmul_route::packed_q8_0>(
      backend, 0u, 1u));
  CHECK_FALSE(emel::text::generator::detail::run_prefill_scalar_tokens<
              attention_mode::nonflash, scalar_matmul_route::q8_k>(backend, 0u,
                                                                   1u));
  CHECK_FALSE(emel::text::generator::detail::run_prefill_scalar_tokens<
              attention_mode::nonflash, scalar_matmul_route::native_quantized>(
      backend, 0u, 1u));
  CHECK_FALSE(
      emel::text::generator::detail::run_prefill_scalar_tokens<
          attention_mode::nonflash,
          scalar_matmul_route::native_quantized_q8_k_logits>(backend, 0u, 1u));
  CHECK_FALSE(emel::text::generator::detail::run_prefill_scalar_tokens<
              attention_mode::nonflash, scalar_matmul_route::kernel>(backend,
                                                                     0u, 1u));

  emel::text::generator::detail::block_weights invalid_shortconv{};
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::packed_q8_0>(backend, invalid_shortconv, 0));
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::q8_k>(backend, invalid_shortconv, 0));
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::native_quantized>(backend, invalid_shortconv,
                                                     0));
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::native_quantized_q8_k_logits>(
      backend, invalid_shortconv, 0));
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::kernel>(backend, invalid_shortconv, 0));

  std::array<float, 48> shortconv_in = {};
  std::array<float, 16> shortconv_out = {};
  for (size_t idx = 0; idx < shortconv_in.size(); ++idx) {
    shortconv_in[idx] = idx % 5u == 0u ? 1.0f : 0.0f;
  }
  for (size_t idx = 0; idx < shortconv_out.size(); ++idx) {
    shortconv_out[idx] = idx % 5u == 0u ? 1.0f : 0.0f;
  }
  auto shortconv_in_tensor = make_tensor_record(
      shortconv_in.data(), emel::kernel::detail::dtype_f32, 4, 12);
  auto shortconv_out_tensor = make_tensor_record(
      shortconv_out.data(), emel::kernel::detail::dtype_f32, 4, 4);
  emel::text::generator::detail::block_weights shortconv_block{};
  shortconv_block.shortconv_in_proj.tensor = &shortconv_in_tensor;
  shortconv_block.shortconv_in_proj.rows = 12;
  shortconv_block.shortconv_in_proj.cols = 4;
  shortconv_block.shortconv_out_proj.tensor = &shortconv_out_tensor;
  shortconv_block.shortconv_out_proj.rows = 4;
  shortconv_block.shortconv_out_proj.cols = 4;
  shortconv_block.shortconv_conv.assign(12u, 0.0f);
  shortconv_block.shortconv_conv[2] = 1.0f;
  backend.shortconv_kernel_size = 3;
  backend.shortconv_state_size = 2;
  backend.norm.assign(4u, 1.0f);
  backend.shortconv_bcx.assign(12u, 0.0f);
  backend.shortconv_bx.assign(4u, 0.0f);
  backend.shortconv_conv_out.assign(4u, 0.0f);
  backend.recurrent_shortconv_cache.assign(8u, 0.0f);
  backend.projected.assign(4u, 0.0f);
  backend.hidden.assign(4u, 0.0f);
  CHECK(emel::text::generator::detail::run_shortconv_block<
        scalar_matmul_route::kernel>(backend, shortconv_block, 0));
  CHECK(emel::text::generator::detail::run_shortconv_block<
        scalar_matmul_route::native_quantized>(backend, shortconv_block, 0));
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::packed_q8_0>(backend, shortconv_block, 0));
  CHECK_FALSE(emel::text::generator::detail::run_shortconv_block<
              scalar_matmul_route::q8_k>(backend, shortconv_block, 0));
}

TEST_CASE("generator_detail_scalar_routes_run_prepared_qwen3_paths") {
  using emel::text::generator::attention_mode;
  using emel::text::generator::detail::scalar_argmax_route;
  using emel::text::generator::detail::scalar_matmul_route;

  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(
        emel::text::generator::detail::run_prefill<attention_mode::nonflash,
                                                   scalar_matmul_route::kernel>(
            fixture->backend));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(
        emel::text::generator::detail::run_prefill<attention_mode::flash,
                                                   scalar_matmul_route::kernel>(
            fixture->backend));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(emel::text::generator::detail::run_prefill<
          attention_mode::nonflash, scalar_matmul_route::native_quantized>(
        fixture->backend));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(emel::text::generator::detail::run_prefill<
          attention_mode::flash, scalar_matmul_route::native_quantized>(
        fixture->backend));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(
        emel::text::generator::detail::run_decode<attention_mode::nonflash,
                                                  scalar_matmul_route::kernel>(
            fixture->backend, fixture->request));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(
        emel::text::generator::detail::run_decode<attention_mode::flash,
                                                  scalar_matmul_route::kernel>(
            fixture->backend, fixture->request));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(emel::text::generator::detail::run_decode<
          attention_mode::nonflash, scalar_matmul_route::native_quantized>(
        fixture->backend, fixture->request));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    CHECK(emel::text::generator::detail::run_decode<
          attention_mode::flash, scalar_matmul_route::native_quantized>(
        fixture->backend, fixture->request));
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_prefill_preselected_argmax<
          attention_mode::nonflash, scalar_matmul_route::kernel,
          scalar_argmax_route::kernel>(fixture->backend, selected_index,
                                       selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_prefill_preselected_argmax<
          attention_mode::flash, scalar_matmul_route::kernel,
          scalar_argmax_route::kernel>(fixture->backend, selected_index,
                                       selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_prefill_preselected_argmax<
          attention_mode::nonflash, scalar_matmul_route::native_quantized,
          scalar_argmax_route::kernel>(fixture->backend, selected_index,
                                       selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_prefill_preselected_argmax<
          attention_mode::flash, scalar_matmul_route::native_quantized,
          scalar_argmax_route::kernel>(fixture->backend, selected_index,
                                       selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_decode_preselected_argmax<
          attention_mode::nonflash, scalar_matmul_route::kernel,
          scalar_argmax_route::kernel>(fixture->backend, fixture->request,
                                       selected_index, selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_decode_preselected_argmax<
          attention_mode::flash, scalar_matmul_route::kernel,
          scalar_argmax_route::kernel>(fixture->backend, fixture->request,
                                       selected_index, selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_decode_preselected_argmax<
          attention_mode::nonflash, scalar_matmul_route::native_quantized,
          scalar_argmax_route::kernel>(fixture->backend, fixture->request,
                                       selected_index, selected_score));
    CHECK(selected_index >= 0);
  }
  {
    auto fixture = std::make_unique<prepared_qwen3_backend_fixture>();
    REQUIRE(fixture->ready);
    int32_t selected_index = -1;
    float selected_score = 0.0f;
    CHECK(emel::text::generator::detail::run_decode_preselected_argmax<
          attention_mode::flash, scalar_matmul_route::native_quantized,
          scalar_argmax_route::kernel>(fixture->backend, fixture->request,
                                       selected_index, selected_score));
    CHECK(selected_index >= 0);
  }
}

TEST_CASE("generator_detail_scalar_routes_run_packed_and_q8_success_paths") {
  using emel::text::generator::attention_mode;
  using emel::text::generator::detail::scalar_matmul_route;

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  {
    auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
    REQUIRE(fixture->ready);
    int32_t err = -1;
    REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                               &err));
    REQUIRE(emel::text::generator::detail::run_prefill<
            attention_mode::nonflash, scalar_matmul_route::packed_q8_0>(
        fixture->backend));
    CHECK(fixture->backend.kv_cache_tokens ==
          chunk4_prefill_runtime_fixture::k_prompt_tokens);
  }
  {
    auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
    REQUIRE(fixture->ready);
    int32_t err = -1;
    REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                               &err));
    REQUIRE(emel::text::generator::detail::run_prefill<
            attention_mode::flash, scalar_matmul_route::packed_q8_0>(
        fixture->backend));
    CHECK(fixture->backend.kv_cache_tokens ==
          chunk4_prefill_runtime_fixture::k_prompt_tokens);
  }
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_DOTPROD)
  {
    auto fixture = std::make_unique<hybrid_chunk4_q8_runtime_fixture>();
    REQUIRE(fixture->ready);
    int32_t err = -1;
    REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                               &err));
    REQUIRE(
        emel::text::generator::detail::run_prefill<attention_mode::nonflash,
                                                   scalar_matmul_route::q8_k>(
            fixture->backend));
    CHECK(fixture->backend.kv_cache_tokens ==
          hybrid_chunk4_q8_runtime_fixture::k_prompt_tokens);
  }
  {
    auto fixture = std::make_unique<hybrid_chunk4_q8_runtime_fixture>();
    REQUIRE(fixture->ready);
    int32_t err = -1;
    REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                               &err));
    REQUIRE(
        emel::text::generator::detail::run_prefill<attention_mode::flash,
                                                   scalar_matmul_route::q8_k>(
            fixture->backend));
    CHECK(fixture->backend.kv_cache_tokens ==
          hybrid_chunk4_q8_runtime_fixture::k_prompt_tokens);
  }
#endif
}

TEST_CASE("generator_detail_prepare_derives_kv_geometry_from_memory_contract") {
  auto model_fixture = std::make_unique<qwen3_runtime_fixture>();
  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  const auto runtime_policy =
      emel::text::generator::test::make_auto_runtime_policy(
          model_fixture->model);

  // Default geometry: n_ctx=8 pads up to one 16-token block.
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, model_fixture->model, matmul.actor, runtime_policy) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(backend->kv_block_tokens == emel::memory::view::DEFAULT_BLOCK_TOKENS);
  CHECK(backend->kv_positions_capacity ==
        emel::memory::view::DEFAULT_BLOCK_TOKENS);
  CHECK(backend->kv_positions_capacity >= backend->n_ctx);
  const size_t default_flash_extent = backend->flash_key_cache.size();

  // Divisible geometry: capacity equals n_ctx exactly (pre-cutover layout).
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, model_fixture->model, matmul.actor, runtime_policy,
              4) == emel::error::cast(emel::model::loader::error::none));
  CHECK(backend->kv_block_tokens == 4);
  CHECK(backend->kv_positions_capacity == backend->n_ctx);
  const size_t divisible_flash_extent = backend->flash_key_cache.size();

  // Non-positive block tokens cannot cover the context window and are
  // rejected through prepare's existing model_invalid validation path.
  REQUIRE(emel::text::generator::detail::prepare(*backend, model_fixture->model,
                                                 matmul.actor, runtime_policy,
                                                 0) ==
          emel::error::cast(emel::model::loader::error::model_invalid));

  // Non-divisible geometry: capacity rounds up to whole blocks.
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, model_fixture->model, matmul.actor, runtime_policy,
              3) == emel::error::cast(emel::model::loader::error::none));
  CHECK(backend->kv_block_tokens == 3);
  CHECK(backend->kv_positions_capacity == 9);

  // Cache extents follow the padded capacity, not raw n_ctx.
  CHECK(default_flash_extent > divisible_flash_extent);
}

TEST_CASE("generator_detail_kv_physical_map_binds_snapshot_block_order") {
  auto model_fixture = std::make_unique<qwen3_runtime_fixture>();
  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  const auto runtime_policy =
      emel::text::generator::test::make_auto_runtime_policy(
          model_fixture->model);
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, model_fixture->model, matmul.actor, runtime_policy,
              4) == emel::error::cast(emel::model::loader::error::none));
  REQUIRE(backend->kv_positions_capacity == 8);

  // prepare leaves the identity map: logical == physical.
  const auto identity_kv =
      emel::text::generator::detail::identity_kv_addressing();
  for (int32_t position = 0; position < 8; ++position) {
    CHECK(emel::text::generator::detail::physical_kv_position(
              identity_kv, position) == static_cast<size_t>(position));
  }
  const int32_t high_identity_position =
      emel::memory::view::MAX_BLOCKS_PER_SEQUENCE + 904;
  CHECK(emel::text::generator::detail::physical_kv_position(
            identity_kv, high_identity_position) ==
        static_cast<size_t>(high_identity_position));

  // A reversed two-block mapping relocates logical block 0 to physical block 1
  // and vice versa; the offset helpers must follow the snapshot, not the
  // logical position.
  emel::memory::view::snapshot snapshot{};
  snapshot.max_sequences = 1;
  snapshot.block_tokens = 4;
  snapshot.sequence_active[0] = 1;
  snapshot.sequence_length_values[0] = 8;
  snapshot.sequence_kv_block_count[0] = 2;
  snapshot.sequence_kv_blocks[0][0] = 1;
  snapshot.sequence_kv_blocks[0][1] = 0;
  auto kv =
      emel::text::generator::detail::kv_addressing_from_snapshot(snapshot, 0);

  for (int32_t position = 0; position < 4; ++position) {
    CHECK(emel::text::generator::detail::physical_kv_position(kv, position) ==
          static_cast<size_t>(position + 4));
    CHECK(emel::text::generator::detail::physical_kv_position(
              kv, position + 4) == static_cast<size_t>(position));
  }

  // Stores land at the mapped physical slot in both layouts.
  const auto &block = backend->blocks.front();
  const int32_t kv_dim =
      emel::text::generator::detail::effective_attention_kv_dim(*backend,
                                                                block);
  std::vector<float> k_row(static_cast<size_t>(kv_dim), 1.5f);
  std::vector<float> v_row(static_cast<size_t>(kv_dim), -2.0f);
  REQUIRE(emel::text::generator::detail::store_attention_kv_cache(
      *backend, kv, block, 0, 0, k_row, v_row));

  const size_t mapped_offset =
      emel::text::generator::detail::layer_cache_offset(*backend, kv, block, 0,
                                                        0);
  const size_t physical_row_offset =
      static_cast<size_t>(4) * static_cast<size_t>(kv_dim);
  CHECK(mapped_offset == physical_row_offset);
  CHECK(backend->key_cache[physical_row_offset] != 0u);
  const size_t flat_row_offset = 0u;
  CHECK(backend->key_cache[flat_row_offset] == 0u);

  // Over-length snapshots bind only up to the prepared capacity, and an
  // inactive sequence binds nothing (zero-iteration fill).
  snapshot.sequence_length_values[0] = 99;
  CHECK(emel::text::generator::detail::physical_kv_position(kv, 7) < 8u);
  snapshot.sequence_length_values[0] = 8;

  snapshot.sequence_active[0] = 0;
  CHECK(emel::text::generator::detail::physical_kv_position(identity_kv, 7) ==
        7u);
  snapshot.sequence_active[0] = 1;

  // The snapshot-resolved recurrent slot shifts shortconv state addressing;
  // slot 0 preserves the flat layout.
  const size_t flat_shortconv =
      emel::text::generator::detail::shortconv_state_layer_offset(
          *backend, identity_kv, 0);
  CHECK(flat_shortconv == 0u);
  snapshot.sequence_recurrent_slot[0] = 3;
  kv.recurrent_slot = snapshot.lookup_recurrent_slot(0);
  CHECK(kv.recurrent_slot == 3);
  CHECK(emel::text::generator::detail::shortconv_state_layer_offset(*backend,
                                                                    kv, 0) ==
        static_cast<size_t>(3) * static_cast<size_t>(backend->n_layer) *
            static_cast<size_t>(backend->shortconv_state_size) *
            static_cast<size_t>(backend->n_embd));
  snapshot.sequence_recurrent_slot[0] = 0;
  kv.recurrent_slot = snapshot.lookup_recurrent_slot(0);

  // Rebinding a contiguous snapshot restores the flat layout addressing.
  snapshot.sequence_kv_blocks[0][0] = 0;
  snapshot.sequence_kv_blocks[0][1] = 1;
  CHECK(emel::text::generator::detail::layer_cache_offset(
            *backend, kv, block, 0, 0) == flat_row_offset);
}

TEST_CASE("generator_detail_kv_physical_map_isolates_interleaved_sequences") {
  auto model_fixture = std::make_unique<qwen3_runtime_fixture>();
  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  const auto runtime_policy =
      emel::text::generator::test::make_auto_runtime_policy(
          model_fixture->model);
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, model_fixture->model, matmul.actor, runtime_policy,
              2) == emel::error::cast(emel::model::loader::error::none));
  REQUIRE(backend->kv_positions_capacity == 8);

  // Two sequences share one snapshot: their bound physical positions must be
  // disjoint, and a freed-then-reused mapping preserves logical order.
  emel::memory::hybrid::sm memory{};
  int32_t err = 0;
  REQUIRE(
      memory.process_event(emel::memory::event::reserve{.max_sequences = 3,
                                                        .max_blocks = 4,
                                                        .block_tokens = 2,
                                                        .error_out = &err}));
  REQUIRE(memory.process_event(
      emel::memory::event::allocate_sequence{.seq_id = 0, .error_out = &err}));
  REQUIRE(memory.process_event(
      emel::memory::event::allocate_sequence{.seq_id = 1, .error_out = &err}));
  REQUIRE(memory.process_event(emel::memory::event::allocate_slots{
      .seq_id = 0, .token_count = 4, .error_out = &err}));
  REQUIRE(memory.process_event(emel::memory::event::allocate_slots{
      .seq_id = 1, .token_count = 4, .error_out = &err}));

  emel::memory::view::snapshot view{};
  emel::error::type view_err =
      emel::error::cast(emel::memory::hybrid::error::none);
  REQUIRE(memory.try_view(view, view_err));

  std::array<int32_t, 4> seq0_physical = {};
  std::array<int32_t, 4> seq1_physical = {};
  auto kv = emel::text::generator::detail::kv_addressing_from_snapshot(view, 0);
  for (int32_t position = 0; position < 4; ++position) {
    seq0_physical[static_cast<size_t>(position)] = static_cast<int32_t>(
        emel::text::generator::detail::physical_kv_position(kv, position));
  }
  kv = emel::text::generator::detail::kv_addressing_from_snapshot(view, 1);
  for (int32_t position = 0; position < 4; ++position) {
    seq1_physical[static_cast<size_t>(position)] = static_cast<int32_t>(
        emel::text::generator::detail::physical_kv_position(kv, position));
  }
  for (const int32_t lhs : seq0_physical) {
    for (const int32_t rhs : seq1_physical) {
      CHECK(lhs != rhs);
    }
  }

  // Free seq 0 and re-allocate: the same physical slots return in logical
  // order without touching seq 1.
  REQUIRE(memory.process_event(
      emel::memory::event::free_sequence{.seq_id = 0, .error_out = &err}));
  REQUIRE(memory.process_event(
      emel::memory::event::allocate_sequence{.seq_id = 2, .error_out = &err}));
  REQUIRE(memory.process_event(emel::memory::event::allocate_slots{
      .seq_id = 2, .token_count = 4, .error_out = &err}));
  REQUIRE(memory.try_view(view, view_err));

  kv = emel::text::generator::detail::kv_addressing_from_snapshot(view, 2);
  std::array<int32_t, 4> seq2_physical = {};
  bool identity = true;
  for (int32_t position = 0; position < 4; ++position) {
    seq2_physical[static_cast<size_t>(position)] = static_cast<int32_t>(
        emel::text::generator::detail::physical_kv_position(kv, position));
    identity =
        identity && seq2_physical[static_cast<size_t>(position)] == position;
  }
  CHECK(identity);
  for (const int32_t lhs : seq2_physical) {
    for (const int32_t rhs : seq1_physical) {
      CHECK(lhs != rhs);
    }
  }
}

TEST_CASE("generator_detail_prepare_block_native_matrices_supports_shortconv_"
          "q6_routes") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::text::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(
      emel::text::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));

  emel::text::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.blocks.resize(1u);
  auto &block = backend.blocks.front();
  block.residual_route =
      emel::model::transformer::generation_residual_route::shortconv;
  block.shortconv_in_proj = q6_matrix;
  block.shortconv_out_proj = q6_matrix;
  block.feed_forward_gate = q6_matrix;
  block.feed_forward_down = q6_matrix;
  block.feed_forward_up = q6_matrix;

  REQUIRE(
      emel::text::generator::detail::prepare_block_native_matrices(backend));
  REQUIRE(emel::text::generator::detail::prepare_q8_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(static_cast<uint8_t>(block.shortconv_out_proj.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.shortconv_out_proj));
  CHECK(backend.q8_input_storage.size() == 1u);
#elif defined(__aarch64__) && defined(__ARM_NEON) &&                           \
    defined(__ARM_FEATURE_DOTPROD)
  CHECK(static_cast<uint8_t>(block.shortconv_out_proj.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.shortconv_out_proj));
  CHECK(backend.q8_input_storage.size() == 1u);
#else
  CHECK(block.shortconv_out_proj.tensor == &q6_tensor);
  CHECK_FALSE(emel::text::generator::guard::detail::q8_input_path_supported(
      backend, block.shortconv_out_proj));
  CHECK(backend.q8_input_storage.empty());
#endif
}

TEST_CASE("generator_detail_prepare_output_logits_packs_q8_0_outputs") {
  std::array<block_q8_0, 4> q8_rows = {};
  for (size_t row = 0; row < q8_rows.size(); ++row) {
    q8_rows[row].d = fp16_bits(0.03125f * static_cast<float>(row + 1u));
    for (size_t idx = 0; idx < q8_rows[row].qs.size(); ++idx) {
      q8_rows[row].qs[idx] = static_cast<int8_t>(
          static_cast<int32_t>(((row + 3u) * 19u + idx * 7u) % 127u) - 63);
    }
  }

  auto q8_tensor =
      make_tensor_record(q8_rows.data(), emel::kernel::detail::dtype_q8_0,
                         static_cast<int32_t>(QK8_0), 4);
  emel::text::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output_native.tensor = &q8_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK8_0);
  backend.output_native.rows = 4;
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  REQUIRE(emel::text::generator::detail::prepare_output_logits(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(backend.output.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q8_0_x4_bl8);
#elif defined(__aarch64__) && defined(__ARM_NEON) &&                           \
    defined(__ARM_FEATURE_DOTPROD)
  REQUIRE(backend.output.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q8_0_x4_bl4);
#else
  CHECK(backend.output.tensor == &q8_tensor);
#endif
}

TEST_CASE("generator_detail_tensor_binding_and_copy_helpers_accept_explicit_"
          "quantized_routes") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::text::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(
      emel::text::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));
  CHECK(q6_matrix.cols == static_cast<int32_t>(QK_K));
  CHECK(q6_matrix.rows == static_cast<int32_t>(Q6_K_X8_ROWS));

  auto invalid_q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, 8, 2);
  emel::text::generator::detail::tensor_matrix invalid_matrix = {};
  CHECK_FALSE(emel::text::generator::detail::bind_tensor_rows(invalid_q6_tensor,
                                                              invalid_matrix));

  block_q3_k q3_block = {};
  q3_block.d = 0x3c00u;
  std::fill(q3_block.scales.begin(), q3_block.scales.end(),
            static_cast<uint8_t>(0x00u));
  std::fill(q3_block.hmask.begin(), q3_block.hmask.end(),
            static_cast<uint8_t>(0x00u));
  std::fill(q3_block.qs.begin(), q3_block.qs.end(),
            static_cast<uint8_t>(0x00u));
  auto q3_tensor =
      make_tensor_record(&q3_block, emel::kernel::detail::dtype_q3_k,
                         static_cast<int32_t>(QK_K), 1);
  std::vector<float> q3_out(QK_K, 0.0f);
  REQUIRE(emel::text::generator::detail::copy_tensor_row(q3_tensor, 0, q3_out));
  CHECK(q3_out.front() == doctest::Approx(128.0f));
  CHECK_FALSE(
      emel::text::generator::detail::copy_tensor_row(q3_tensor, 1, q3_out));

  block_q8_0 q8_block = {};
  q8_block.d = emel::text::generator::detail::quant::fp32_to_fp16(0.25f);
  q8_block.qs.fill(4);
  auto q8_tensor =
      make_tensor_record(&q8_block, emel::kernel::detail::dtype_q8_0,
                         static_cast<int32_t>(QK8_0), 1);
  std::vector<float> q8_out(QK8_0, 0.0f);
  REQUIRE(emel::text::generator::detail::copy_tensor_row(q8_tensor, 0, q8_out));
  CHECK(q8_out.front() == doctest::Approx(1.0f));
  CHECK(q8_out.back() == doctest::Approx(1.0f));

  std::vector<float> q6_out = {};
  REQUIRE(emel::text::generator::detail::dequantize_tensor_vector(
              q6_tensor, q6_out) == false);
  auto q6_vector_tensor =
      make_tensor_record(q6_rows.data(), emel::kernel::detail::dtype_q6_k,
                         static_cast<int32_t>(QK_K), 1);
  REQUIRE(emel::text::generator::detail::dequantize_tensor_vector(
      q6_vector_tensor, q6_out));
  std::vector<float> q6_expected(QK_K, 0.0f);
  emel::text::generator::detail::quant::dequantize_row_q6_k(
      q6_rows.data(), q6_expected.data(), QK_K);
  CHECK(q6_out.size() == QK_K);
  CHECK(q6_out.front() == doctest::Approx(q6_expected.front()));
  CHECK(q6_out.back() == doctest::Approx(q6_expected.back()));
}

TEST_CASE(
    "generator_detail_logits_route_helpers_cover_explicit_failure_edges") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::text::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;
  backend.q8_input_storage.resize(1u);

  emel::model::data::tensor_record packed_tensor = q6_tensor;
  packed_tensor.type = emel::kernel::detail::dtype_q6_k_x8;
  backend.output_argmax.tensor = &packed_tensor;
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_DOTPROD)
  CHECK(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));
#else
  CHECK_FALSE(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));
#endif

  emel::model::data::tensor_record unsupported_tensor = q6_tensor;
  unsupported_tensor.type = emel::kernel::detail::dtype_f32;
  backend.output_argmax.tensor = &unsupported_tensor;
  CHECK_FALSE(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          backend));

  emel::text::generator::detail::native_backend null_backend{};
  CHECK_FALSE(
      emel::text::generator::detail::prepare_output_logits(null_backend));
  CHECK_FALSE(emel::text::generator::detail::prepare_prepared_output_logits(
      null_backend));
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_output_logits(
      null_backend));

  auto invalid_q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, 8, 2);
  emel::text::generator::detail::native_backend invalid_backend{};
  invalid_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  invalid_backend.output_native.tensor = &invalid_q6_tensor;
  invalid_backend.output_native.cols = 8;
  invalid_backend.output_native.rows = 2;
  invalid_backend.output = invalid_backend.output_native;
  invalid_backend.output_argmax = invalid_backend.output_native;
  CHECK_FALSE(emel::text::generator::detail::prepare_prepared_output_logits(
      invalid_backend));
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_output_logits(
      invalid_backend));

  emel::text::generator::detail::native_backend no_simd_backend{};
  no_simd_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  no_simd_backend.output_native.tensor = &q6_tensor;
  no_simd_backend.output_native.cols = static_cast<int32_t>(QK_K);
  no_simd_backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  no_simd_backend.output = no_simd_backend.output_native;
  no_simd_backend.output_argmax = no_simd_backend.output_native;
  REQUIRE(
      emel::text::generator::detail::prepare_output_logits(no_simd_backend));
  CHECK(no_simd_backend.output.tensor == &q6_tensor);
  CHECK(no_simd_backend.output_argmax.tensor == &q6_tensor);

  std::vector<uint8_t> oversized_packed_storage(
      emel::text::generator::detail::row_storage_bytes(
          packed_tensor,
          static_cast<int32_t>(
              (emel::text::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) *
              QK_K)) *
      static_cast<size_t>(Q6_K_X8_ROWS));
  auto oversized_packed_tensor = make_tensor_record(
      oversized_packed_storage.data(), emel::kernel::detail::dtype_q6_k_x8,
      static_cast<int32_t>(
          (emel::text::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) * QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::text::generator::detail::native_backend huge_backend{};
  huge_backend.output.tensor = &oversized_packed_tensor;
  huge_backend.output.cols = static_cast<int32_t>(
      (emel::text::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) * QK_K);
  huge_backend.output.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  huge_backend.output_argmax = huge_backend.output;
  CHECK_FALSE(
      emel::text::generator::detail::prepare_q8_input_workspace(huge_backend));
}

TEST_CASE("generator_detail_packed_q8_0_input_workspace_helpers_are_explicit") {
  std::vector<uint8_t> packed_storage(
      sizeof(emel::kernel::detail::quant::block_q8_0x4));
  auto packed_tensor = make_tensor_record(
      packed_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8,
      static_cast<int32_t>(QK8_0), 4);

  emel::text::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output.tensor = &packed_tensor;
  backend.output.cols = static_cast<int32_t>(QK8_0);
  backend.output.rows = 4;

  REQUIRE(emel::text::generator::detail::prepare_packed_q8_0_input_workspace(
      backend));
  CHECK(backend.packed_q8_0_input_storage.size() == 1u);

#if defined(__aarch64__) && defined(__ARM_NEON)
  CHECK(emel::text::generator::guard::detail::packed_q8_0_input_path_supported(
      backend, backend.output));
#else
  CHECK_FALSE(
      emel::text::generator::guard::detail::packed_q8_0_input_path_supported(
          backend, backend.output));
#endif

  std::vector<uint8_t> oversized_storage(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) *
      (emel::text::generator::detail::quant::MAX_Q8_0_BLOCKS + 1u));
  auto oversized_tensor = make_tensor_record(
      oversized_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8,
      static_cast<int32_t>(
          (emel::text::generator::detail::quant::MAX_Q8_0_BLOCKS + 1u) * QK8_0),
      4);
  backend.output.tensor = &oversized_tensor;
  backend.output.cols = static_cast<int32_t>(
      (emel::text::generator::detail::quant::MAX_Q8_0_BLOCKS + 1u) * QK8_0);
  backend.output.rows = 4;
  CHECK_FALSE(
      emel::text::generator::detail::prepare_packed_q8_0_input_workspace(
          backend));
}

TEST_CASE("generator_detail_prefill_chunk4_q8_gemm_support_is_explicit") {
  constexpr int32_t rows = 4;
  constexpr int32_t cols = static_cast<int32_t>(QK8_0);
  std::vector<uint8_t> packed_storage(
      sizeof(emel::kernel::detail::quant::block_q8_0x4));
  auto packed_tensor =
      make_tensor_record(packed_storage.data(),
                         emel::kernel::detail::dtype_q8_0_x4_bl8, cols, rows);

  emel::text::generator::detail::tensor_matrix packed_matrix = {};
  packed_matrix.tensor = &packed_tensor;
  packed_matrix.rows = rows;
  packed_matrix.cols = cols;

  emel::text::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.n_layer = 1;
  backend.n_embd = 4;
  backend.n_head = 1;
  backend.head_dim = 4;
  backend.n_head_kv = 1;
  backend.head_dim_kv = 4;
  backend.output = packed_matrix;
  backend.blocks.resize(1u);
  auto &block = backend.blocks.front();
  block.attention_q = packed_matrix;
  block.attention_k = packed_matrix;
  block.attention_v = packed_matrix;
  block.attention_output = packed_matrix;
  block.feed_forward_gate = packed_matrix;
  block.feed_forward_down = packed_matrix;
  block.feed_forward_up = packed_matrix;

  REQUIRE(
      emel::text::generator::detail::prepare_packed_q8_0_chunk4_input_workspace(
          backend));
  backend.hidden_chunk4.resize(16u);
  backend.norm_chunk4.resize(16u);
  backend.projected_chunk4.resize(16u);
  backend.attn_ctx_chunk4.resize(16u);
  backend.q_chunk4.resize(16u);
  backend.k_chunk4.resize(16u);
  backend.v_chunk4.resize(16u);
  backend.gate_chunk4.resize(16u);
  backend.up_chunk4.resize(16u);
  backend.ffn_hidden_chunk4.resize(16u);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
      backend));
  backend.k_chunk4.clear();
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          backend));
  backend.k_chunk4.resize(16u);
  backend.v_chunk4.clear();
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          backend));
  backend.v_chunk4.resize(16u);
  backend.up_chunk4.clear();
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          backend));
#else
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          backend));
#endif
}

TEST_CASE("generator_detail_prefill_chunk4_q8_gemm_support_accepts_hybrid_q8_k_"
          "x4_paths") {
  auto fixture = std::make_unique<hybrid_chunk4_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_DOTPROD)
  CHECK(emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
      fixture->backend));
  fixture->backend.q8_input_chunk4_storage.clear();
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          fixture->backend));
  fixture->backend.q8_input_chunk4_storage.resize(
      fixture->backend.q8_input_storage.size() * static_cast<size_t>(4u));
  fixture->backend.shortconv_bcx_chunk4.clear();
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          fixture->backend));
#else
  CHECK_FALSE(
      emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
          fixture->backend));
#endif
}

TEST_CASE("generator_detail_chunk4_packed_q8_0_helpers_are_explicit_and_"
          "numeric_match") {
  using emel::kernel::detail::quant::block_q8_0;
  using emel::text::generator::detail::quant::Q8_0_X4_ROWS;

  constexpr int32_t row_count = 8;
  constexpr int32_t rhs_rows = static_cast<int32_t>(Q8_0_X4_ROWS);
  constexpr int32_t col_count = static_cast<int32_t>(QK8_0 * 8u);
  constexpr size_t block_count =
      static_cast<size_t>(col_count / static_cast<int32_t>(QK8_0));

  std::vector<block_q8_0> native_rows(static_cast<size_t>(row_count) *
                                      block_count);
  for (int32_t row = 0; row < row_count; ++row) {
    for (size_t block = 0; block < block_count; ++block) {
      auto &cell = native_rows[static_cast<size_t>(row) * block_count + block];
      cell.d = emel::kernel::detail::quant::fp32_to_fp16(
          0.0078125f *
          static_cast<float>(
              ((row + 3) * static_cast<int32_t>(block + 5)) % 23 + 1));
      for (size_t idx = 0; idx < cell.qs.size(); ++idx) {
        const int32_t centered =
            static_cast<int32_t>(
                ((row + 7) * 17 + (block + 11) * 13 + idx * 5) % 255) -
            127;
        cell.qs[idx] = static_cast<int8_t>(std::clamp(centered, -127, 127));
      }
    }
  }

  std::vector<float> rhs_dense(
      static_cast<size_t>(rhs_rows) * static_cast<size_t>(col_count), 0.0f);
  for (int32_t row = 0; row < rhs_rows; ++row) {
    for (int32_t col = 0; col < col_count; ++col) {
      const int32_t centered = ((row + 5) * 19 + col * 7) % 63 - 31;
      rhs_dense[static_cast<size_t>(row) * static_cast<size_t>(col_count) +
                static_cast<size_t>(col)] =
          static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_0> rhs_q8(static_cast<size_t>(rhs_rows) * block_count);
  for (int32_t row = 0; row < rhs_rows; ++row) {
    emel::kernel::detail::quant::quantize_row_q8_0_strided(
        rhs_dense.data() +
            static_cast<size_t>(row) * static_cast<size_t>(col_count),
        1u, rhs_q8.data() + static_cast<size_t>(row) * block_count,
        static_cast<int64_t>(col_count));
  }

  std::vector<float> reference(
      static_cast<size_t>(rhs_rows) * static_cast<size_t>(row_count), 0.0f);
  for (int32_t rhs_row = 0; rhs_row < rhs_rows; ++rhs_row) {
    for (int32_t lhs_row = 0; lhs_row < row_count; ++lhs_row) {
      reference[static_cast<size_t>(rhs_row) * static_cast<size_t>(row_count) +
                static_cast<size_t>(lhs_row)] =
          emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
              native_rows.data() + static_cast<size_t>(lhs_row) * block_count,
              rhs_q8.data() + static_cast<size_t>(rhs_row) * block_count,
              block_count);
    }
  }

  std::vector<uint8_t> packed_storage(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) *
      emel::kernel::detail::quant::packed_q8_0_x4_group_count(
          static_cast<uint64_t>(row_count)) *
      block_count);
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      native_rows.data(), static_cast<uint64_t>(row_count),
      static_cast<uint64_t>(col_count), packed_storage.data()));

  auto packed_tensor = make_tensor_record(
      packed_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8, col_count,
      row_count);
  emel::text::generator::detail::tensor_matrix packed_matrix = {};
  packed_matrix.tensor = &packed_tensor;
  packed_matrix.rows = row_count;
  packed_matrix.cols = col_count;

  emel::text::generator::detail::native_backend backend{};
  matmul_actor_fixture matmul = {};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  bind_test_matmul_actor(backend, matmul);
  backend.output = packed_matrix;
  REQUIRE(
      emel::text::generator::detail::prepare_packed_q8_0_chunk4_input_workspace(
          backend));

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(emel::text::generator::detail::prepare_packed_q8_0_chunk4_input(
      backend, rhs_dense, col_count));
  std::vector<float> output(
      static_cast<size_t>(rhs_rows) * static_cast<size_t>(row_count), 0.0f);
  REQUIRE(
      emel::text::generator::detail::matmul_chunk4_prepared_packed_q8_0_input(
          backend, packed_matrix, col_count, output));
  for (size_t idx = 0; idx < output.size(); ++idx) {
    CHECK(output[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-6f));
  }
  CHECK(backend.kernel_dispatch_calls == 1u);
  CHECK(backend.packed_q8_0_dispatch_calls == 1u);
#else
  REQUIRE(emel::text::generator::detail::prepare_packed_q8_0_chunk4_input(
      backend, rhs_dense, col_count));
  std::vector<float> output(
      static_cast<size_t>(rhs_rows) * static_cast<size_t>(row_count), 0.0f);
  CHECK_FALSE(
      emel::text::generator::detail::matmul_chunk4_prepared_packed_q8_0_input(
          backend, packed_matrix, col_count, output));
#endif
}

TEST_CASE("generator_detail_tensor_helpers_reject_invalid_records_explicitly") {
  emel::text::generator::detail::tensor_matrix out = {};
  emel::model::data::tensor_record empty_tensor = {};
  CHECK_FALSE(
      emel::text::generator::detail::bind_tensor_rows(empty_tensor, out));

  std::array<float, 4> f32_data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto f32_tensor = make_tensor_record(f32_data.data(),
                                       emel::kernel::detail::dtype_f32, 2, 2);
  std::vector<float> copy_out(2u, 0.0f);
  CHECK_FALSE(
      emel::text::generator::detail::copy_tensor_row(f32_tensor, -1, copy_out));

  emel::model::data::tensor_record bad_dims = f32_tensor;
  bad_dims.dims[1] = 0u;
  CHECK_FALSE(emel::text::generator::detail::bind_tensor_rows(bad_dims, out));

  emel::model::data::tensor_record unsupported = f32_tensor;
  unsupported.type = 255;
  CHECK_FALSE(
      emel::text::generator::detail::bind_tensor_rows(unsupported, out));
  CHECK_FALSE(
      emel::text::generator::detail::copy_tensor_row(unsupported, 0, copy_out));

  std::vector<float> wrong_size(3u, 0.0f);
  CHECK_FALSE(emel::text::generator::detail::copy_tensor_row(f32_tensor, 0,
                                                             wrong_size));
  CHECK_FALSE(
      emel::text::generator::detail::copy_tensor_row(f32_tensor, 3, copy_out));
}

TEST_CASE("generator_detail_numeric_helpers_reject_invalid_shapes_explicitly") {
  emel::text::generator::detail::native_backend backend{};
  emel::text::generator::detail::tensor_matrix empty_matrix = {};
  std::array<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 2> output = {};
  int32_t selected_index = -1;
  float selected_score = 0.0f;
  std::array<emel::kernel::detail::quant::block_q8_k, 1> q8_input = {};
  std::array<emel::kernel::detail::quant::block_q8_0, 1> q8_0_input = {};

  CHECK_FALSE(emel::text::generator::detail::matmul_vector(
      backend, empty_matrix, std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::text::generator::detail::matmul_vector_argmax(
      backend, empty_matrix, std::span<const float>(input.data(), input.size()),
      selected_index, selected_score));
  CHECK_FALSE(emel::text::generator::detail::quantize_vector_q8_k(
      std::span<const float>(input.data(), 3u),
      std::span(q8_input.data(), q8_input.size())));
  CHECK_FALSE(emel::text::generator::detail::quantize_vector_q8_0(
      std::span<const float>(input.data(), 3u),
      std::span(q8_0_input.data(), q8_0_input.size())));
  CHECK_FALSE(emel::text::generator::detail::matmul_vector_q8_input(
      backend, empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_k>(q8_input.data(),
                                                               q8_input.size()),
      static_cast<int32_t>(QK_K),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::text::generator::detail::matmul_vector_q8_0_input(
      backend, empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_0>(
          q8_0_input.data(), q8_0_input.size()),
      static_cast<int32_t>(QK8_0),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::text::generator::detail::matmul_vector_q8_input_argmax(
      backend, empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_k>(q8_input.data(),
                                                               q8_input.size()),
      static_cast<int32_t>(QK_K), selected_index, selected_score));

  std::array<float, 2> weight = {1.0f, 1.0f};
  CHECK_FALSE(emel::text::generator::detail::rms_norm(
      std::span<const float>(input.data(), 0u),
      std::span<const float>(weight.data(), weight.size()), 1.0e-5f,
      std::span<float>(output.data(), output.size())));

  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k,
      static_cast<int32_t>(QK_K), static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::text::generator::detail::native_backend packed_backend{};
  packed_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  packed_backend.output_native.tensor = &q6_tensor;
  packed_backend.output_native.cols = static_cast<int32_t>(QK_K);
  packed_backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  packed_backend.output = packed_backend.output_native;
  packed_backend.output_argmax = packed_backend.output_native;
  REQUIRE(emel::text::generator::detail::prepare_packed_output_logits(
      packed_backend));
  std::vector<float> packed_copy(QK_K, 0.0f);
  CHECK_FALSE(emel::text::generator::detail::copy_tensor_row(
      *packed_backend.output.tensor, 0, packed_copy));

  std::array<float, 4> rope_identity = {1.0f, 2.0f, 3.0f, 4.0f};
  const auto rope_before = rope_identity;
  emel::text::generator::detail::apply_rope(rope_identity, 1, 1, 1, 3,
                                            10000.0f);
  CHECK(rope_identity == rope_before);
}

TEST_CASE("generator_detail_request_and_backend_validators_reject_invalid_"
          "inputs_explicitly") {
  int32_t err = 123;
  CHECK_FALSE(emel::text::generator::detail::check_backend(nullptr, &err));
  CHECK(err == emel::text::generator::detail::k_error_invalid);

  emel::text::generator::detail::native_backend backend{};
  backend.model = reinterpret_cast<const emel::model::data *>(0x1);
  backend.n_embd = 4;
  backend.n_head = 1;
  backend.n_head_kv = 1;
  backend.n_layer = 1;
  backend.n_vocab = 8;
  backend.n_ctx = 4;
  backend.kv_block_tokens = 4;
  backend.kv_positions_capacity = 4;
  backend.head_dim = 4;
  backend.head_dim_kv = 4;
  CHECK_FALSE(emel::text::generator::detail::check_backend(&backend, &err));
  CHECK(err == emel::text::generator::detail::k_error_invalid);

  emel::graph::processor::event::execute request = {};
  CHECK(emel::text::generator::detail::request_plan(request, &err) == nullptr);
  CHECK(err == emel::text::generator::detail::k_error_invalid);

  backend.bound_tokens.resize(2u);
  backend.bound_positions.resize(2u);
  CHECK_FALSE(emel::text::generator::detail::store_bound_request(
      backend, request, &err));
  CHECK(err == emel::text::generator::detail::k_error_invalid);

  std::array<float, 4> probs = {1.0f, 1.0f, 1.0f, 1.0f};
  std::array<float, 4> rounded = {1.0f, 1.0f, 1.0f, 1.0f};
  emel::text::generator::detail::fill_masked_softmax_probs_ggml(
      std::span<const float>(), 0, probs, rounded);
  CHECK(std::all_of(probs.begin(), probs.end(),
                    [](const float value) { return value == 0.0f; }));
  CHECK(std::all_of(rounded.begin(), rounded.end(),
                    [](const float value) { return value == 0.0f; }));
}

TEST_CASE("generator_detail_effective_dimension_helpers_cover_fallbacks") {
  emel::text::generator::detail::native_backend backend{};
  backend.n_head = 2;
  backend.n_head_kv = 1;
  backend.head_dim = 8;
  backend.head_dim_kv = 4;
  backend.n_rot = 6;
  backend.rope_freq_base = 10000.0f;
  backend.shortconv_state_size = 3;
  backend.n_embd = 4;
  backend.recurrent_shortconv_cache = {1.0f, 2.0f, 3.0f, 4.0f};

  emel::text::generator::detail::block_weights block{};
  block.attention_rope_freq_base = 5000.0f;
  CHECK(emel::text::generator::detail::effective_attention_head_dim(
            backend, block) == 8);
  CHECK(emel::text::generator::detail::effective_attention_head_dim_kv(
            backend, block) == 4);
  CHECK(emel::text::generator::detail::effective_attention_rope_dim(
            backend, block) == 6);
  CHECK(emel::text::generator::detail::effective_attention_rope_freq_base(
            backend, block) == doctest::Approx(10000.0f));
  CHECK(emel::text::generator::detail::effective_max_q_dim(backend) == 16);
  CHECK(emel::text::generator::detail::effective_max_kv_dim(backend) == 4);
  CHECK(emel::text::generator::detail::effective_max_ffn_dim(backend) == 0);
  CHECK(emel::text::generator::detail::first_attention_block(backend) ==
        nullptr);
  CHECK(emel::text::generator::detail::shortconv_state_layer_offset(backend,
                                                                    2) == 24u);
  emel::text::generator::detail::reset_shortconv_cache(backend);
  CHECK(std::all_of(backend.recurrent_shortconv_cache.begin(),
                    backend.recurrent_shortconv_cache.end(),
                    [](const float value) { return value == 0.0f; }));

  backend.blocks.resize(2u);
  backend.blocks[0].residual_route =
      emel::model::transformer::generation_residual_route::shortconv;
  backend.blocks[1].residual_route =
      emel::model::transformer::generation_residual_route::attention;
  backend.blocks[1].feed_forward_gate.rows = 32;
  CHECK(emel::text::generator::detail::first_attention_block(backend) ==
        &backend.blocks[1]);
  CHECK(emel::text::generator::detail::effective_max_ffn_dim(backend) == 0);
  backend.max_ffn_dim = 64;
  CHECK(emel::text::generator::detail::effective_max_ffn_dim(backend) == 64);

  emel::model::transformer::execution_view execution{};
  CHECK(emel::text::generator::detail::select_output_projection_tensor(
            execution) == nullptr);
  emel::model::data::tensor_record token_embedding{};
  float token_data = 0.0f;
  token_embedding.data = &token_data;
  token_embedding.n_dims = 2;
  execution.token_embedding.tensor = &token_embedding;
  CHECK(emel::text::generator::detail::select_output_projection_tensor(
            execution) == &token_embedding);
  emel::model::data::tensor_record output{};
  output.data = &token_data;
  execution.output.tensor = &output;
  CHECK(emel::text::generator::detail::select_output_projection_tensor(
            execution) == &output);

  std::array<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 2> weights = {1.0f, 1.0f};
  CHECK_FALSE(emel::text::generator::detail::apply_headwise_rms_norm(
      std::span<float>(values.data(), values.size()),
      std::span<const float>(weights.data(), 1u), 2, 2, 1.0e-5f));
}

TEST_CASE(
    "generator_detail_packed_matrix_layout_helpers_validate_matrix_inputs") {
  emel::text::generator::detail::tensor_matrix matrix{};
  emel::text::generator::detail::packed_matrix_binding packed{};

  CHECK_FALSE(emel::text::generator::detail::prepare_packed_q8_0_matrix_layout<
              emel::kernel::detail::dtype_q8_0_x4_bl8>(matrix, packed));
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_q4_matrix_layout<
              emel::kernel::detail::dtype_q4_k_x8_bl8>(matrix, packed));
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_q6_matrix_layout<
              emel::kernel::detail::dtype_q6_k_x8>(matrix, packed));

  emel::model::data::tensor_record tensor{};
  tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
  matrix.tensor = &tensor;
  CHECK(emel::text::generator::detail::prepare_packed_q8_0_matrix_layout<
        emel::kernel::detail::dtype_q8_0_x4_bl8>(matrix, packed));
  CHECK(emel::text::generator::detail::prepare_packed_q4_matrix_layout<
        emel::kernel::detail::dtype_q4_k_x8_bl8>(matrix, packed));
  CHECK(emel::text::generator::detail::prepare_packed_q6_matrix_layout<
        emel::kernel::detail::dtype_q6_k_x8>(matrix, packed));

  std::vector<uint8_t> storage;
  emel::model::data::tensor_record packed_tensor{};
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_q8_0_tensor_layout<
              emel::kernel::detail::dtype_q8_0_x4_bl8>(tensor, 4, 32,
                                                       packed_tensor, storage));
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_q4_tensor_layout<
              emel::kernel::detail::dtype_q4_k_x8_bl8>(tensor, 4, 256,
                                                       packed_tensor, storage));
  CHECK_FALSE(emel::text::generator::detail::prepare_packed_q6_tensor_layout<
              emel::kernel::detail::dtype_q6_k_x8>(tensor, 8, 256,
                                                   packed_tensor, storage));
}

TEST_CASE("generator_detail_graph_callbacks_accept_guarded_requests_without_"
          "error_channel") {
  auto fixture = std::make_unique<runtime_request_fixture>();
  int32_t err = -1;
  bool reused = true;

  CHECK(emel::text::generator::detail::validate_guarded_compute(
      fixture->request, &err));
  CHECK(err == -1);

  fixture->request.expected_outputs = 2;
  CHECK(emel::text::generator::detail::validate_guarded_compute(
      fixture->request, &err));
  CHECK(err == -1);
  fixture->request.expected_outputs = fixture->plan.expected_outputs;

  CHECK(emel::text::generator::detail::validate_guarded_preselected_argmax(
      fixture->request, &err));
  CHECK(err == -1);

  fixture->io.selected_score_out = nullptr;
  CHECK(emel::text::generator::detail::validate_guarded_preselected_argmax(
      fixture->request, &err));
  CHECK(err == -1);
  fixture->io.selected_score_out = &fixture->selected_score;

  emel::graph::processor::event::execute_ctx execute_ctx{};
  emel::graph::processor::action::context processor_context{};
  const auto execute_is_valid = [&]() {
    const emel::graph::processor::event::execute_step execute_step{
        fixture->request,
        execute_ctx,
    };
    return emel::graph::processor::guard::valid_execute{}(execute_step,
                                                          processor_context);
  };
  CHECK(execute_is_valid());
  const auto expect_kv_contract_rejected = [&]() {
    CHECK_FALSE(execute_is_valid());
  };

  const auto *saved_memory_view = fixture->request.memory_view;
  fixture->request.memory_view = nullptr;
  err = -1;
  CHECK_FALSE(emel::text::generator::detail::validate_guarded_compute(
      fixture->request, &err));
  CHECK(err == static_cast<int32_t>(emel::error::cast(
                   emel::graph::processor::error::invalid_request)));
  expect_kv_contract_rejected();
  fixture->request.memory_view = saved_memory_view;

  const int32_t *saved_seq_primary_ids = fixture->request.seq_primary_ids;
  fixture->request.seq_primary_ids = nullptr;
  expect_kv_contract_rejected();
  fixture->request.seq_primary_ids = saved_seq_primary_ids;

  fixture->request.seq_primary_ids_count = 0;
  expect_kv_contract_rejected();
  fixture->request.seq_primary_ids_count =
      static_cast<int32_t>(fixture->seq_primary_ids.size());

  std::array<int32_t, 2> multi_seq_ids = {0, 0};
  fixture->request.seq_primary_ids = multi_seq_ids.data();
  fixture->request.seq_primary_ids_count =
      static_cast<int32_t>(multi_seq_ids.size());
  err = -1;
  CHECK_FALSE(emel::text::generator::detail::validate_guarded_compute(
      fixture->request, &err));
  CHECK(err == static_cast<int32_t>(emel::error::cast(
                   emel::graph::processor::error::invalid_request)));
  fixture->request.seq_primary_ids = saved_seq_primary_ids;
  fixture->request.seq_primary_ids_count =
      static_cast<int32_t>(fixture->seq_primary_ids.size());

  fixture->backend.shortconv_state_size = 1;
  fixture->memory_snapshot.sequence_recurrent_slot[0] = 1;
  err = -1;
  CHECK_FALSE(emel::text::generator::detail::validate_guarded_compute(
      fixture->request, &err));
  CHECK(err == static_cast<int32_t>(emel::error::cast(
                   emel::graph::processor::error::invalid_request)));
  fixture->memory_snapshot.sequence_recurrent_slot[0] = 0;
  fixture->backend.shortconv_state_size = 0;

  fixture->memory_snapshot.sequence_active[0] = 0;
  expect_kv_contract_rejected();
  fixture->memory_snapshot.sequence_active[0] = 1;

  err = -1;
  CHECK(emel::text::generator::detail::validate_guarded_compute(
      fixture->request, &err));
  CHECK(err == -1);

  CHECK(emel::text::generator::detail::prepare_graph(fixture->request, &reused,
                                                     &err));
  CHECK_FALSE(reused);
  CHECK(err == emel::text::generator::detail::k_error_ok);

  CHECK(emel::text::generator::detail::alloc_graph(fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
}

TEST_CASE("generator_detail_graph_callbacks_reject_incoherent_kv_snapshots") {
  namespace gen_detail = emel::text::generator::detail;

  const int32_t invalid_request = static_cast<int32_t>(
      emel::error::cast(emel::graph::processor::error::invalid_request));

  {
    auto fixture = std::make_unique<runtime_request_fixture>();
    fixture->memory_snapshot.block_tokens = 4;

    int32_t err = -1;
    CHECK_FALSE(gen_detail::validate_guarded_compute(fixture->request, &err));
    CHECK(err == invalid_request);
  }

  {
    auto fixture = std::make_unique<runtime_request_fixture>();
    std::array<int32_t, 1> decode_positions = {3};
    fixture->backend.kv_block_tokens = 2;
    fixture->backend.kv_positions_capacity = 8;
    fixture->memory_snapshot.block_tokens = 2;
    fixture->memory_snapshot.sequence_length_values[0] = 4;
    fixture->memory_snapshot.sequence_kv_block_count[0] = 2;
    fixture->memory_snapshot.sequence_kv_blocks[0][0] =
        emel::memory::view::INVALID_KV_BLOCK;
    fixture->memory_snapshot.sequence_kv_blocks[0][1] = 0;
    fixture->request.positions = decode_positions.data();

    int32_t err = -1;
    CHECK_FALSE(gen_detail::validate_guarded_compute(fixture->request, &err));
    CHECK(err == invalid_request);
  }

  {
    auto fixture = std::make_unique<runtime_request_fixture>();
    std::array<int32_t, 1> decode_positions = {8};
    fixture->backend.kv_positions_capacity = 32;
    fixture->memory_snapshot.sequence_length_values[0] = 9;
    fixture->memory_snapshot.sequence_kv_block_count[0] = 2;
    fixture->memory_snapshot.sequence_kv_blocks[0][0] = 0;
    fixture->memory_snapshot.sequence_kv_blocks[0][1] = 2;
    fixture->request.positions = decode_positions.data();

    fixture->io.selected_attention_mode =
        emel::text::generator::attention_mode::nonflash;
    int32_t err = -1;
    CHECK(gen_detail::validate_guarded_compute(fixture->request, &err));
    CHECK(err == -1);

    fixture->io.selected_attention_mode =
        emel::text::generator::attention_mode::flash;
    err = -1;
    CHECK_FALSE(gen_detail::validate_guarded_compute(fixture->request, &err));
    CHECK(err == invalid_request);
  }
}

TEST_CASE(
    "generator_detail_runtime_callbacks_bind_run_and_extract_guarded_data") {
  auto fixture = std::make_unique<runtime_request_fixture>();
  int32_t err = -1;
  int32_t outputs = 0;

  CHECK(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                           &err));
  CHECK(err == -1);
  CHECK(fixture->backend.bound_ready);
  CHECK(fixture->backend.bound_token_count == 1);
  CHECK(fixture->backend.bound_position_count == 1);

  CHECK(emel::text::generator::detail::extract_guarded_outputs(fixture->request,
                                                               &outputs, &err));
  CHECK(outputs == 1);
  CHECK(err == -1);
  CHECK(fixture->logits[0] == doctest::Approx(0.25f));
  CHECK(fixture->logits[1] == doctest::Approx(0.5f));
  CHECK(fixture->logits[2] == doctest::Approx(0.75f));
  CHECK(fixture->logits[3] == doctest::Approx(1.0f));
  CHECK(fixture->logits[4] == doctest::Approx(-1.0f));
  CHECK(fixture->logits[5] == doctest::Approx(-1.0f));

  CHECK(emel::text::generator::detail::extract_guarded_preselected_argmax(
      fixture->request, &outputs, &err));
  CHECK(outputs == 1);
  CHECK(err == -1);
}

TEST_CASE("generator_detail_builds_flash_request_over_head_major_kv_cache") {
  emel::text::generator::detail::native_backend backend{};
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.n_layer = 1;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 2;
  backend.kv_block_tokens = 2;
  backend.kv_positions_capacity = 2;
  backend.blocks.resize(1u);
  backend.blocks.front().attention_q_dim = 4;
  backend.blocks.front().attention_kv_dim = 4;
  backend.blocks.front().attention_head_dim = 2;
  backend.blocks.front().attention_head_dim_kv = 2;
  backend.layer_cache_offsets = {0u};
  backend.flash_layer_cache_offsets = {0u};
  backend.q = {1.0f, 0.1f, 0.2f, 1.0f};
  backend.q_attn = {9.0f, 9.0f, 9.0f, 9.0f};
  backend.flash_key_cache = {
      fp16_bits(1.0f), fp16_bits(0.0f), fp16_bits(0.0f),  fp16_bits(1.0f),
      fp16_bits(0.5f), fp16_bits(0.5f), fp16_bits(0.25f), fp16_bits(0.75f),
  };
  backend.flash_value_cache = {
      fp16_bits(2.0f), fp16_bits(0.0f), fp16_bits(0.0f), fp16_bits(4.0f),
      fp16_bits(1.0f), fp16_bits(1.0f), fp16_bits(0.5f), fp16_bits(1.5f),
  };
  backend.attn_ctx.resize(4);

  const auto request = emel::text::generator::detail::make_flash_attn_request(
      backend, backend.blocks.front(), 0, 1);
  float scale = 0.0f;
  uint32_t total_tokens = 0u;
  std::memcpy(&scale, request.op_params.data(), sizeof(scale));
  std::memcpy(&total_tokens, request.op_params.data() + sizeof(scale),
              sizeof(total_tokens));

  CHECK(request.src0.ne[0] == 2u);
  CHECK(request.src0.ne[2] == 2u);
  CHECK(request.src0.data == backend.q.data());
  CHECK(request.src1.ne[0] == 2u);
  CHECK(request.src1.ne[1] == 2u);
  CHECK(request.src1.ne[2] == 2u);
  CHECK(request.src1.nb[1] == sizeof(uint16_t) * 2u);
  CHECK(request.src1.nb[2] == sizeof(uint16_t) * 4u);
  CHECK(request.dst.ne[0] == 2u);
  CHECK(request.dst.ne[2] == 2u);
  CHECK(request.op_params_size == sizeof(float) + sizeof(uint32_t));
  CHECK(scale == doctest::Approx(1.0f / std::sqrt(2.0f)));
  CHECK(total_tokens == 2u);
  CHECK(request.src1.type == emel::kernel::event::dtype::f16);
  CHECK(request.src2.type == emel::kernel::event::dtype::f16);
  CHECK(emel::kernel::detail::can_run_flash_attn_ext(request));
}

TEST_CASE("generator_detail_flash_dispatch_matches_online_softmax_reference_on_"
          "same_backend_state") {
  emel::text::generator::detail::native_backend backend{};
  backend.n_head = 12;
  backend.n_head_kv = 12;
  backend.n_layer = 1;
  backend.n_rep = 1;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 256;
  backend.kv_block_tokens = 256;
  backend.kv_positions_capacity = 256;
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.blocks.resize(1u);
  backend.blocks.front().attention_q_dim = 768;
  backend.blocks.front().attention_kv_dim = 768;
  backend.blocks.front().attention_head_dim = 64;
  backend.blocks.front().attention_head_dim_kv = 64;
  backend.layer_cache_offsets = {0u};
  backend.flash_layer_cache_offsets = {0u};

  const size_t n_embd = static_cast<size_t>(backend.n_head) *
                        static_cast<size_t>(backend.head_dim);
  const size_t kv_dim = static_cast<size_t>(backend.n_head_kv) *
                        static_cast<size_t>(backend.head_dim_kv);
  const int32_t position = 255;
  const int32_t position_limit = position + 1;

  backend.q.resize(n_embd);
  backend.q_attn.resize(n_embd);
  backend.flash_key_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.flash_value_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs_rounded.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_value_column.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(n_embd);

  for (size_t idx = 0; idx < backend.q.size(); ++idx) {
    const float raw =
        static_cast<float>(std::sin(static_cast<double>(idx + 1u) * 0.03125));
    backend.q[idx] = raw;
  }

  for (int32_t token = 0; token < position_limit; ++token) {
    for (int32_t head = 0; head < backend.n_head_kv; ++head) {
      for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
        const size_t offset =
            emel::text::generator::detail::
                flash_layer_cache_head_position_offset(
                    backend, backend.blocks.front(), 0, head, token) +
            static_cast<size_t>(dim);
        const double base =
            static_cast<double>((token + 1) * (head + 3) * (dim + 5));
        backend.flash_key_cache[offset] =
            emel::text::generator::detail::quant::fp32_to_fp16(
                static_cast<float>(std::cos(base * 0.0078125)));
        backend.flash_value_cache[offset] =
            emel::text::generator::detail::quant::fp32_to_fp16(
                static_cast<float>(std::sin(base * 0.01171875)));
      }
    }
  }

  std::vector<float> flash_ctx(backend.attn_ctx.size(), 0.0f);
  std::vector<float> expected_ctx(backend.attn_ctx.size(), 0.0f);
  backend.attn_ctx = flash_ctx;
  REQUIRE(emel::text::generator::detail::dispatch_flash_attention(
      backend, backend.blocks.front(), 0, position));
  flash_ctx = backend.attn_ctx;
  expected_ctx =
      flash_attention_online_reference(backend, 0, position, backend.q);

  for (size_t idx = 0; idx < flash_ctx.size(); ++idx) {
    CHECK(within_flash_online_f16_tolerance(flash_ctx[idx], expected_ctx[idx]));
  }
}

TEST_CASE("generator_detail_flash_dispatch_matches_online_softmax_reference_"
          "across_long_reuse") {
  emel::text::generator::detail::native_backend backend{};
  backend.n_head = 12;
  backend.n_head_kv = 12;
  backend.n_layer = 1;
  backend.n_rep = 1;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 1024;
  backend.kv_block_tokens = 1024;
  backend.kv_positions_capacity = 1024;
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.blocks.resize(1u);
  backend.blocks.front().attention_q_dim = 768;
  backend.blocks.front().attention_kv_dim = 768;
  backend.blocks.front().attention_head_dim = 64;
  backend.blocks.front().attention_head_dim_kv = 64;
  backend.layer_cache_offsets = {0u};
  backend.flash_layer_cache_offsets = {0u};

  const size_t n_embd = static_cast<size_t>(backend.n_head) *
                        static_cast<size_t>(backend.head_dim);
  const size_t kv_dim = static_cast<size_t>(backend.n_head_kv) *
                        static_cast<size_t>(backend.head_dim_kv);

  backend.q.resize(n_embd);
  backend.q_attn.resize(n_embd);
  backend.flash_key_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.flash_value_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs_rounded.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_value_column.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(n_embd);

  int32_t first_position = -1;
  size_t first_index = 0u;
  float first_flash = 0.0f;
  float first_expected = 0.0f;
  float max_abs = 0.0f;

  for (int32_t position = 0; position < backend.n_ctx; ++position) {
    for (size_t idx = 0; idx < backend.q.size(); ++idx) {
      const double q_base = static_cast<double>((position + 1) * (idx + 1u));
      const float raw = static_cast<float>(std::sin(q_base * 0.0009765625));
      backend.q[idx] = raw;
    }

    for (int32_t head = 0; head < backend.n_head_kv; ++head) {
      for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
        const size_t offset =
            emel::text::generator::detail::
                flash_layer_cache_head_position_offset(
                    backend, backend.blocks.front(), 0, head, position) +
            static_cast<size_t>(dim);
        const double kv_base =
            static_cast<double>((position + 1) * (head + 3) * (dim + 5));
        backend.flash_key_cache[offset] =
            emel::text::generator::detail::quant::fp32_to_fp16(
                static_cast<float>(std::cos(kv_base * 0.0078125)));
        backend.flash_value_cache[offset] =
            emel::text::generator::detail::quant::fp32_to_fp16(
                static_cast<float>(std::sin(kv_base * 0.01171875)));
      }
    }

    std::vector<float> flash_ctx(backend.attn_ctx.size(), 0.0f);
    std::vector<float> expected_ctx(backend.attn_ctx.size(), 0.0f);
    backend.attn_ctx = flash_ctx;
    REQUIRE(emel::text::generator::detail::dispatch_flash_attention(
        backend, backend.blocks.front(), 0, position));
    flash_ctx = backend.attn_ctx;
    expected_ctx =
        flash_attention_online_reference(backend, 0, position, backend.q);

    for (size_t idx = 0; idx < flash_ctx.size(); ++idx) {
      const float diff = std::fabs(flash_ctx[idx] - expected_ctx[idx]);
      if (diff > max_abs) {
        max_abs = diff;
      }
      if (first_position < 0 && !within_flash_online_f16_tolerance(
                                    flash_ctx[idx], expected_ctx[idx])) {
        first_position = position;
        first_index = idx;
        first_flash = flash_ctx[idx];
        first_expected = expected_ctx[idx];
      }
    }
  }

  INFO("first_position=" << first_position << " first_index=" << first_index
                         << " first_flash=" << first_flash << " first_expected="
                         << first_expected << " max_abs=" << max_abs);
  CHECK(max_abs <= k_flash_online_f16_abs_tolerance);
}

TEST_CASE(
    "generator_detail_qwen3_generator_applies_per_head_qk_norm_before_rope") {
  auto fixture = std::make_unique<qwen3_runtime_fixture>();
  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  const auto runtime_policy =
      emel::text::generator::test::make_auto_runtime_policy(fixture->model);
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, fixture->model, matmul.actor, runtime_policy) ==
          emel::error::cast(emel::model::loader::error::none));
  REQUIRE(emel::text::generator::detail::copy_tensor_row(
      *backend->token_embedding.tensor, 0, backend->hidden));

  std::array<float, 4> hidden_after_input_norm = {};
  REQUIRE(emel::text::generator::detail::rms_norm(
      backend->hidden, backend->blocks[0].attention_norm, backend->rms_epsilon,
      std::span<float>(hidden_after_input_norm.data(),
                       hidden_after_input_norm.size())));

  std::array<float, 4> expected_q = hidden_after_input_norm;
  std::array<float, 4> expected_k = hidden_after_input_norm;
  constexpr std::array<float, 2> q_norm = {2.0f, 0.5f};
  constexpr std::array<float, 2> k_norm = {1.5f, 0.25f};
  apply_qwen3_headwise_rms_norm(expected_q, q_norm, 2, 2, backend->rms_epsilon);
  apply_qwen3_headwise_rms_norm(expected_k, k_norm, 2, 2, backend->rms_epsilon);
  apply_rope_reference(expected_q, 2, 2, 2, 1, backend->rope_freq_base);
  apply_rope_reference(expected_k, 2, 2, 2, 1, backend->rope_freq_base);

  REQUIRE(emel::text::generator::layer::run_layer_nonflash(*backend, 0, 1));
  const size_t cache_offset = emel::text::generator::detail::layer_cache_offset(
      *backend, backend->blocks[0], 0, 1);

  for (size_t idx = 0; idx < expected_q.size(); ++idx) {
    CHECK(backend->q[idx] == doctest::Approx(expected_q[idx]).epsilon(1.0e-5));
    CHECK(backend->q_attn[idx] ==
          doctest::Approx(round_fp16_value(expected_q[idx])).epsilon(1.0e-5));
    CHECK(backend->k[idx] == doctest::Approx(expected_k[idx]).epsilon(1.0e-5));
    CHECK(emel::text::generator::detail::quant::fp16_to_fp32(
              backend->key_cache[cache_offset + idx]) ==
          doctest::Approx(round_fp16_value(expected_k[idx])).epsilon(1.0e-5));
  }
}

TEST_CASE(
    "generator_detail_gemma4_generator_applies_per_head_qk_norm_before_rope") {
  auto fixture = std::make_unique<gemma4_runtime_fixture>();
  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  const auto runtime_policy =
      emel::text::generator::test::make_auto_runtime_policy(fixture->model);
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, fixture->model, matmul.actor, runtime_policy) ==
          emel::error::cast(emel::model::loader::error::none));
  REQUIRE(emel::text::generator::detail::copy_tensor_row(
      *backend->token_embedding.tensor, 0, backend->hidden));

  std::array<float, 4> hidden_after_input_norm = {};
  REQUIRE(emel::text::generator::detail::rms_norm(
      backend->hidden, backend->blocks[0].attention_norm, backend->rms_epsilon,
      std::span<float>(hidden_after_input_norm.data(),
                       hidden_after_input_norm.size())));

  std::array<float, 4> expected_q = hidden_after_input_norm;
  std::array<float, 2> expected_k = {hidden_after_input_norm[0],
                                     hidden_after_input_norm[1]};
  constexpr std::array<float, 2> q_norm = {2.0f, 0.5f};
  constexpr std::array<float, 2> k_norm = {1.5f, 0.25f};
  apply_qwen3_headwise_rms_norm(expected_q, q_norm, 2, 2, backend->rms_epsilon);
  apply_qwen3_headwise_rms_norm(expected_k, k_norm, 1, 2, backend->rms_epsilon);
  apply_rope_reference(expected_q, 2, 2, 2, 1, backend->rope_freq_base);
  apply_rope_reference(expected_k, 1, 2, 2, 1, backend->rope_freq_base);

  REQUIRE(emel::text::generator::layer::run_layer_nonflash(*backend, 0, 1));
  const size_t cache_offset = emel::text::generator::detail::layer_cache_offset(
      *backend, backend->blocks[0], 0, 1);

  for (size_t idx = 0; idx < expected_q.size(); ++idx) {
    CHECK(backend->q[idx] == doctest::Approx(expected_q[idx]).epsilon(1.0e-5));
    CHECK(backend->q_attn[idx] ==
          doctest::Approx(round_fp16_value(expected_q[idx])).epsilon(1.0e-5));
  }
  for (size_t idx = 0; idx < expected_k.size(); ++idx) {
    CHECK(backend->k[idx] == doctest::Approx(expected_k[idx]).epsilon(1.0e-5));
    CHECK(emel::text::generator::detail::quant::fp16_to_fp32(
              backend->key_cache[cache_offset + idx]) ==
          doctest::Approx(round_fp16_value(expected_k[idx])).epsilon(1.0e-5));
  }
}

TEST_CASE("generator_detail_gemma4_shared_kv_layer_rms_norms_value_branch_"
          "before_cache") {
  auto fixture = std::make_unique<gemma4_runtime_fixture>();
  auto backend =
      std::make_unique<emel::text::generator::detail::native_backend>();
  matmul_actor_fixture matmul = {};
  const auto runtime_policy =
      emel::text::generator::test::make_auto_runtime_policy(fixture->model);
  REQUIRE(emel::text::generator::detail::prepare(
              *backend, fixture->model, matmul.actor, runtime_policy) ==
          emel::error::cast(emel::model::loader::error::none));

  backend->n_layer = 16;
  backend->blocks.resize(16u, backend->blocks.front());
  backend->blocks[15].attention_v = backend->blocks[15].attention_k;
  backend->blocks[15].value_route = emel::model::transformer::
      generation_attention_value_route::shared_key_value;
  backend->blocks[15].v_norm_route =
      emel::model::transformer::generation_attention_v_norm_route::rms;
  backend->layer_cache_offsets.resize(16u, 0u);
  backend->flash_layer_cache_offsets.resize(16u, 0u);

  REQUIRE(emel::text::generator::detail::copy_tensor_row(
      *backend->token_embedding.tensor, 0, backend->hidden));

  std::array<float, 4> hidden_after_input_norm = {};
  REQUIRE(emel::text::generator::detail::rms_norm(
      backend->hidden, backend->blocks[15].attention_norm, backend->rms_epsilon,
      std::span<float>(hidden_after_input_norm.data(),
                       hidden_after_input_norm.size())));

  std::array<float, 2> expected_v = {hidden_after_input_norm[0],
                                     hidden_after_input_norm[1]};
  const float v_square_sum =
      expected_v[0] * expected_v[0] + expected_v[1] * expected_v[1];
  const float v_scale =
      1.0f / std::sqrt(v_square_sum / 2.0f + backend->rms_epsilon);
  expected_v[0] *= v_scale;
  expected_v[1] *= v_scale;

  REQUIRE(emel::text::generator::layer::run_layer_nonflash(*backend, 15, 0));
  const size_t cache_offset = emel::text::generator::detail::layer_cache_offset(
      *backend, backend->blocks[15], 15, 0);

  for (size_t idx = 0; idx < expected_v.size(); ++idx) {
    CHECK(emel::text::generator::detail::quant::fp16_to_fp32(
              backend->value_cache[cache_offset + idx]) ==
          doctest::Approx(round_fp16_value(expected_v[idx])).epsilon(1.0e-5));
  }
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk4_batches_"
          "explicit_q8_gemm") {
  auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                             &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(emel::text::generator::detail::
              run_kernel_nonflash_prefill_chunk4_packed_q8_0(fixture->request,
                                                             &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens ==
        chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 8u);
  CHECK(fixture->backend.kernel_dispatch_calls == 8u);
  CHECK(std::all_of(fixture->backend.bound_logits.begin(),
                    fixture->backend.bound_logits.end(),
                    [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(emel::text::generator::detail::
                  run_kernel_nonflash_prefill_chunk4_packed_q8_0(
                      fixture->request, &err));
#endif
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk4_preselected_"
          "argmax_batches_explicit_q8_gemm") {
  auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                             &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(emel::text::generator::detail::
              run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0(
                  fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens ==
        chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 7u);
  CHECK(fixture->backend.kernel_dispatch_calls == 8u);
  CHECK(fixture->selected_token == 0);
  CHECK(fixture->selected_score == doctest::Approx(0.0f));
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::text::generator::detail::
          run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0(
              fixture->request, &err));
#endif
}

TEST_CASE("generator_detail_run_kernel_flash_prefill_chunk4_batches_explicit_"
          "q8_gemm") {
  auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                             &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(
      emel::text::generator::detail::
          run_kernel_flash_prefill_chunk4_packed_q8_0(fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens ==
        chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 8u);
  CHECK(fixture->backend.flash_attention_dispatch_calls ==
        chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.kernel_dispatch_calls == 12u);
  CHECK(std::all_of(fixture->backend.bound_logits.begin(),
                    fixture->backend.bound_logits.end(),
                    [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::text::generator::detail::
          run_kernel_flash_prefill_chunk4_packed_q8_0(fixture->request, &err));
#endif
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk4_batches_hybrid_"
          "q8_k_x4_gemm") {
  auto fixture = std::make_unique<hybrid_chunk4_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_DOTPROD)
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                             &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(
      emel::text::generator::detail::run_kernel_nonflash_prefill_chunk4_q8_k(
          fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens ==
        hybrid_chunk4_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 0u);
  CHECK(fixture->backend.kernel_dispatch_calls == 13u);
  CHECK(std::all_of(fixture->backend.bound_logits.begin(),
                    fixture->backend.bound_logits.end(),
                    [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::text::generator::detail::run_kernel_nonflash_prefill_chunk4_q8_k(
          fixture->request, &err));
#endif
}

TEST_CASE("generator_detail_nonflash_chunk4_prefill_matches_scalar_q8_k_on_"
          "nonzero_hybrid_fixture") {
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_DOTPROD)
  using fixture_type = hybrid_chunk4_q8_runtime_fixture;
  auto scalar_fixture = std::make_unique<fixture_type>();
  REQUIRE(scalar_fixture->ready);
  REQUIRE(seed_nonzero_hybrid_fixture_weights(*scalar_fixture));
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(
      scalar_fixture->request, &err));
  REQUIRE(emel::text::generator::detail::run_prefill<
          emel::text::generator::attention_mode::nonflash,
          emel::text::generator::detail::scalar_matmul_route::q8_k>(
      scalar_fixture->backend));

  auto chunk_fixture = std::make_unique<fixture_type>();
  REQUIRE(chunk_fixture->ready);
  REQUIRE(seed_nonzero_hybrid_fixture_weights(*chunk_fixture));
  err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(
      chunk_fixture->request, &err));
  REQUIRE(emel::text::generator::detail::run_prefill_chunk4<
          emel::text::generator::attention_mode::nonflash,
          emel::text::generator::detail::chunk4_rhs_route::q8_k>(
      chunk_fixture->backend));

  REQUIRE(chunk_fixture->backend.bound_logits.size() ==
          scalar_fixture->backend.bound_logits.size());
  float max_delta = 0.0f;
  for (size_t idx = 0; idx < scalar_fixture->backend.bound_logits.size();
       ++idx) {
    max_delta = std::max(max_delta,
                         std::fabs(chunk_fixture->backend.bound_logits[idx] -
                                   scalar_fixture->backend.bound_logits[idx]));
    CHECK(chunk_fixture->backend.bound_logits[idx] ==
          doctest::Approx(scalar_fixture->backend.bound_logits[idx])
              .epsilon(1.0e-5f));
  }
  CHECK(max_delta <= 1.0e-4f);
#endif
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk8_batches_hybrid_"
          "q8_k_x8_gemm") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                             &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(
      emel::text::generator::detail::run_kernel_nonflash_prefill_chunk8_q8_k(
          fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens ==
        hybrid_chunk8_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 0u);
  CHECK(fixture->backend.kernel_dispatch_calls == 13u);
  CHECK(std::all_of(fixture->backend.bound_logits.begin(),
                    fixture->backend.bound_logits.end(),
                    [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::text::generator::detail::run_kernel_nonflash_prefill_chunk8_q8_k(
          fixture->request, &err));
#endif
}

TEST_CASE("generator_detail_nonflash_chunk8_prefill_matches_scalar_q8_k_on_"
          "nonzero_hybrid_fixture") {
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  using fixture_type = hybrid_chunk8_q8_runtime_fixture;
  auto scalar_fixture = std::make_unique<fixture_type>();
  REQUIRE(scalar_fixture->ready);
  REQUIRE(seed_nonzero_hybrid_fixture_weights(*scalar_fixture));
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(
      scalar_fixture->request, &err));
  REQUIRE(emel::text::generator::detail::run_prefill<
          emel::text::generator::attention_mode::nonflash,
          emel::text::generator::detail::scalar_matmul_route::q8_k>(
      scalar_fixture->backend));

  auto chunk_fixture = std::make_unique<fixture_type>();
  REQUIRE(chunk_fixture->ready);
  REQUIRE(seed_nonzero_hybrid_fixture_weights(*chunk_fixture));
  err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(
      chunk_fixture->request, &err));
  REQUIRE(emel::text::generator::detail::run_prefill_chunk8_q8_k<
          emel::text::generator::attention_mode::nonflash>(
      chunk_fixture->backend));

  REQUIRE(chunk_fixture->backend.bound_logits.size() ==
          scalar_fixture->backend.bound_logits.size());
  float max_delta = 0.0f;
  for (size_t idx = 0; idx < scalar_fixture->backend.bound_logits.size();
       ++idx) {
    max_delta = std::max(max_delta,
                         std::fabs(chunk_fixture->backend.bound_logits[idx] -
                                   scalar_fixture->backend.bound_logits[idx]));
    CHECK(chunk_fixture->backend.bound_logits[idx] ==
          doctest::Approx(scalar_fixture->backend.bound_logits[idx])
              .epsilon(1.0e-5f));
  }
  CHECK(max_delta <= 1.0e-4f);
#endif
}

TEST_CASE("generator_detail_run_kernel_flash_prefill_chunk8_batches_hybrid_q8_"
          "k_x8_gemm") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(fixture->request,
                                                             &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(emel::text::generator::detail::run_kernel_flash_prefill_chunk8_q8_k(
      fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens ==
        hybrid_chunk8_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 0u);
  CHECK(fixture->backend.flash_attention_dispatch_calls ==
        hybrid_chunk8_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.kernel_dispatch_calls == 21u);
  CHECK(std::all_of(fixture->backend.bound_logits.begin(),
                    fixture->backend.bound_logits.end(),
                    [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::text::generator::detail::run_kernel_flash_prefill_chunk8_q8_k(
          fixture->request, &err));
#endif
}

TEST_CASE("generator_detail_run_kernel_flash_prefill_parallel_chunk8_keeps_"
          "matmuls_on_lane_kernels_and_matches_serial") {
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  // 12 prompt tokens: one full chunk8 batch plus a 4-token scalar remainder,
  // so the assertion covers the chunk gemm matmuls (q/k/v, attention output,
  // shortconv projections, ffn gate/up/down), the scalar remainder layers,
  // and the logits matmul.
  using parallel_fixture_type = hybrid_chunked_q8_runtime_fixture<12, 16>;
  auto serial_fixture = std::make_unique<parallel_fixture_type>();
  REQUIRE(serial_fixture->ready);
  int32_t err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(
      serial_fixture->request, &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(emel::text::generator::detail::run_kernel_flash_prefill_chunk8_q8_k(
      serial_fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);

  auto parallel_fixture = std::make_unique<parallel_fixture_type>();
  REQUIRE(parallel_fixture->ready);
  err = -1;
  REQUIRE(emel::text::generator::detail::bind_guarded_inputs(
      parallel_fixture->request, &err));
  err = emel::text::generator::detail::k_error_ok;
  REQUIRE(emel::text::generator::detail::
              run_kernel_flash_prefill_parallel_chunk8_q8_k(
                  parallel_fixture->request, &err));
  CHECK(err == emel::text::generator::detail::k_error_ok);

  // The parallel route must keep every prefill matmul on the matmul lane
  // actors; the matmul actor's serial kernel stays untouched.
  CHECK(parallel_fixture->backend.kernel.optimized_q4_dispatch_count() == 0u);
  CHECK(parallel_fixture->matmul.actor.serial_kernel()
            .optimized_q4_dispatch_count() == 0u);
  uint64_t lane_q4_dispatches = 0u;
  for (const auto &lane_kernel :
       parallel_fixture->matmul.actor.parallel_lane_kernels()) {
    lane_q4_dispatches += lane_kernel.optimized_q4_dispatch_count();
  }
  CHECK(lane_q4_dispatches > 0u);

  // Row-sliced lanes write disjoint dst rows and reorder no reductions, so
  // the parallel output is bit-identical to the serial dispatch.
  REQUIRE(parallel_fixture->backend.bound_logits.size() ==
          serial_fixture->backend.bound_logits.size());
  for (size_t idx = 0; idx < serial_fixture->backend.bound_logits.size();
       ++idx) {
    CHECK(parallel_fixture->backend.bound_logits[idx] ==
          serial_fixture->backend.bound_logits[idx]);
  }
#endif
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk8_preselected_"
          "argmax_route_guard_rejects_without_direct_argmax_support") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

  CHECK_FALSE(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          fixture->backend));
}

TEST_CASE("generator_detail_run_kernel_flash_prefill_chunk8_preselected_argmax_"
          "route_guard_rejects_without_direct_argmax_support") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

  CHECK_FALSE(
      emel::text::generator::guard::detail::preselected_argmax_direct_supported(
          fixture->backend));
}
