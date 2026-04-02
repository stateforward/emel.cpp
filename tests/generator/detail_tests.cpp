#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/generator/detail.hpp"
#include "emel/generator/guards.hpp"
#include "../kernel/test_helpers.hpp"

namespace {

using emel::generator::detail::quant::QK_K;
using emel::generator::detail::quant::Q4_K_X8_ROWS;
using emel::generator::detail::quant::Q6_K_X8_ROWS;
using emel::generator::detail::quant::QK8_0;
using emel::generator::detail::quant::block_q2_k;
using emel::generator::detail::quant::block_q3_k;
using emel::generator::detail::quant::block_q4_k;
using emel::generator::detail::quant::block_q6_k;
using emel::generator::detail::quant::block_q8_0;
using emel::kernel::test::flash_attn_reference_online_softmax_f16_values;
using emel::kernel::test::k_flash_online_f16_abs_tolerance;
using emel::kernel::test::within_flash_online_f16_tolerance;

uint16_t fp16_bits(const float value) {
  return emel::generator::detail::quant::fp32_to_fp16(value);
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
      rows[row].ql[idx] = static_cast<uint8_t>(((row + 1u) * 17u + idx * 7u) & 0xffu);
    }
    for (size_t idx = 0; idx < rows[row].qh.size(); ++idx) {
      rows[row].qh[idx] = static_cast<uint8_t>(((row + 3u) * 11u + idx * 5u) & 0xffu);
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
      rows[row].scales[idx] = static_cast<uint8_t>(((row + 5u) * 23u + idx * 11u) & 0xffu);
    }
    for (size_t idx = 0; idx < rows[row].qs.size(); ++idx) {
      rows[row].qs[idx] = static_cast<uint8_t>(((row + 1u) * 29u + idx * 3u) & 0xffu);
    }
  }
  return rows;
}

emel::model::data::tensor_record make_tensor_record(void * data,
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
      emel::generator::detail::row_storage_bytes(tensor, cols) * static_cast<uint64_t>(rows);
  return tensor;
}

struct runtime_request_fixture {
  emel::model::data model = {};
  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan plan = {};
  emel::generator::detail::native_backend backend = {};
  std::array<int32_t, 1> token_ids = {0};
  std::array<int32_t, 1> positions = {0};
  std::array<float, 6> logits = {};
  int32_t selected_token = -1;
  float selected_score = 0.0f;
  emel::generator::compute_io io = {};
  emel::graph::processor::event::execute request = {};

  explicit runtime_request_fixture(
      const emel::model::llama::detail::step_kind kind =
          emel::model::llama::detail::step_kind::prefill) {
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

    request.step_plan = &plan;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = static_cast<int32_t>(positions.size());
    request.kv_tokens = 0;
  }
};

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
    model.n_layers = 1;
    model.weights_data = model.tensors.data();
    model.weights_size = 1u;
    std::memcpy(model.architecture_name.data(), "qwen3", 5u);

    uint32_t tensor_index = 0u;
    const auto add_name = [&](emel::model::data::tensor_record & tensor,
                              const std::string_view name) {
      tensor.name_offset = model.name_bytes_used;
      tensor.name_length = static_cast<uint32_t>(name.size());
      std::memcpy(model.name_storage.data() + model.name_bytes_used, name.data(), name.size());
      model.name_bytes_used += static_cast<uint32_t>(name.size());
    };
    const auto add_vector = [&](const std::string_view name, const std::vector<float> & values) {
      auto & tensor = model.tensors[tensor_index++];
      add_name(tensor, name);
      tensor_storage.push_back(values);
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
      tensor.n_dims = 1;
      tensor.dims[0] = static_cast<uint64_t>(values.size());
      tensor.data = tensor_storage.back().data();
      tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
    };
    const auto add_matrix = [&](const std::string_view name,
                                const int32_t rows,
                                const int32_t cols,
                                const std::vector<float> & values) {
      auto & tensor = model.tensors[tensor_index++];
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
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    add_matrix("token_embd.weight", 2, 4, {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
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

struct chunk4_prefill_runtime_fixture {
  static constexpr int32_t k_vocab = static_cast<int32_t>(QK8_0);
  static constexpr int32_t k_embd = static_cast<int32_t>(QK8_0);
  static constexpr int32_t k_ctx = 8;
  static constexpr int32_t k_prompt_tokens = 4;

  emel::model::data model = {};
  emel::generator::detail::native_backend backend = {};
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

  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan plan = {};
  std::array<int32_t, k_prompt_tokens> token_ids = {0, 1, 2, 3};
  std::array<int32_t, k_prompt_tokens> positions = {0, 1, 2, 3};
  std::vector<float> logits = {};
  int32_t selected_token = -1;
  float selected_score = -1.0f;
  emel::generator::compute_io io = {};
  emel::graph::processor::event::execute request = {};
  bool ready = false;

  chunk4_prefill_runtime_fixture() {
    token_embedding_storage.resize(static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
    for (int32_t token = 0; token < k_prompt_tokens; ++token) {
      token_embedding_storage[static_cast<size_t>(token) * static_cast<size_t>(k_embd) +
                              static_cast<size_t>(token)] = 1.0f;
    }
    output_argmax_storage.resize(static_cast<size_t>(k_vocab * k_embd), 0.0f);
    output_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    ffn_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);

    const size_t block_count = static_cast<size_t>(k_embd / static_cast<int32_t>(QK8_0));
    zero_rows.resize(static_cast<size_t>(k_embd) * block_count);
    packed_storage.resize(
        sizeof(emel::kernel::detail::quant::block_q8_0x4) *
        emel::kernel::detail::quant::packed_q8_0_x4_group_count(static_cast<uint64_t>(k_embd)) *
        block_count);

    token_embedding_tensor = make_tensor_record(
        token_embedding_storage.data(), emel::kernel::detail::dtype_f32, k_embd, k_prompt_tokens);
    packed_tensor = make_tensor_record(
        packed_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8, k_embd, k_embd);
    output_argmax_tensor = make_tensor_record(
        output_argmax_storage.data(), emel::kernel::detail::dtype_f32, k_embd, k_vocab);

    backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    backend.model = &model;
    backend.n_vocab = k_vocab;
    backend.n_embd = k_embd;
    backend.n_head = 1;
    backend.n_head_kv = 1;
    backend.n_layer = 1;
    backend.n_ctx = k_ctx;
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
    auto & block = backend.blocks.front();
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
    backend.hidden_chunk4.resize(static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
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
    plan.kind = emel::model::llama::detail::step_kind::prefill;
    plan.expected_outputs = 1;
    io.backend_ctx = &backend;
    io.token_ids = token_ids.data();
    io.token_count = k_prompt_tokens;
    io.logits = logits.data();
    io.logits_capacity = k_vocab;
    io.selected_token_out = &selected_token;
    io.selected_score_out = &selected_score;
    request.step_plan = &plan;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = k_prompt_tokens;
    request.kv_tokens = 0;

    ready =
        emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
            zero_rows.data(),
            static_cast<uint64_t>(k_embd),
            static_cast<uint64_t>(k_embd),
            packed_storage.data()) &&
        emel::generator::detail::prepare_packed_q8_0_input_workspace(backend) &&
        emel::generator::detail::prepare_packed_q8_0_chunk4_input_workspace(backend);
  }
};

template <int32_t prompt_tokens>
struct hybrid_chunked_q8_runtime_fixture {
  static constexpr int32_t k_vocab = static_cast<int32_t>(QK_K);
  static constexpr int32_t k_embd = static_cast<int32_t>(QK_K);
  static constexpr int32_t k_ctx = 8;
  static constexpr int32_t k_prompt_tokens = prompt_tokens;
  static constexpr int32_t k_shortconv_kernel_size = 3;

  emel::model::data model = {};
  emel::generator::detail::native_backend backend = {};
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

  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan plan = {};
  std::array<int32_t, k_prompt_tokens> token_ids = {};
  std::array<int32_t, k_prompt_tokens> positions = {};
  std::vector<float> logits = {};
  emel::generator::compute_io io = {};
  emel::graph::processor::event::execute request = {};
  bool ready = false;

  hybrid_chunked_q8_runtime_fixture() {
    std::memcpy(model.architecture_name.data(), "lfm2", 4u);
    for (int32_t token = 0; token < k_prompt_tokens; ++token) {
      token_ids[static_cast<size_t>(token)] = token;
      positions[static_cast<size_t>(token)] = token;
    }
    token_embedding_storage.resize(static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
    for (int32_t token = 0; token < k_prompt_tokens; ++token) {
      token_embedding_storage[static_cast<size_t>(token) * static_cast<size_t>(k_embd) +
                              static_cast<size_t>(token)] = 1.0f;
    }
    output_argmax_storage.resize(static_cast<size_t>(k_vocab * k_embd), 0.0f);
    output_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    ffn_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_q_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    attention_k_norm_storage.assign(static_cast<size_t>(k_embd), 1.0f);
    shortconv_conv_storage.assign(
        static_cast<size_t>(k_shortconv_kernel_size) * static_cast<size_t>(k_embd), 0.0f);

    const size_t block_count = static_cast<size_t>(k_embd / static_cast<int32_t>(QK_K));
    square_rows.resize(static_cast<size_t>(k_embd) * block_count);
    shortconv_in_rows.resize(static_cast<size_t>(3 * k_embd) * block_count);
    packed_square_storage.resize(
        sizeof(emel::kernel::detail::quant::block_q4_kx8) *
        emel::kernel::detail::quant::packed_q4_k_x8_group_count(static_cast<uint64_t>(k_embd)) *
        block_count);
    packed_shortconv_in_storage.resize(
        sizeof(emel::kernel::detail::quant::block_q4_kx8) *
        emel::kernel::detail::quant::packed_q4_k_x8_group_count(static_cast<uint64_t>(3 * k_embd)) *
        block_count);

    token_embedding_tensor = make_tensor_record(
        token_embedding_storage.data(), emel::kernel::detail::dtype_f32, k_embd, k_prompt_tokens);
    packed_square_tensor = make_tensor_record(
        packed_square_storage.data(), emel::kernel::detail::dtype_q4_k_x8_bl8, k_embd, k_embd);
    packed_shortconv_in_tensor = make_tensor_record(
        packed_shortconv_in_storage.data(),
        emel::kernel::detail::dtype_q4_k_x8_bl8,
        k_embd,
        3 * k_embd);
    output_argmax_tensor = make_tensor_record(
        output_argmax_storage.data(), emel::kernel::detail::dtype_f32, k_embd, k_vocab);

    backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    backend.model = &model;
    backend.n_vocab = k_vocab;
    backend.n_embd = k_embd;
    backend.n_head = 1;
    backend.n_head_kv = 1;
    backend.n_layer = 2;
    backend.n_ctx = k_ctx;
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
    auto & shortconv_block = backend.blocks[0];
    shortconv_block.uses_attention = false;
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

    auto & attention_block = backend.blocks[1];
    attention_block.attention_norm = attention_norm_storage;
    attention_block.attention_q.tensor = &packed_square_tensor;
    attention_block.attention_q.rows = k_embd;
    attention_block.attention_q.cols = k_embd;
    attention_block.attention_k = attention_block.attention_q;
    attention_block.attention_v = attention_block.attention_q;
    attention_block.attention_output = attention_block.attention_q;
    attention_block.attention_q_norm = attention_q_norm_storage;
    attention_block.attention_k_norm = attention_k_norm_storage;
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
    backend.shortconv_bcx_chunk4.resize(static_cast<size_t>(k_prompt_tokens * 3 * k_embd), 0.0f);
    backend.shortconv_conv_out_chunk4.resize(static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
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
    backend.key_cache.resize(static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.value_cache.resize(static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.flash_key_cache.resize(static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.flash_value_cache.resize(static_cast<size_t>(backend.n_layer * k_ctx * k_embd), 0u);
    backend.recurrent_shortconv_cache.resize(
        static_cast<size_t>(backend.n_layer * backend.shortconv_state_size * k_embd), 0.0f);
    backend.hidden_chunk4.resize(static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
    backend.hidden_chunk8.resize(static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);
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
        static_cast<size_t>(k_prompt_tokens * 3 * k_embd), 0.0f);
    backend.shortconv_conv_out_chunk8.resize(
        static_cast<size_t>(k_prompt_tokens * k_embd), 0.0f);

    logits.resize(static_cast<size_t>(k_vocab), -1.0f);
    topology.execution = &execution;
    plan.graph = &topology;
    plan.kind = emel::model::llama::detail::step_kind::prefill;
    plan.expected_outputs = 1;
    io.backend_ctx = &backend;
    io.token_ids = token_ids.data();
    io.token_count = k_prompt_tokens;
    io.logits = logits.data();
    io.logits_capacity = k_vocab;
    request.step_plan = &plan;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = k_prompt_tokens;
    request.kv_tokens = 0;

    ready =
        emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
            square_rows.data(),
            static_cast<uint64_t>(k_embd),
            static_cast<uint64_t>(k_embd),
            packed_square_storage.data()) &&
        emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
            shortconv_in_rows.data(),
            static_cast<uint64_t>(3 * k_embd),
            static_cast<uint64_t>(k_embd),
            packed_shortconv_in_storage.data()) &&
        emel::generator::detail::prepare_q8_input_workspace(backend) &&
        emel::generator::detail::prepare_q8_input_chunk4_workspace(backend) &&
        emel::generator::detail::prepare_q8_input_chunk8_workspace(backend);
  }
};

using hybrid_chunk4_q8_runtime_fixture = hybrid_chunked_q8_runtime_fixture<4>;
using hybrid_chunk8_q8_runtime_fixture = hybrid_chunked_q8_runtime_fixture<8>;

float round_fp16_value(const float value) {
  return emel::generator::detail::quant::fp16_to_fp32(
      emel::generator::detail::quant::fp32_to_fp16(value));
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

void apply_rope_reference(std::span<float> vector,
                          const int32_t head_count,
                          const int32_t head_dim,
                          const int32_t n_rot,
                          const int32_t position,
                          const float rope_freq_base) {
  const int32_t rot_dim = std::min(n_rot, head_dim);
  if (head_count <= 0 || head_dim <= 1 || rot_dim <= 1) {
    return;
  }

  const float theta_scale = ::powf(rope_freq_base, -2.0f / static_cast<float>(rot_dim));
  for (int32_t head = 0; head < head_count; ++head) {
    float * head_ptr =
        vector.data() + (static_cast<size_t>(head) * static_cast<size_t>(head_dim));
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

std::vector<float> flash_attention_online_reference(
    const emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position,
    const std::span<const float> q_vector) {
  const int32_t head_count = backend.n_head;
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const uint64_t kv_tokens = static_cast<uint64_t>(position + 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> out(static_cast<size_t>(head_count) * static_cast<size_t>(head_dim), 0.0f);
  std::vector<uint16_t> head_k(
      static_cast<size_t>(head_dim) * static_cast<size_t>(kv_tokens), 0u);
  std::vector<uint16_t> head_v(
      static_cast<size_t>(head_dim) * static_cast<size_t>(kv_tokens), 0u);

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);

    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const size_t src_offset = emel::generator::detail::flash_layer_cache_head_position_offset(
          backend, layer_index, kv_head, static_cast<int32_t>(token), kv_head_dim);
      const size_t dst_offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const size_t dim_offset = static_cast<size_t>(dim);
        head_k[dst_offset + dim_offset] = backend.flash_key_cache[src_offset + dim_offset];
        head_v[dst_offset + dim_offset] = backend.flash_value_cache[src_offset + dim_offset];
      }
    }

    const std::vector<float> expected_head = flash_attn_reference_online_softmax_f16_values(
        q_vector.subspan(q_offset, static_cast<size_t>(head_dim)),
        std::span<const uint16_t>(head_k.data(), head_k.size()),
        std::span<const uint16_t>(head_v.data(), head_v.size()),
        static_cast<uint64_t>(head_dim),
        kv_tokens,
        scale);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      out[q_offset + static_cast<size_t>(dim)] = expected_head[static_cast<size_t>(dim)];
    }
  }

  return out;
}

}  // namespace

TEST_CASE("generator_detail_fp16_to_fp32_handles_normal_special_and_subnormal_values") {
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x3c00u) == doctest::Approx(1.0f));
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x3800u) == doctest::Approx(0.5f));
  CHECK(std::isinf(emel::generator::detail::quant::fp16_to_fp32(0x7c00u)));
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x0001u) > 0.0f);
}

TEST_CASE("generator_detail_fp16_conversion_matches_native_arm_fp16_rounding") {
#if defined(__ARM_NEON) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && \
    !defined(__MUSACC__)
  constexpr std::array<float, 8> samples = {
      0.0f,
      0.5f,
      -0.934325f,
      0.0345459f,
      -36.4516f,
      65504.0f,
      6.1035156e-05f,
      -6.1035156e-05f,
  };

  for (const float sample : samples) {
    uint16_t native_bits = 0u;
    const __fp16 native_value = sample;
    std::memcpy(&native_bits, &native_value, sizeof(native_bits));

    CHECK(emel::generator::detail::quant::fp32_to_fp16(sample) == native_bits);
    CHECK(emel::generator::detail::quant::fp16_to_fp32(native_bits) ==
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

  emel::generator::detail::apply_rope(actual, 1, 64, 64, 103, 10000.0f);
  apply_rope_reference(reference, 1, 64, 64, 103, 10000.0f);

  for (size_t idx = 0; idx < actual.size(); ++idx) {
    CHECK(actual[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-8));
  }
}

TEST_CASE("generator_detail_dequantizes_q2_k_blocks") {
  block_q2_k block = {};
  block.d = 0x3c00u;
  block.dmin = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(), static_cast<uint8_t>(0x11u));
  std::fill(block.qs.begin(), block.qs.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::generator::detail::quant::dequantize_row_q2_k(&block, out.data(), QK_K);

  CHECK(out.front() == doctest::Approx(-1.0f));
  CHECK(out[127] == doctest::Approx(-1.0f));
  CHECK(out.back() == doctest::Approx(-1.0f));
}

TEST_CASE("generator_detail_dequantizes_q3_k_blocks_without_unsigned_wrap") {
  block_q3_k block = {};
  block.d = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(), static_cast<uint8_t>(0x00u));
  std::fill(block.hmask.begin(), block.hmask.end(), static_cast<uint8_t>(0x00u));
  std::fill(block.qs.begin(), block.qs.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::generator::detail::quant::dequantize_row_q3_k(&block, out.data(), QK_K);

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
  emel::generator::detail::quant::dequantize_row_q6_k(&block, out.data(), QK_K);

  CHECK(out.front() == doctest::Approx(-32.0f));
  CHECK(out[127] == doctest::Approx(-32.0f));
  CHECK(out.back() == doctest::Approx(-32.0f));
}

TEST_CASE("generator_detail_dequantizes_q8_0_blocks") {
  block_q8_0 block = {};
  block.d = emel::generator::detail::quant::fp32_to_fp16(0.5f);
  for (size_t idx = 0; idx < block.qs.size(); ++idx) {
    block.qs[idx] = static_cast<int8_t>((static_cast<int32_t>(idx % 7u)) - 3);
  }

  std::array<float, QK8_0> out = {};
  emel::generator::detail::quant::dequantize_row_q8_0(&block, out.data(), QK8_0);

  CHECK(out.front() == doctest::Approx(-1.5f));
  CHECK(out[1] == doctest::Approx(-1.0f));
  CHECK(out[2] == doctest::Approx(-0.5f));
  CHECK(out.back() == doctest::Approx(0.0f));
}

TEST_CASE("generator_detail_prepares_explicit_logits_routes_and_support_predicates") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.n_embd = static_cast<int32_t>(QK_K);
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  CHECK(emel::generator::detail::packed_q6_k_x8_logits_supported(backend));
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(emel::generator::detail::prepared_q6_k_x8_q8_logits_supported(backend));
#endif
  CHECK_FALSE(emel::generator::detail::q8_input_workspace_candidate({}));
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(backend));

  REQUIRE(emel::generator::detail::prepare_output_logits(backend));
  REQUIRE(emel::generator::detail::prepare_q8_input_workspace(backend));
  CHECK(backend.q8_input_storage.size() == 1u);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, backend.output));
  CHECK(emel::generator::detail::q8_input_argmax_path_supported(backend, backend.output_argmax));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared);
  CHECK(emel::generator::detail::row_storage_bytes(*backend.output.tensor,
                                                   static_cast<int32_t>(QK_K)) ==
        emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(QK_K));
  CHECK(emel::generator::detail::row_storage_bytes(*backend.output_argmax.tensor,
                                                   static_cast<int32_t>(QK_K)) ==
        emel::kernel::detail::quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(QK_K));
  CHECK(emel::generator::detail::preselected_argmax_direct_supported(backend));
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) == emel::kernel::detail::dtype_q6_k_x8);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::generator::detail::preselected_argmax_direct_supported(backend));
#else
  CHECK(backend.output.tensor == &q6_tensor);
  CHECK(backend.output_argmax.tensor == &q6_tensor);
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(backend));
#endif
}

TEST_CASE("generator_detail_explicit_logits_routes_cover_packed_and_passthrough_helpers") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend backend{};
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  emel::generator::detail::reset_output_logits(backend);
  CHECK(backend.output.tensor == &q6_tensor);
  CHECK(backend.output_argmax.tensor == &q6_tensor);
  CHECK(backend.output_packed_storage.empty());
  CHECK(backend.output_prepared_storage.empty());
  CHECK(backend.output_argmax_prepared_storage.empty());

  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  REQUIRE(emel::generator::detail::prepare_packed_output_logits(backend));
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) == emel::kernel::detail::dtype_q6_k_x8);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::generator::detail::matrix_buffer_bytes(backend.output) ==
        emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(QK_K));

  std::array<float, 8> f32_rows = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
  auto f32_tensor =
      make_tensor_record(f32_rows.data(), emel::kernel::detail::dtype_f32, 4, 2);
  emel::generator::detail::native_backend passthrough{};
  passthrough.output_native.tensor = &f32_tensor;
  passthrough.output_native.cols = 4;
  passthrough.output_native.rows = 2;
  passthrough.output = passthrough.output_native;
  passthrough.output_argmax = passthrough.output_native;
  REQUIRE(emel::generator::detail::prepare_output_logits(passthrough));
  CHECK(passthrough.output.tensor == &f32_tensor);
  CHECK(passthrough.output_argmax.tensor == &f32_tensor);
}

TEST_CASE("generator_detail_routes_static_q6_block_matrices_through_generic_q8_input") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(emel::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.blocks.resize(1u);
  auto & block = backend.blocks.front();
  block.uses_attention = true;
  block.attention_q = q6_matrix;
  block.attention_k = q6_matrix;
  block.attention_v = q6_matrix;
  block.attention_output = q6_matrix;
  block.feed_forward_gate = q6_matrix;
  block.feed_forward_down = q6_matrix;
  block.feed_forward_up = q6_matrix;

  REQUIRE(emel::generator::detail::prepare_block_native_matrices(backend));
  REQUIRE(emel::generator::detail::prepare_q8_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);

  std::array<float, QK_K> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    input[idx] = static_cast<float>(static_cast<int32_t>((idx * 7u) % 19u) - 9) * 0.125f;
  }

  emel::generator::detail::native_backend reference_backend{};
  reference_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  std::vector<float> reference(Q6_K_X8_ROWS, 0.0f);
  REQUIRE(emel::generator::detail::matmul_vector(
      reference_backend,
      q6_matrix,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(reference.data(), reference.size())));

  std::vector<float> output(Q6_K_X8_ROWS, 0.0f);
  REQUIRE(emel::generator::detail::matmul_vector(
      backend,
      block.feed_forward_down,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));

  for (size_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    CHECK(output[row] == doctest::Approx(reference[row]).epsilon(1.0e-5f));
  }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);
#else
  CHECK(block.feed_forward_down.tensor == &q6_tensor);
  CHECK_FALSE(emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.empty());
#endif
}

TEST_CASE("generator_detail_routes_static_q4_block_matrices_through_generic_q8_input") {
  auto q4_rows = make_q4_rows();
  auto q4_tensor = make_tensor_record(
      q4_rows.data(), emel::kernel::detail::dtype_q4_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q4_K_X8_ROWS));

  emel::generator::detail::tensor_matrix q4_matrix = {};
  REQUIRE(emel::generator::detail::bind_tensor_rows(q4_tensor, q4_matrix));

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.blocks.resize(1u);
  auto & block = backend.blocks.front();
  block.uses_attention = true;
  block.attention_q = q4_matrix;
  block.attention_k = q4_matrix;
  block.attention_v = q4_matrix;
  block.attention_output = q4_matrix;
  block.feed_forward_gate = q4_matrix;
  block.feed_forward_down = q4_matrix;
  block.feed_forward_up = q4_matrix;

  REQUIRE(emel::generator::detail::prepare_block_native_matrices(backend));
  REQUIRE(emel::generator::detail::prepare_q8_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q4_k_x8_bl8);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);

  std::array<float, QK_K> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    input[idx] = static_cast<float>(static_cast<int32_t>((idx * 5u) % 21u) - 10) * 0.125f;
  }

  emel::generator::detail::native_backend reference_backend{};
  reference_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  std::vector<float> reference(Q4_K_X8_ROWS, 0.0f);
  REQUIRE(emel::generator::detail::matmul_vector(
      reference_backend,
      q4_matrix,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(reference.data(), reference.size())));

  std::vector<float> output(Q4_K_X8_ROWS, 0.0f);
  REQUIRE(emel::generator::detail::matmul_vector(
      backend,
      block.feed_forward_down,
      std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));

  for (size_t row = 0; row < Q4_K_X8_ROWS; ++row) {
    CHECK(output[row] == doctest::Approx(reference[row]).epsilon(1.0e-5f));
  }
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(static_cast<uint8_t>(block.feed_forward_down.tensor->type) ==
        emel::kernel::detail::dtype_q4_k_x8_bl4);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.size() == 1u);
#else
  CHECK(block.feed_forward_down.tensor == &q4_tensor);
  CHECK_FALSE(emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_down));
  CHECK(backend.q8_input_storage.empty());
#endif
}

TEST_CASE("generator_detail_q6_logits_paths_slice_oversized_q8_workspace") {
#if !(defined(__aarch64__) && defined(__ARM_NEON))
  SUCCEED();
#else
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
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
        static_cast<float>(static_cast<int32_t>((idx * 9u) % 23u) - 11) * 0.0625f;
  }

  REQUIRE(emel::generator::detail::prepare_output_logits(backend));
  backend.q8_input_storage.resize(2u);
  REQUIRE(emel::generator::detail::q8_input_path_supported(backend, backend.output));
  REQUIRE(emel::generator::detail::q8_input_argmax_path_supported(backend, backend.output_argmax));
  REQUIRE(emel::generator::detail::compute_logits(backend));

  int32_t selected_index = -1;
  float selected_score = 0.0f;
  REQUIRE(emel::generator::detail::compute_logits_preselected_argmax(
      backend, selected_index, selected_score));
  CHECK(selected_index >= 0);
  CHECK(selected_index < static_cast<int32_t>(Q6_K_X8_ROWS));
#endif
}

TEST_CASE("generator_detail_prepare_block_native_matrices_supports_shortconv_q6_routes") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(emel::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.blocks.resize(1u);
  auto & block = backend.blocks.front();
  block.uses_attention = false;
  block.shortconv_in_proj = q6_matrix;
  block.shortconv_out_proj = q6_matrix;
  block.feed_forward_gate = q6_matrix;
  block.feed_forward_down = q6_matrix;
  block.feed_forward_up = q6_matrix;

  REQUIRE(emel::generator::detail::prepare_block_native_matrices(backend));
  REQUIRE(emel::generator::detail::prepare_q8_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(static_cast<uint8_t>(block.shortconv_out_proj.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, block.shortconv_out_proj));
  CHECK(backend.q8_input_storage.size() == 1u);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(static_cast<uint8_t>(block.shortconv_out_proj.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::generator::detail::q8_input_path_supported(backend, block.shortconv_out_proj));
  CHECK(backend.q8_input_storage.size() == 1u);
#else
  CHECK(block.shortconv_out_proj.tensor == &q6_tensor);
  CHECK_FALSE(emel::generator::detail::q8_input_path_supported(backend, block.shortconv_out_proj));
  CHECK(backend.q8_input_storage.empty());
#endif
}

TEST_CASE("generator_detail_prepare_output_logits_packs_q8_0_outputs") {
  std::array<block_q8_0, 4> q8_rows = {};
  for (size_t row = 0; row < q8_rows.size(); ++row) {
    q8_rows[row].d = fp16_bits(0.03125f * static_cast<float>(row + 1u));
    for (size_t idx = 0; idx < q8_rows[row].qs.size(); ++idx) {
      q8_rows[row].qs[idx] =
          static_cast<int8_t>(static_cast<int32_t>(((row + 3u) * 19u + idx * 7u) % 127u) - 63);
    }
  }

  auto q8_tensor = make_tensor_record(
      q8_rows.data(), emel::kernel::detail::dtype_q8_0, static_cast<int32_t>(QK8_0), 4);
  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output_native.tensor = &q8_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK8_0);
  backend.output_native.rows = 4;
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  REQUIRE(emel::generator::detail::prepare_output_logits(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(backend.output.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q8_0_x4_bl8);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  REQUIRE(backend.output.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q8_0_x4_bl4);
#else
  CHECK(backend.output.tensor == &q8_tensor);
#endif
}

TEST_CASE("generator_detail_tensor_binding_and_copy_helpers_accept_explicit_quantized_routes") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(emel::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));
  CHECK(q6_matrix.cols == static_cast<int32_t>(QK_K));
  CHECK(q6_matrix.rows == static_cast<int32_t>(Q6_K_X8_ROWS));

  auto invalid_q6_tensor =
      make_tensor_record(q6_rows.data(), emel::kernel::detail::dtype_q6_k, 8, 2);
  emel::generator::detail::tensor_matrix invalid_matrix = {};
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(invalid_q6_tensor, invalid_matrix));

  block_q3_k q3_block = {};
  q3_block.d = 0x3c00u;
  std::fill(q3_block.scales.begin(), q3_block.scales.end(), static_cast<uint8_t>(0x00u));
  std::fill(q3_block.hmask.begin(), q3_block.hmask.end(), static_cast<uint8_t>(0x00u));
  std::fill(q3_block.qs.begin(), q3_block.qs.end(), static_cast<uint8_t>(0x00u));
  auto q3_tensor = make_tensor_record(&q3_block, emel::kernel::detail::dtype_q3_k,
                                      static_cast<int32_t>(QK_K), 1);
  std::vector<float> q3_out(QK_K, 0.0f);
  REQUIRE(emel::generator::detail::copy_tensor_row(q3_tensor, 0, q3_out));
  CHECK(q3_out.front() == doctest::Approx(128.0f));
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(q3_tensor, 1, q3_out));

  block_q8_0 q8_block = {};
  q8_block.d = emel::generator::detail::quant::fp32_to_fp16(0.25f);
  q8_block.qs.fill(4);
  auto q8_tensor = make_tensor_record(&q8_block, emel::kernel::detail::dtype_q8_0,
                                      static_cast<int32_t>(QK8_0), 1);
  std::vector<float> q8_out(QK8_0, 0.0f);
  REQUIRE(emel::generator::detail::copy_tensor_row(q8_tensor, 0, q8_out));
  CHECK(q8_out.front() == doctest::Approx(1.0f));
  CHECK(q8_out.back() == doctest::Approx(1.0f));

  std::vector<float> q6_out = {};
  REQUIRE(emel::generator::detail::dequantize_tensor_vector(q6_tensor, q6_out) == false);
  auto q6_vector_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K), 1);
  REQUIRE(emel::generator::detail::dequantize_tensor_vector(q6_vector_tensor, q6_out));
  std::vector<float> q6_expected(QK_K, 0.0f);
  emel::generator::detail::quant::dequantize_row_q6_k(q6_rows.data(), q6_expected.data(), QK_K);
  CHECK(q6_out.size() == QK_K);
  CHECK(q6_out.front() == doctest::Approx(q6_expected.front()));
  CHECK(q6_out.back() == doctest::Approx(q6_expected.back()));
}

TEST_CASE("generator_detail_logits_route_helpers_cover_explicit_failure_edges") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::generator::detail::native_backend backend{};
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
  CHECK(emel::generator::detail::preselected_argmax_direct_supported(backend));

  emel::model::data::tensor_record unsupported_tensor = q6_tensor;
  unsupported_tensor.type = emel::kernel::detail::dtype_f32;
  backend.output_argmax.tensor = &unsupported_tensor;
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(backend));

  emel::generator::detail::native_backend null_backend{};
  CHECK_FALSE(emel::generator::detail::prepare_output_logits(null_backend));
  CHECK_FALSE(emel::generator::detail::prepare_prepared_output_logits(null_backend));
  CHECK_FALSE(emel::generator::detail::prepare_packed_output_logits(null_backend));

  auto invalid_q6_tensor =
      make_tensor_record(q6_rows.data(), emel::kernel::detail::dtype_q6_k, 8, 2);
  emel::generator::detail::native_backend invalid_backend{};
  invalid_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  invalid_backend.output_native.tensor = &invalid_q6_tensor;
  invalid_backend.output_native.cols = 8;
  invalid_backend.output_native.rows = 2;
  invalid_backend.output = invalid_backend.output_native;
  invalid_backend.output_argmax = invalid_backend.output_native;
  CHECK_FALSE(emel::generator::detail::prepare_prepared_output_logits(invalid_backend));
  CHECK_FALSE(emel::generator::detail::prepare_packed_output_logits(invalid_backend));

  emel::generator::detail::native_backend no_simd_backend{};
  no_simd_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  no_simd_backend.output_native.tensor = &q6_tensor;
  no_simd_backend.output_native.cols = static_cast<int32_t>(QK_K);
  no_simd_backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  no_simd_backend.output = no_simd_backend.output_native;
  no_simd_backend.output_argmax = no_simd_backend.output_native;
  REQUIRE(emel::generator::detail::prepare_output_logits(no_simd_backend));
  CHECK(no_simd_backend.output.tensor == &q6_tensor);
  CHECK(no_simd_backend.output_argmax.tensor == &q6_tensor);

  std::vector<uint8_t> oversized_packed_storage(
      emel::generator::detail::row_storage_bytes(
          packed_tensor,
          static_cast<int32_t>(
              (emel::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) * QK_K)) *
      static_cast<size_t>(Q6_K_X8_ROWS));
  auto oversized_packed_tensor = make_tensor_record(
      oversized_packed_storage.data(),
      emel::kernel::detail::dtype_q6_k_x8,
      static_cast<int32_t>((emel::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) * QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend huge_backend{};
  huge_backend.output.tensor = &oversized_packed_tensor;
  huge_backend.output.cols = static_cast<int32_t>(
      (emel::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) * QK_K);
  huge_backend.output.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  huge_backend.output_argmax = huge_backend.output;
  CHECK_FALSE(emel::generator::detail::prepare_q8_input_workspace(huge_backend));
}

TEST_CASE("generator_detail_packed_q8_0_input_workspace_helpers_are_explicit") {
  std::vector<uint8_t> packed_storage(sizeof(emel::kernel::detail::quant::block_q8_0x4));
  auto packed_tensor = make_tensor_record(
      packed_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8,
      static_cast<int32_t>(QK8_0), 4);

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output.tensor = &packed_tensor;
  backend.output.cols = static_cast<int32_t>(QK8_0);
  backend.output.rows = 4;

  REQUIRE(emel::generator::detail::prepare_packed_q8_0_input_workspace(backend));
  CHECK(backend.packed_q8_0_input_storage.size() == 1u);

#if defined(__aarch64__) && defined(__ARM_NEON)
  CHECK(emel::generator::detail::packed_q8_0_input_path_supported(backend, backend.output));
#else
  CHECK_FALSE(emel::generator::detail::packed_q8_0_input_path_supported(backend, backend.output));
#endif

  std::vector<uint8_t> oversized_storage(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) *
      (emel::generator::detail::quant::MAX_Q8_0_BLOCKS + 1u));
  auto oversized_tensor = make_tensor_record(
      oversized_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8,
      static_cast<int32_t>(
          (emel::generator::detail::quant::MAX_Q8_0_BLOCKS + 1u) * QK8_0),
      4);
  backend.output.tensor = &oversized_tensor;
  backend.output.cols = static_cast<int32_t>(
      (emel::generator::detail::quant::MAX_Q8_0_BLOCKS + 1u) * QK8_0);
  backend.output.rows = 4;
  CHECK_FALSE(emel::generator::detail::prepare_packed_q8_0_input_workspace(backend));
}

TEST_CASE("generator_detail_prefill_chunk4_q8_gemm_support_is_explicit") {
  constexpr int32_t rows = 4;
  constexpr int32_t cols = static_cast<int32_t>(QK8_0);
  std::vector<uint8_t> packed_storage(sizeof(emel::kernel::detail::quant::block_q8_0x4));
  auto packed_tensor = make_tensor_record(
      packed_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8, cols, rows);

  emel::generator::detail::tensor_matrix packed_matrix = {};
  packed_matrix.tensor = &packed_tensor;
  packed_matrix.rows = rows;
  packed_matrix.cols = cols;

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.n_layer = 1;
  backend.n_embd = 4;
  backend.n_head = 1;
  backend.head_dim = 4;
  backend.n_head_kv = 1;
  backend.head_dim_kv = 4;
  backend.output = packed_matrix;
  backend.blocks.resize(1u);
  auto & block = backend.blocks.front();
  block.attention_q = packed_matrix;
  block.attention_k = packed_matrix;
  block.attention_v = packed_matrix;
  block.attention_output = packed_matrix;
  block.feed_forward_gate = packed_matrix;
  block.feed_forward_down = packed_matrix;
  block.feed_forward_up = packed_matrix;

  REQUIRE(emel::generator::detail::prepare_packed_q8_0_chunk4_input_workspace(backend));
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

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(backend));
  backend.k_chunk4.clear();
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(backend));
  backend.k_chunk4.resize(16u);
  backend.v_chunk4.clear();
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(backend));
  backend.v_chunk4.resize(16u);
  backend.up_chunk4.clear();
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(backend));
#else
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(backend));
#endif
}

TEST_CASE("generator_detail_prefill_chunk4_q8_gemm_support_accepts_hybrid_q8_k_x4_paths") {
  auto fixture = std::make_unique<hybrid_chunk4_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(fixture->backend));
  fixture->backend.q8_input_chunk4_storage.clear();
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
      fixture->backend));
  fixture->backend.q8_input_chunk4_storage.resize(
      fixture->backend.q8_input_storage.size() * static_cast<size_t>(4u));
  fixture->backend.shortconv_bcx_chunk4.clear();
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
      fixture->backend));
#else
  CHECK_FALSE(emel::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
      fixture->backend));
#endif
}

TEST_CASE("generator_detail_chunk4_packed_q8_0_helpers_are_explicit_and_numeric_match") {
  using emel::generator::detail::quant::Q8_0_X4_ROWS;
  using emel::kernel::detail::quant::block_q8_0;

  constexpr int32_t row_count = 8;
  constexpr int32_t rhs_rows = static_cast<int32_t>(Q8_0_X4_ROWS);
  constexpr int32_t col_count = static_cast<int32_t>(QK8_0 * 8u);
  constexpr size_t block_count = static_cast<size_t>(col_count / static_cast<int32_t>(QK8_0));

  std::vector<block_q8_0> native_rows(static_cast<size_t>(row_count) * block_count);
  for (int32_t row = 0; row < row_count; ++row) {
    for (size_t block = 0; block < block_count; ++block) {
      auto & cell = native_rows[static_cast<size_t>(row) * block_count + block];
      cell.d = emel::kernel::detail::quant::fp32_to_fp16(
          0.0078125f * static_cast<float>(((row + 3) * static_cast<int32_t>(block + 5)) % 23 + 1));
      for (size_t idx = 0; idx < cell.qs.size(); ++idx) {
        const int32_t centered =
            static_cast<int32_t>(((row + 7) * 17 + (block + 11) * 13 + idx * 5) % 255) - 127;
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
                static_cast<size_t>(col)] = static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_0> rhs_q8(static_cast<size_t>(rhs_rows) * block_count);
  for (int32_t row = 0; row < rhs_rows; ++row) {
    emel::kernel::detail::quant::quantize_row_q8_0_strided(
        rhs_dense.data() + static_cast<size_t>(row) * static_cast<size_t>(col_count),
        1u,
        rhs_q8.data() + static_cast<size_t>(row) * block_count,
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
      emel::kernel::detail::quant::packed_q8_0_x4_group_count(static_cast<uint64_t>(row_count)) *
      block_count);
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      native_rows.data(), static_cast<uint64_t>(row_count), static_cast<uint64_t>(col_count),
      packed_storage.data()));

  auto packed_tensor = make_tensor_record(
      packed_storage.data(), emel::kernel::detail::dtype_q8_0_x4_bl8, col_count, row_count);
  emel::generator::detail::tensor_matrix packed_matrix = {};
  packed_matrix.tensor = &packed_tensor;
  packed_matrix.rows = row_count;
  packed_matrix.cols = col_count;

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output = packed_matrix;
  REQUIRE(emel::generator::detail::prepare_packed_q8_0_chunk4_input_workspace(backend));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(emel::generator::detail::prepare_packed_q8_0_chunk4_input(
      backend, rhs_dense, col_count));
  std::vector<float> output(
      static_cast<size_t>(rhs_rows) * static_cast<size_t>(row_count), 0.0f);
  REQUIRE(emel::generator::detail::matmul_chunk4_prepared_packed_q8_0_input(
      backend, packed_matrix, col_count, output));
  for (size_t idx = 0; idx < output.size(); ++idx) {
    CHECK(output[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-6f));
  }
  CHECK(backend.kernel_dispatch_calls == 1u);
  CHECK(backend.packed_q8_0_dispatch_calls == 1u);
#else
  CHECK_FALSE(emel::generator::detail::prepare_packed_q8_0_chunk4_input(
      backend, rhs_dense, col_count));
  std::vector<float> output(
      static_cast<size_t>(rhs_rows) * static_cast<size_t>(row_count), 0.0f);
  CHECK_FALSE(emel::generator::detail::matmul_chunk4_prepared_packed_q8_0_input(
      backend, packed_matrix, col_count, output));
#endif
}

TEST_CASE("generator_detail_tensor_helpers_reject_invalid_records_explicitly") {
  emel::generator::detail::tensor_matrix out = {};
  emel::model::data::tensor_record empty_tensor = {};
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(empty_tensor, out));

  std::array<float, 4> f32_data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto f32_tensor =
      make_tensor_record(f32_data.data(), emel::kernel::detail::dtype_f32, 2, 2);
  std::vector<float> copy_out(2u, 0.0f);
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(f32_tensor, -1, copy_out));

  emel::model::data::tensor_record bad_dims = f32_tensor;
  bad_dims.dims[1] = 0u;
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(bad_dims, out));

  emel::model::data::tensor_record unsupported = f32_tensor;
  unsupported.type = 255;
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(unsupported, out));
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(unsupported, 0, copy_out));

  std::vector<float> wrong_size(3u, 0.0f);
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(f32_tensor, 0, wrong_size));
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(f32_tensor, 3, copy_out));
}

TEST_CASE("generator_detail_numeric_helpers_reject_invalid_shapes_explicitly") {
  emel::generator::detail::native_backend backend{};
  emel::generator::detail::tensor_matrix empty_matrix = {};
  std::array<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 2> output = {};
  int32_t selected_index = -1;
  float selected_score = 0.0f;
  std::array<emel::kernel::detail::quant::block_q8_k, 1> q8_input = {};
  std::array<emel::kernel::detail::quant::block_q8_0, 1> q8_0_input = {};

  CHECK_FALSE(emel::generator::detail::matmul_vector(
      backend, empty_matrix, std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_argmax(
      backend,
      empty_matrix,
      std::span<const float>(input.data(), input.size()),
      selected_index,
      selected_score));
  CHECK_FALSE(emel::generator::detail::quantize_vector_q8_k(
      std::span<const float>(input.data(), 3u), std::span(q8_input.data(), q8_input.size())));
  CHECK_FALSE(emel::generator::detail::quantize_vector_q8_0(
      std::span<const float>(input.data(), 3u), std::span(q8_0_input.data(), q8_0_input.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_q8_input(
      backend,
      empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_k>(q8_input.data(), q8_input.size()),
      static_cast<int32_t>(QK_K),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_q8_0_input(
      backend,
      empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_0>(q8_0_input.data(), q8_0_input.size()),
      static_cast<int32_t>(QK8_0),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_q8_input_argmax(
      backend,
      empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_k>(q8_input.data(), q8_input.size()),
      static_cast<int32_t>(QK_K),
      selected_index,
      selected_score));

  std::array<float, 2> weight = {1.0f, 1.0f};
  CHECK_FALSE(emel::generator::detail::rms_norm(
      std::span<const float>(input.data(), 0u),
      std::span<const float>(weight.data(), weight.size()),
      1.0e-5f,
      std::span<float>(output.data(), output.size())));

  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend packed_backend{};
  packed_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  packed_backend.output_native.tensor = &q6_tensor;
  packed_backend.output_native.cols = static_cast<int32_t>(QK_K);
  packed_backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  packed_backend.output = packed_backend.output_native;
  packed_backend.output_argmax = packed_backend.output_native;
  REQUIRE(emel::generator::detail::prepare_packed_output_logits(packed_backend));
  std::vector<float> packed_copy(QK_K, 0.0f);
  CHECK_FALSE(
      emel::generator::detail::copy_tensor_row(*packed_backend.output.tensor, 0, packed_copy));

  std::array<float, 4> rope_identity = {1.0f, 2.0f, 3.0f, 4.0f};
  const auto rope_before = rope_identity;
  emel::generator::detail::apply_rope(rope_identity, 1, 1, 1, 3, 10000.0f);
  CHECK(rope_identity == rope_before);
}

TEST_CASE("generator_detail_request_and_backend_validators_reject_invalid_inputs_explicitly") {
  int32_t err = 123;
  CHECK_FALSE(emel::generator::detail::check_backend(nullptr, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  emel::generator::detail::native_backend backend{};
  backend.model = reinterpret_cast<const emel::model::data *>(0x1);
  backend.n_embd = 4;
  backend.n_head = 1;
  backend.n_head_kv = 1;
  backend.n_layer = 1;
  backend.n_vocab = 8;
  backend.n_ctx = 4;
  backend.head_dim = 4;
  backend.head_dim_kv = 4;
  CHECK_FALSE(emel::generator::detail::check_backend(&backend, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  emel::graph::processor::event::execute request = {};
  CHECK(emel::generator::detail::request_plan(request, &err) == nullptr);
  CHECK(err == emel::generator::detail::k_error_invalid);

  backend.bound_tokens.resize(2u);
  backend.bound_positions.resize(2u);
  CHECK_FALSE(emel::generator::detail::store_bound_request(backend, request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  std::array<float, 4> probs = {1.0f, 1.0f, 1.0f, 1.0f};
  std::array<float, 4> rounded = {1.0f, 1.0f, 1.0f, 1.0f};
  emel::generator::detail::fill_masked_softmax_probs_ggml(
      std::span<const float>(), 0, probs, rounded);
  CHECK(std::all_of(probs.begin(), probs.end(), [](const float value) { return value == 0.0f; }));
  CHECK(std::all_of(
      rounded.begin(), rounded.end(), [](const float value) { return value == 0.0f; }));
}

TEST_CASE("generator_detail_runtime_wrappers_validate_requests_explicitly") {
  runtime_request_fixture fixture{};
  int32_t err = -1;
  bool reused = true;

  CHECK(emel::generator::detail::validate(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.request.expected_outputs = 2;
  CHECK_FALSE(emel::generator::detail::validate(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  fixture.request.expected_outputs = fixture.plan.expected_outputs;

  CHECK(emel::generator::detail::validate_preselected_argmax(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.io.selected_score_out = nullptr;
  CHECK_FALSE(emel::generator::detail::validate_preselected_argmax(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  fixture.io.selected_score_out = &fixture.selected_score;

  CHECK(emel::generator::detail::prepare_graph(fixture.request, &reused, &err));
  CHECK_FALSE(reused);
  CHECK(err == emel::generator::detail::k_error_ok);

  CHECK(emel::generator::detail::alloc_graph(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.request.compute_ctx = nullptr;
  CHECK_FALSE(emel::generator::detail::bind_inputs(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
}

TEST_CASE("generator_detail_runtime_wrappers_bind_run_and_extract_explicitly") {
  runtime_request_fixture fixture{};
  int32_t err = -1;
  int32_t outputs = 0;

  CHECK(emel::generator::detail::bind_inputs(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture.backend.bound_ready);
  CHECK(fixture.backend.bound_token_count == 1);
  CHECK(fixture.backend.bound_position_count == 1);

  fixture.backend.bound_ready = false;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_preselected_argmax(
      fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  fixture.backend.bound_ready = true;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_preselected_argmax(
      fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  CHECK(emel::generator::detail::extract_outputs(fixture.request, &outputs, &err));
  CHECK(outputs == 1);
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture.logits[0] == doctest::Approx(0.25f));
  CHECK(fixture.logits[1] == doctest::Approx(0.5f));
  CHECK(fixture.logits[2] == doctest::Approx(0.75f));
  CHECK(fixture.logits[3] == doctest::Approx(1.0f));
  CHECK(fixture.logits[4] == doctest::Approx(-1.0f));
  CHECK(fixture.logits[5] == doctest::Approx(-1.0f));

  fixture.io.logits = nullptr;
  CHECK_FALSE(emel::generator::detail::extract_outputs(fixture.request, &outputs, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  fixture.io.logits = fixture.logits.data();

  CHECK(emel::generator::detail::extract_preselected_argmax(fixture.request, &outputs, &err));
  CHECK(outputs == 1);
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.io.selected_score_out = nullptr;
  CHECK_FALSE(emel::generator::detail::extract_preselected_argmax(fixture.request, &outputs, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
}

TEST_CASE("generator_detail_builds_flash_request_over_head_major_kv_cache") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 2;
  backend.q = {1.0f, 0.1f, 0.2f, 1.0f};
  backend.q_attn = {9.0f, 9.0f, 9.0f, 9.0f};
  backend.flash_key_cache = {
      fp16_bits(1.0f), fp16_bits(0.0f), fp16_bits(0.0f), fp16_bits(1.0f),
      fp16_bits(0.5f), fp16_bits(0.5f), fp16_bits(0.25f), fp16_bits(0.75f),
  };
  backend.flash_value_cache = {
      fp16_bits(2.0f), fp16_bits(0.0f), fp16_bits(0.0f), fp16_bits(4.0f),
      fp16_bits(1.0f), fp16_bits(1.0f), fp16_bits(0.5f), fp16_bits(1.5f),
  };
  backend.attn_ctx.resize(4);

  const auto request = emel::generator::detail::make_flash_attn_request(backend, 0, 1);
  float scale = 0.0f;
  uint32_t total_tokens = 0u;
  std::memcpy(&scale, request.op_params.data(), sizeof(scale));
  std::memcpy(&total_tokens, request.op_params.data() + sizeof(scale), sizeof(total_tokens));

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

TEST_CASE("generator_detail_flash_dispatch_matches_online_softmax_reference_on_same_backend_state") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 12;
  backend.n_head_kv = 12;
  backend.n_rep = 1;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 256;
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;

  const size_t n_embd =
      static_cast<size_t>(backend.n_head) * static_cast<size_t>(backend.head_dim);
  const size_t kv_dim =
      static_cast<size_t>(backend.n_head_kv) * static_cast<size_t>(backend.head_dim_kv);
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
    const float raw = static_cast<float>(std::sin(static_cast<double>(idx + 1u) * 0.03125));
    backend.q[idx] = raw;
  }

  for (int32_t token = 0; token < position_limit; ++token) {
    for (int32_t head = 0; head < backend.n_head_kv; ++head) {
      for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
        const size_t offset = emel::generator::detail::flash_layer_cache_head_position_offset(
            backend, 0, head, token, backend.head_dim_kv) + static_cast<size_t>(dim);
        const double base =
            static_cast<double>((token + 1) * (head + 3) * (dim + 5));
        backend.flash_key_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::cos(base * 0.0078125)));
        backend.flash_value_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(base * 0.01171875)));
      }
    }
  }

  std::vector<float> flash_ctx(backend.attn_ctx.size(), 0.0f);
  std::vector<float> expected_ctx(backend.attn_ctx.size(), 0.0f);
  backend.attn_ctx = flash_ctx;
  REQUIRE(emel::generator::detail::dispatch_flash_attention(backend, 0, position));
  flash_ctx = backend.attn_ctx;
  expected_ctx = flash_attention_online_reference(backend, 0, position, backend.q);

  for (size_t idx = 0; idx < flash_ctx.size(); ++idx) {
    CHECK(within_flash_online_f16_tolerance(flash_ctx[idx], expected_ctx[idx]));
  }
}

TEST_CASE("generator_detail_flash_dispatch_matches_online_softmax_reference_across_long_reuse") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 12;
  backend.n_head_kv = 12;
  backend.n_rep = 1;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 1024;
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;

  const size_t n_embd =
      static_cast<size_t>(backend.n_head) * static_cast<size_t>(backend.head_dim);
  const size_t kv_dim =
      static_cast<size_t>(backend.n_head_kv) * static_cast<size_t>(backend.head_dim_kv);

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
        const size_t offset = emel::generator::detail::flash_layer_cache_head_position_offset(
            backend, 0, head, position, backend.head_dim_kv) + static_cast<size_t>(dim);
        const double kv_base =
            static_cast<double>((position + 1) * (head + 3) * (dim + 5));
        backend.flash_key_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::cos(kv_base * 0.0078125)));
        backend.flash_value_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(kv_base * 0.01171875)));
      }
    }

    std::vector<float> flash_ctx(backend.attn_ctx.size(), 0.0f);
    std::vector<float> expected_ctx(backend.attn_ctx.size(), 0.0f);
    backend.attn_ctx = flash_ctx;
    REQUIRE(emel::generator::detail::dispatch_flash_attention(backend, 0, position));
    flash_ctx = backend.attn_ctx;
    expected_ctx = flash_attention_online_reference(backend, 0, position, backend.q);

    for (size_t idx = 0; idx < flash_ctx.size(); ++idx) {
      const float diff = std::fabs(flash_ctx[idx] - expected_ctx[idx]);
      if (diff > max_abs) {
        max_abs = diff;
      }
      if (first_position < 0 &&
          !within_flash_online_f16_tolerance(flash_ctx[idx], expected_ctx[idx])) {
        first_position = position;
        first_index = idx;
        first_flash = flash_ctx[idx];
        first_expected = expected_ctx[idx];
      }
    }
  }

  INFO("first_position=" << first_position << " first_index=" << first_index
                         << " first_flash=" << first_flash
                         << " first_expected=" << first_expected
                         << " max_abs=" << max_abs);
  CHECK(max_abs <= k_flash_online_f16_abs_tolerance);
}

TEST_CASE("generator_detail_qwen3_generator_applies_per_head_qk_norm_before_rope") {
  auto fixture = std::make_unique<qwen3_runtime_fixture>();
  auto backend = std::make_unique<emel::generator::detail::native_backend>();
  REQUIRE(emel::generator::detail::prepare(*backend, fixture->model) ==
          emel::error::cast(emel::model::loader::error::none));
  REQUIRE(emel::generator::detail::copy_tensor_row(
      *backend->token_embedding.tensor, 0, backend->hidden));

  std::array<float, 4> hidden_after_input_norm = {};
  REQUIRE(emel::generator::detail::rms_norm(
      backend->hidden,
      backend->blocks[0].attention_norm,
      backend->rms_epsilon,
      std::span<float>(hidden_after_input_norm.data(), hidden_after_input_norm.size())));

  std::array<float, 4> expected_q = hidden_after_input_norm;
  std::array<float, 4> expected_k = hidden_after_input_norm;
  constexpr std::array<float, 2> q_norm = {2.0f, 0.5f};
  constexpr std::array<float, 2> k_norm = {1.5f, 0.25f};
  apply_qwen3_headwise_rms_norm(expected_q, q_norm, 2, 2, backend->rms_epsilon);
  apply_qwen3_headwise_rms_norm(expected_k, k_norm, 2, 2, backend->rms_epsilon);
  apply_rope_reference(expected_q, 2, 2, 2, 1, backend->rope_freq_base);
  apply_rope_reference(expected_k, 2, 2, 2, 1, backend->rope_freq_base);

  REQUIRE(emel::generator::detail::run_layer_nonflash(*backend, 0, 1));
  const size_t cache_offset = emel::generator::detail::layer_cache_offset(
      *backend, 0, 1, backend->n_head_kv * backend->head_dim_kv);

  for (size_t idx = 0; idx < expected_q.size(); ++idx) {
    CHECK(backend->q[idx] == doctest::Approx(expected_q[idx]).epsilon(1.0e-5));
    CHECK(backend->q_attn[idx] ==
          doctest::Approx(round_fp16_value(expected_q[idx])).epsilon(1.0e-5));
    CHECK(backend->k[idx] == doctest::Approx(expected_k[idx]).epsilon(1.0e-5));
    CHECK(emel::generator::detail::quant::fp16_to_fp32(backend->key_cache[cache_offset + idx]) ==
          doctest::Approx(round_fp16_value(expected_k[idx])).epsilon(1.0e-5));
  }
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk4_batches_explicit_q8_gemm") {
  auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  REQUIRE(emel::generator::detail::run_kernel_nonflash_prefill_chunk4_packed_q8_0(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens == chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 8u);
  CHECK(fixture->backend.kernel_dispatch_calls == 8u);
  CHECK(std::all_of(
      fixture->backend.bound_logits.begin(),
      fixture->backend.bound_logits.end(),
      [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_prefill_chunk4_packed_q8_0(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE(
    "generator_detail_run_kernel_nonflash_prefill_chunk4_preselected_argmax_batches_explicit_q8_gemm") {
  auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  REQUIRE(
      emel::generator::detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens == chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 7u);
  CHECK(fixture->backend.kernel_dispatch_calls == 8u);
  CHECK(fixture->selected_token == 0);
  CHECK(fixture->selected_score == doctest::Approx(0.0f));
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::generator::detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0(
          fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE("generator_detail_run_kernel_flash_prefill_chunk4_batches_explicit_q8_gemm") {
  auto fixture = std::make_unique<chunk4_prefill_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  REQUIRE(emel::generator::detail::run_kernel_flash_prefill_chunk4_packed_q8_0(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens == chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 8u);
  CHECK(fixture->backend.flash_attention_dispatch_calls ==
        chunk4_prefill_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.kernel_dispatch_calls == 12u);
  CHECK(std::all_of(
      fixture->backend.bound_logits.begin(),
      fixture->backend.bound_logits.end(),
      [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(emel::generator::detail::run_kernel_flash_prefill_chunk4_packed_q8_0(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk4_batches_hybrid_q8_k_x4_gemm") {
  auto fixture = std::make_unique<hybrid_chunk4_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  REQUIRE(emel::generator::detail::run_kernel_nonflash_prefill_chunk4_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens == hybrid_chunk4_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 0u);
  CHECK(fixture->backend.kernel_dispatch_calls == 13u);
  CHECK(std::all_of(
      fixture->backend.bound_logits.begin(),
      fixture->backend.bound_logits.end(),
      [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_prefill_chunk4_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE("generator_detail_run_kernel_nonflash_prefill_chunk8_batches_hybrid_q8_k_x8_gemm") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  REQUIRE(emel::generator::detail::run_kernel_nonflash_prefill_chunk8_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens == hybrid_chunk8_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 0u);
  CHECK(fixture->backend.kernel_dispatch_calls == 13u);
  CHECK(std::all_of(
      fixture->backend.bound_logits.begin(),
      fixture->backend.bound_logits.end(),
      [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_prefill_chunk8_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE("generator_detail_run_kernel_flash_prefill_chunk8_batches_hybrid_q8_k_x8_gemm") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  REQUIRE(emel::generator::detail::run_kernel_flash_prefill_chunk8_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture->backend.kv_cache_tokens == hybrid_chunk8_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.packed_q8_0_dispatch_calls == 0u);
  CHECK(fixture->backend.flash_attention_dispatch_calls ==
        hybrid_chunk8_q8_runtime_fixture::k_prompt_tokens);
  CHECK(fixture->backend.kernel_dispatch_calls == 21u);
  CHECK(std::all_of(
      fixture->backend.bound_logits.begin(),
      fixture->backend.bound_logits.end(),
      [](const float value) { return value == 0.0f; }));
#else
  int32_t err = -1;
  CHECK_FALSE(emel::generator::detail::run_kernel_flash_prefill_chunk8_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE(
    "generator_detail_run_kernel_nonflash_prefill_chunk8_preselected_argmax_rejects_without_direct_argmax_support") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(fixture->backend));
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_prefill_chunk8_preselected_argmax_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#else
  int32_t err = -1;
  CHECK_FALSE(
      emel::generator::detail::run_kernel_nonflash_prefill_chunk8_preselected_argmax_q8_k(
          fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}

TEST_CASE(
    "generator_detail_run_kernel_flash_prefill_chunk8_preselected_argmax_rejects_without_direct_argmax_support") {
  auto fixture = std::make_unique<hybrid_chunk8_q8_runtime_fixture>();
  REQUIRE(fixture->ready);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  int32_t err = -1;
  REQUIRE(emel::generator::detail::bind_inputs(fixture->request, &err));
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(fixture->backend));
  CHECK_FALSE(emel::generator::detail::run_kernel_flash_prefill_chunk8_preselected_argmax_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#else
  int32_t err = -1;
  CHECK_FALSE(emel::generator::detail::run_kernel_flash_prefill_chunk8_preselected_argmax_q8_k(
      fixture->request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
#endif
}
