#include <algorithm>
#include <array>
#include <doctest/doctest.h>
#include <memory>
#include <span>
#include <vector>

// Component-private SML rule regression tests.
// Maintained generator behavior proof lives in lifecycle, parity, and benchmark
// tests that drive public generator events. This file intentionally covers
// private action/guard predicates so rule regressions are caught without making
// it a maintained end-to-end behavior source.

#include "emel/kernel/detail.hpp"
#include "emel/model/data.hpp"
#include "emel/text/generator/actions.hpp"
#include "emel/text/generator/guards.hpp"
#include "emel/text/generator/initializer/guards.hpp"
#include "emel/text/generator/matmul/sm.hpp"
#include "emel/text/generator/prefill/actions.hpp"
#include "emel/text/generator/prefill/guards.hpp"
#include "generator_test_policies.hpp"

namespace {

struct callback_tracker {
  bool initialize_done_called = false;
  bool initialize_error_called = false;
  bool generate_done_called = false;
  bool generate_error_called = false;
  emel::error::type err = emel::error::cast(emel::text::generator::error::none);
  int32_t tokens_generated = -1;
  size_t output_length = 0;
};

void on_initialize_done(
    void *owner, const emel::text::generator::events::initialize_done &) {
  static_cast<callback_tracker *>(owner)->initialize_done_called = true;
}

void on_initialize_error(
    void *owner, const emel::text::generator::events::initialize_error &ev) {
  auto *tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_error_called = true;
  tracker->err = ev.err;
}

void on_generate_done(
    void *owner, const emel::text::generator::events::generation_done &ev) {
  auto *tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_done_called = true;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

void on_generate_error(
    void *owner, const emel::text::generator::events::generation_error &ev) {
  auto *tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_error_called = true;
  tracker->err = ev.err;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void *,
                             const emel::text::tokenizer::event::bind &) {
  return true;
}

bool tokenizer_tokenize_dispatch(
    void *, const emel::text::tokenizer::event::tokenize &) {
  return true;
}

emel::error::type sampler_passthrough(int32_t &, float &, int32_t &,
                                      int32_t &) {
  return emel::error::cast(emel::logits::sampler::error::none);
}

int dummy_tokenizer_actor = 4;
std::array<emel::logits::sampler::fn, 1> dummy_samplers = {
    emel::logits::sampler::fn::from<sampler_passthrough>(),
};
constexpr std::array<emel::text::formatter::chat_message, 1>
    k_generate_messages = {
        emel::text::formatter::chat_message{
            .role = "user",
            .content = "hello",
        },
};

emel::model::data &test_model() {
  static auto *model = []() {
    auto *created = new emel::model::data{};
    created->params.n_ctx = 8;
    created->vocab_data.eos_id = 7;
    created->vocab_data.eot_id = 8;
    return created;
  }();
  return *model;
}

constexpr bool host_supports_chunk4_prefill_q8_route() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    (defined(__ARM_FEATURE_DOTPROD) || defined(__ARM_FEATURE_MATMUL_INT8))
  return true;
#else
  return false;
#endif
}

constexpr bool host_supports_chunk8_prefill_q8_route() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  return true;
#else
  return false;
#endif
}

constexpr uint8_t chunk4_test_dtype() noexcept {
#if defined(__ARM_FEATURE_MATMUL_INT8)
  return emel::kernel::detail::dtype_q4_k_x8_bl8;
#elif defined(__ARM_FEATURE_DOTPROD)
  return emel::kernel::detail::dtype_q4_k_x8_bl4;
#else
  return emel::kernel::detail::dtype_q4_k_x8_bl4;
#endif
}

struct chunk4_planning_backend_fixture {
  emel::text::generator::action::context context = {};
  emel::model::data::tensor_record matrix = {};

  chunk4_planning_backend_fixture() {
    auto &backend = context.compute.backend;
    backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    backend.routes = emel::text::generator::test::k_generation_route_policy;
    backend.n_layer = 1;
    backend.n_embd = 4;
    backend.n_head = 1;
    backend.head_dim = 1;
    backend.n_head_kv = 1;
    backend.head_dim_kv = 1;
    backend.shortconv_state_size = 0;

    matrix.type = static_cast<int32_t>(chunk4_test_dtype());
    matrix.n_dims = 2;
    matrix.dims[0] = 4;
    matrix.dims[1] = 4;

    auto &block = backend.blocks.emplace_back();
    block.residual_route =
        emel::model::transformer::generation_residual_route::attention;
    block.attention_q = {&matrix, 4, 4};
    block.attention_k = {&matrix, 4, 4};
    block.attention_v = {&matrix, 4, 4};
    block.attention_output = {&matrix, 4, 4};
    block.feed_forward_gate = {&matrix, 4, 4};
    block.feed_forward_down = {&matrix, 4, 4};
    block.feed_forward_up = {&matrix, 4, 4};

    backend.q8_input_storage.resize(1u);
    backend.q8_input_chunk4_storage.resize(1u);
    backend.q8_input_chunk8_storage.resize(2u);
    backend.hidden_chunk4.resize(16u);
    backend.hidden_chunk8.resize(32u);
    backend.norm_chunk4.resize(16u);
    backend.norm_chunk8.resize(32u);
    backend.projected_chunk4.resize(16u);
    backend.projected_chunk8.resize(32u);
    backend.attn_ctx_chunk4.resize(4u);
    backend.attn_ctx_chunk8.resize(8u);
    backend.q_chunk4.resize(4u);
    backend.q_chunk8.resize(8u);
    backend.k_chunk4.resize(4u);
    backend.k_chunk8.resize(8u);
    backend.v_chunk4.resize(4u);
    backend.v_chunk8.resize(8u);
    backend.gate_chunk4.resize(16u);
    backend.gate_chunk8.resize(32u);
    backend.up_chunk4.resize(16u);
    backend.up_chunk8.resize(32u);
    backend.ffn_hidden_chunk4.resize(16u);
    backend.ffn_hidden_chunk8.resize(32u);
  }
};

struct native_quantized_route_fixture {
  emel::text::generator::action::context context = {};
  emel::model::data::tensor_record body_tensor = {};
  emel::model::data::tensor_record logits_tensor = {};

  native_quantized_route_fixture() {
    auto &backend = context.compute.backend;
    backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    backend.routes = emel::text::generator::test::k_generation_route_policy;
    backend.n_layer = 1;
    backend.n_embd = static_cast<int32_t>(emel::kernel::detail::quant::QK_K);
    backend.n_head = 1;
    backend.head_dim = 1;
    backend.n_head_kv = 1;
    backend.head_dim_kv = 1;

    body_tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
    body_tensor.n_dims = 2;
    body_tensor.dims[0] = backend.n_embd;
    body_tensor.dims[1] = backend.n_embd;

    logits_tensor.type = emel::kernel::detail::dtype_q6_k_x8_q8_prepared;
    logits_tensor.n_dims = 2;
    logits_tensor.dims[0] = backend.n_embd;
    logits_tensor.dims[1] = backend.n_embd;

    auto &block = backend.blocks.emplace_back();
    block.residual_route =
        emel::model::transformer::generation_residual_route::attention;
    block.attention_q = {&body_tensor, backend.n_embd, backend.n_embd};
    block.attention_k = {&body_tensor, backend.n_embd, backend.n_embd};
    block.attention_v = {&body_tensor, backend.n_embd, backend.n_embd};
    block.attention_output = {&body_tensor, backend.n_embd, backend.n_embd};
    block.feed_forward_gate = {&body_tensor, backend.n_embd, backend.n_embd};
    block.feed_forward_down = {&body_tensor, backend.n_embd, backend.n_embd};
    block.feed_forward_up = {&body_tensor, backend.n_embd, backend.n_embd};
    backend.output = {&logits_tensor, backend.n_embd, backend.n_embd};
    backend.q8_input_storage.resize(1u);
  }
};

struct compute_guard_fixture {
  emel::text::generator::action::context context = {};
  emel::text::generator::matmul::lane_pool<7u, 128u, 1048576u> parallel_matmul_lanes = {};
  emel::text::generator::matmul::execution_policy matmul_policy =
      emel::text::generator::matmul::make_auto_execution_policy(
          parallel_matmul_lanes);
  emel::text::generator::matmul::sm matmul_actor{matmul_policy};
  emel::model::data model = {};
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 1>
      reserve_tensors = {};
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 1>
      prefill_tensors = {};
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 1>
      decode_tensors = {};
  emel::graph::processor::event::lifecycle_phase prefill_phase = {};
  emel::graph::processor::event::lifecycle_phase decode_phase = {};
  emel::graph::processor::event::lifecycle_manifest reserve_lifecycle = {};
  std::array<float, 4> logits = {};
  emel::model::data::tensor_record argmax_tensor = {};

  compute_guard_fixture() {
    context.compute.backend_ready = true;
    context.limits.prompt_capacity = 4;
    context.buffers.vocab_size = 4;
    context.buffers.logits = std::make_unique<float[]>(4);

    reserve_lifecycle.tensors = reserve_tensors.data();
    reserve_lifecycle.tensor_count =
        static_cast<int32_t>(reserve_tensors.size());
    context.state.graph_reservation.lifecycle = &reserve_lifecycle;
    context.state.graph_reservation.node_count = 1u;
    context.state.graph_reservation.tensor_count = 1u;

    auto &backend = context.compute.backend;
    backend.matmul_actor = &matmul_actor;
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
    // Coherent captured snapshot for the single maintained sequence: one
    // 8-token block covering the 1-token workload the guard tests drive.
    context.state.memory_snapshot.max_sequences = 1;
    context.state.memory_snapshot.block_tokens = 8;
    context.state.memory_snapshot.sequence_active[0] = 1;
    context.state.memory_snapshot.sequence_length_values[0] = 1;
    context.state.memory_snapshot.sequence_kv_block_count[0] = 1;
    context.state.memory_snapshot.sequence_kv_blocks[0][0] = 0;
    backend.blocks.resize(1u);
    backend.bound_tokens.resize(4u);
    backend.bound_positions.resize(4u);
    backend.bound_logits.resize(4u);
    backend.token_embedding.rows = 4;
    backend.topology.execution = &backend.execution;
    backend.prefill_plan.graph = &backend.topology;
    backend.prefill_plan.kind =
        emel::text::generator::detail::step_kind::prefill;
    backend.prefill_plan.expected_outputs = 1;
    backend.prefill_plan.max_step_tokens = 4;
    backend.decode_plan.graph = &backend.topology;
    backend.decode_plan.kind = emel::text::generator::detail::step_kind::decode;
    backend.decode_plan.expected_outputs = 1;
    backend.decode_plan.max_step_tokens = 1;
    backend.prefill_lifecycle.tensors = prefill_tensors.data();
    backend.prefill_lifecycle.tensor_count =
        static_cast<int32_t>(prefill_tensors.size());
    backend.prefill_lifecycle.phase = &prefill_phase;
    backend.decode_lifecycle.tensors = decode_tensors.data();
    backend.decode_lifecycle.tensor_count =
        static_cast<int32_t>(decode_tensors.size());
    backend.decode_lifecycle.phase = &decode_phase;
  }

  emel::text::generator::event::generate_run
  make_generate_run(emel::text::generator::event::generate &request,
                    emel::text::generator::event::generate_ctx &runtime) {
    runtime.prompt_token_count = 1;
    runtime.prefill_step_size = 1;
    runtime.plan_outputs = 1;
    runtime.selected_token = 0;
    runtime.kv_tokens = 0;
    context.compute.backend.kv_cache_tokens = 0;
    return {request, runtime};
  }

  void enable_direct_preselected_argmax() {
    context.state.selection_mode =
        emel::text::generator::selection_mode::preselected_argmax;
    context.compute.backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
    context.compute.backend.q8_input_storage.resize(1u);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    argmax_tensor.type = emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
#else
    argmax_tensor.type = emel::kernel::detail::dtype_q6_k_x8;
#endif
    argmax_tensor.n_dims = 2;
    argmax_tensor.dims[0] = 4;
    argmax_tensor.dims[1] = 4;
    context.compute.backend.output_argmax = {&argmax_tensor, 4, 4};
  }
};

emel::text::generator::event::initialize make_initialize_request(
    callback_tracker *tracker, emel::error::type *error_out,
    const emel::text::generator::selection_mode selection_mode =
        emel::text::generator::selection_mode::sample_logits) {
  const std::span<emel::logits::sampler::fn> sampler_span =
      selection_mode == emel::text::generator::selection_mode::sample_logits
          ? std::span<emel::logits::sampler::fn>{dummy_samplers}
          : std::span<emel::logits::sampler::fn>{};
  emel::text::generator::event::initialize request{
      &dummy_tokenizer_actor,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      sampler_span,
  };
  request.selection_mode = selection_mode;
  request.max_prompt_tokens = 8;
  request.max_generated_tokens = 4;
  request.max_blocks = 2;
  request.block_tokens = 4;
  request.error_out = error_out;
  request.on_done =
      tracker == nullptr
          ? emel::callback<void(
                const emel::text::generator::events::initialize_done &)>{}
          : emel::callback<void(
                const emel::text::generator::events::initialize_done &)>(
                tracker, on_initialize_done);
  request.on_error =
      tracker == nullptr
          ? emel::callback<void(
                const emel::text::generator::events::initialize_error &)>{}
          : emel::callback<void(
                const emel::text::generator::events::initialize_error &)>(
                tracker, on_initialize_error);
  return request;
}

emel::text::generator::event::generate
make_generate_request(callback_tracker *tracker, emel::error::type *error_out,
                      size_t &output_length_out) {
  static char output_storage[16] = {};
  emel::text::generator::event::generate request{
      std::span<const emel::text::formatter::chat_message>{k_generate_messages},
      2,
      std::span<char>{output_storage},
      output_length_out,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.error_out = error_out;
  request.on_done =
      tracker == nullptr
          ? emel::callback<void(
                const emel::text::generator::events::generation_done &)>{}
          : emel::callback<void(
                const emel::text::generator::events::generation_done &)>(
                tracker, on_generate_done);
  request.on_error =
      tracker == nullptr
          ? emel::callback<void(
                const emel::text::generator::events::generation_error &)>{}
          : emel::callback<void(
                const emel::text::generator::events::generation_error &)>(
                tracker, on_generate_error);
  return request;
}

} // namespace

TEST_CASE("generator initialize dispatch actions cover channel variants") {
  emel::text::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);

  auto with_both = make_initialize_request(&tracker, &error_out);
  emel::text::generator::event::initialize_ctx init_ctx{};
  emel::text::generator::event::initialize_run init_run{with_both, init_ctx};
  emel::text::generator::action::
      dispatch_initialize_done_with_callback_and_error_out(init_run, context);
  CHECK(tracker.initialize_done_called);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::none));

  tracker = {};
  error_out = emel::error::cast(emel::text::generator::error::none);
  auto callback_only = make_initialize_request(&tracker, nullptr);
  emel::text::generator::event::initialize_run callback_only_run{callback_only,
                                                                 init_ctx};
  emel::text::generator::action::dispatch_initialize_done_with_callback_only(
      callback_only_run, context);
  CHECK(tracker.initialize_done_called);

  tracker = {};
  error_out = emel::error::cast(emel::text::generator::error::backend);
  auto error_out_only = make_initialize_request(nullptr, &error_out);
  emel::text::generator::event::initialize_run error_out_only_run{
      error_out_only, init_ctx};
  emel::text::generator::action::dispatch_initialize_done_with_error_out_only(
      error_out_only_run, context);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::none));

  emel::text::generator::action::dispatch_initialize_done_without_channels(
      error_out_only_run, context);

  tracker = {};
  error_out = emel::error::cast(emel::text::generator::error::none);
  init_ctx.err = emel::error::cast(emel::text::generator::error::backend);
  emel::text::generator::action::
      dispatch_initialize_error_with_callback_and_error_out(init_run, context);
  CHECK(tracker.initialize_error_called);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::backend));

  tracker = {};
  emel::text::generator::action::dispatch_initialize_error_with_callback_only(
      callback_only_run, context);
  CHECK(tracker.initialize_error_called);

  error_out = emel::error::cast(emel::text::generator::error::none);
  emel::text::generator::action::dispatch_initialize_error_with_error_out_only(
      error_out_only_run, context);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::backend));

  emel::text::generator::action::dispatch_initialize_error_without_channels(
      error_out_only_run, context);
}

TEST_CASE("generator generate dispatch actions cover channel variants") {
  emel::text::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto with_both =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx gen_ctx{};
  gen_ctx.tokens_generated = 2;
  gen_ctx.output_length = 10;
  gen_ctx.err = emel::error::cast(emel::text::generator::error::backend);
  emel::text::generator::event::generate_run gen_run{with_both, gen_ctx};
  emel::text::generator::action::
      dispatch_generate_done_with_callback_and_error_out(gen_run, context);
  CHECK(tracker.generate_done_called);
  CHECK(tracker.tokens_generated == 2);
  CHECK(tracker.output_length == 10);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::none));

  tracker = {};
  auto callback_only =
      make_generate_request(&tracker, nullptr, output_length_out);
  emel::text::generator::event::generate_run callback_only_run{callback_only,
                                                               gen_ctx};
  emel::text::generator::action::dispatch_generate_done_with_callback_only(
      callback_only_run, context);
  CHECK(tracker.generate_done_called);

  tracker = {};
  error_out = emel::error::cast(emel::text::generator::error::backend);
  auto error_out_only =
      make_generate_request(nullptr, &error_out, output_length_out);
  emel::text::generator::event::generate_run error_out_only_run{error_out_only,
                                                                gen_ctx};
  emel::text::generator::action::dispatch_generate_done_with_error_out_only(
      error_out_only_run, context);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::none));

  emel::text::generator::action::dispatch_generate_done_without_channels(
      error_out_only_run, context);

  tracker = {};
  error_out = emel::error::cast(emel::text::generator::error::none);
  gen_ctx.err = emel::error::cast(emel::text::generator::error::backend);
  emel::text::generator::action::
      dispatch_generate_error_with_callback_and_error_out(gen_run, context);
  CHECK(tracker.generate_error_called);
  CHECK(tracker.tokens_generated == 2);
  CHECK(tracker.output_length == 10);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::backend));

  tracker = {};
  emel::text::generator::action::dispatch_generate_error_with_callback_only(
      callback_only_run, context);
  CHECK(tracker.generate_error_called);

  error_out = emel::error::cast(emel::text::generator::error::none);
  emel::text::generator::action::dispatch_generate_error_with_error_out_only(
      error_out_only_run, context);
  CHECK(error_out == emel::error::cast(emel::text::generator::error::backend));

  emel::text::generator::action::dispatch_generate_error_without_channels(
      error_out_only_run, context);
}

TEST_CASE("generator request_planning_scalar accepts multi-token "
          "single-sequence prompt metadata") {
  emel::text::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 2;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};
  context.buffers.prompt_tokens[0] = 11;
  context.buffers.prompt_tokens[1] = 13;

  emel::text::generator::action::request_planning_scalar(generate_run, context);

  CHECK(generate_ctx.phase_accepted);
  CHECK(generate_ctx.phase_code == 0);
  CHECK(generate_ctx.prefill_step_size == 1);
  CHECK(generate_ctx.plan_step_count == 2);
  CHECK(generate_ctx.plan_outputs >= 0);
}

TEST_CASE("generator planning guards select explicit chunk4 prefill routing") {
  chunk4_planning_backend_fixture fixture{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count =
      emel::text::generator::detail::k_prefill_q8_chunk_rows;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  if (host_supports_chunk4_prefill_q8_route()) {
    CHECK(emel::text::generator::guard::planning_uses_chunk4_prefill{}(
        generate_run, fixture.context));
    CHECK_FALSE(emel::text::generator::guard::planning_uses_scalar_prefill{}(
        generate_run, fixture.context));
  } else {
    CHECK_FALSE(emel::text::generator::guard::planning_uses_chunk4_prefill{}(
        generate_run, fixture.context));
    CHECK(emel::text::generator::guard::planning_uses_scalar_prefill{}(
        generate_run, fixture.context));
  }
}

TEST_CASE("generator planning guards keep sub-chunk prompts on scalar route") {
  chunk4_planning_backend_fixture fixture{};
  auto &backend = fixture.context.compute.backend;
  backend.routes.prefill_chunk4_min_tokens = 0;
  backend.routes.prefill_chunk8_min_tokens = 0;

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count =
      emel::text::generator::detail::k_prefill_q8_chunk_rows - 1;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  CHECK_FALSE(emel::text::generator::guard::planning_uses_chunk8_prefill{}(
      generate_run, fixture.context));
  CHECK_FALSE(emel::text::generator::guard::planning_uses_chunk4_prefill{}(
      generate_run, fixture.context));
  CHECK(emel::text::generator::guard::planning_uses_scalar_prefill{}(
      generate_run, fixture.context));
}

TEST_CASE("generator planning guards select explicit chunk8 prefill routing") {
  chunk4_planning_backend_fixture fixture{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count =
      emel::text::generator::detail::k_prefill_q8_chunk8_rows;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  if (host_supports_chunk8_prefill_q8_route()) {
    CHECK(emel::text::generator::guard::planning_uses_chunk8_prefill{}(
        generate_run, fixture.context));
    CHECK_FALSE(emel::text::generator::guard::planning_uses_chunk4_prefill{}(
        generate_run, fixture.context));
    CHECK_FALSE(emel::text::generator::guard::planning_uses_scalar_prefill{}(
        generate_run, fixture.context));
  } else if (host_supports_chunk4_prefill_q8_route()) {
    CHECK_FALSE(emel::text::generator::guard::planning_uses_chunk8_prefill{}(
        generate_run, fixture.context));
    CHECK(emel::text::generator::guard::planning_uses_chunk4_prefill{}(
        generate_run, fixture.context));
  } else {
    CHECK_FALSE(emel::text::generator::guard::planning_uses_chunk8_prefill{}(
        generate_run, fixture.context));
    CHECK(emel::text::generator::guard::planning_uses_scalar_prefill{}(
        generate_run, fixture.context));
  }
}

TEST_CASE(
    "generator request_planning_chunk4 batches prompt metadata explicitly") {
  emel::text::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 8;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};
  for (int32_t idx = 0; idx < generate_ctx.prompt_token_count; ++idx) {
    context.buffers.prompt_tokens[static_cast<size_t>(idx)] = idx;
  }

  emel::text::generator::action::request_planning_chunk4(generate_run, context);

  CHECK(generate_ctx.phase_accepted);
  CHECK(generate_ctx.phase_code == 0);
  CHECK(generate_ctx.prefill_step_size ==
        emel::text::generator::detail::k_prefill_q8_chunk_rows);
  CHECK(generate_ctx.plan_step_count == 2);
  CHECK(generate_ctx.plan_outputs >= 0);
}

TEST_CASE(
    "generator request_planning_chunk8 batches prompt metadata explicitly") {
  emel::text::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::backend);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 16;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};
  for (int32_t idx = 0; idx < generate_ctx.prompt_token_count; ++idx) {
    context.buffers.prompt_tokens[static_cast<size_t>(idx)] = idx;
  }

  emel::text::generator::action::request_planning_chunk8(generate_run, context);

  CHECK(generate_ctx.phase_accepted);
  CHECK(generate_ctx.phase_code == 0);
  CHECK(generate_ctx.prefill_step_size ==
        emel::text::generator::detail::k_prefill_q8_chunk8_rows);
  CHECK(generate_ctx.plan_step_count == 2);
  CHECK(generate_ctx.plan_outputs >= 0);
}

TEST_CASE("generator structured message request and channel guards classify "
          "callback variants") {
  emel::text::generator::action::context context{};
  emel::text::conditioner::sm conditioner{};
  context.model = &test_model();
  context.conditioner = &conditioner;
  context.format_prompt = emel::text::formatter::format_raw;
  context.limits.decode_capacity = 4;
  context.state.sequence_live = true;

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.target_tokens = 2;
  generate_ctx.tokens_generated = 0;
  generate_ctx.render_status = emel::text::renderer::sequence_status::running;
  generate_ctx.selected_token = 1;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  CHECK(generate.messages.size() == 1u);
  CHECK(generate.messages[0].role == "user");
  CHECK(generate.messages[0].content == "hello");
  CHECK(generate.add_generation_prompt);
  CHECK_FALSE(generate.enable_thinking);
  CHECK(emel::text::generator::guard::valid_generate{}(generate_run, context));
  CHECK(emel::text::generator::guard::valid_generate_with_reset{}(generate_run,
                                                                  context));
  CHECK_FALSE(emel::text::generator::guard::valid_generate_without_reset{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::sequence_needs_reset{}(generate_run,
                                                             context));
  CHECK_FALSE(
      emel::text::generator::guard::sequence_is_clear{}(generate_run, context));
  CHECK(emel::text::generator::guard::generate_done_callback_with_error_out{}(
      generate_run, context));
  CHECK_FALSE(
      emel::text::generator::guard::generate_no_done_callback_with_error_out{}(
          generate_run, context));
  CHECK_FALSE(
      emel::text::generator::guard::generate_done_callback_without_error_out{}(
          generate_run, context));
  CHECK(emel::text::generator::guard::generate_error_callback_with_error_out{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_should_continue{}(generate_run,
                                                               context));

  context.state.sequence_live = false;
  CHECK(emel::text::generator::guard::valid_generate_without_reset{}(
      generate_run, context));
  CHECK_FALSE(emel::text::generator::guard::valid_generate_with_reset{}(
      generate_run, context));

  generate.on_done = {};
  generate.on_error = {};
  generate.error_out = nullptr;
  CHECK(
      emel::text::generator::guard::
          generate_no_done_callback_without_error_out{}(generate_run, context));
  CHECK(emel::text::generator::guard::
            generate_no_error_callback_without_error_out{}(generate_run,
                                                           context));

  generate.max_tokens = 0;
  CHECK(
      emel::text::generator::guard::invalid_generate{}(generate_run, context));
  generate.max_tokens = 2;
  std::array<char, 16> output_buffer = {};
  generate.messages = {};
  CHECK(
      emel::text::generator::guard::invalid_generate{}(generate_run, context));
  generate.messages = std::span<const emel::text::formatter::chat_message>{
      static_cast<const emel::text::formatter::chat_message *>(nullptr), 1u};
  CHECK(
      emel::text::generator::guard::invalid_generate{}(generate_run, context));
  generate.messages =
      std::span<const emel::text::formatter::chat_message>{k_generate_messages};
  generate.output = std::span<char>{static_cast<char *>(nullptr), 1u};
  CHECK(
      emel::text::generator::guard::invalid_generate{}(generate_run, context));
  generate.output = std::span<char>{output_buffer};
  generate_ctx.tokens_generated = 2;
  CHECK(emel::text::generator::guard::decode_complete{}(generate_run, context));
  generate_ctx.tokens_generated = 0;
  generate_ctx.selected_token = test_model().vocab_data.eos_id;
  CHECK(emel::text::generator::guard::decode_complete{}(generate_run, context));

  auto initialize = make_initialize_request(&tracker, &error_out);
  emel::text::generator::event::initialize_ctx initialize_ctx{};
  emel::text::generator::event::initialize_run initialize_run{initialize,
                                                              initialize_ctx};
  CHECK(emel::text::generator::guard::initialize_done_callback_with_error_out{}(
      initialize_run, context));
  CHECK(
      emel::text::generator::guard::initialize_error_callback_with_error_out{}(
          initialize_run, context));
  initialize.on_done = {};
  initialize.on_error = {};
  initialize.error_out = nullptr;
  CHECK(emel::text::generator::guard::
            initialize_no_done_callback_without_error_out{}(initialize_run,
                                                            context));
  CHECK(emel::text::generator::guard::
            initialize_no_error_callback_without_error_out{}(initialize_run,
                                                             context));

  auto preselected_initialize = make_initialize_request(
      &tracker, &error_out,
      emel::text::generator::selection_mode::preselected_argmax);
  emel::text::generator::event::initialize_ctx preselected_ctx{};
  emel::text::generator::event::initialize_run preselected_run{
      preselected_initialize, preselected_ctx};
  context.state.selection_mode =
      emel::text::generator::selection_mode::preselected_argmax;
  emel::text::generator::initializer::action::context initializer_context{
      context};
  const emel::text::generator::initializer::event::run initializer_run{
      preselected_run.request,
      preselected_run.ctx,
  };
  CHECK(emel::text::generator::guard::valid_initialize{}(preselected_run,
                                                         context));
  CHECK(emel::text::generator::initializer::guard::uses_preselected_argmax{}(
      initializer_run, initializer_context));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::uses_materialized_logits{}(
          initializer_run, initializer_context));
}

TEST_CASE("generator phase guards classify invalid and backend errors") {
  emel::text::generator::action::context context{};
  context.model = &test_model();
  context.limits.decode_capacity = 4;

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;

  auto initialize = make_initialize_request(&tracker, &error_out);
  emel::text::generator::event::initialize_ctx initialize_ctx{};
  emel::text::generator::event::initialize_run initialize_run{initialize,
                                                              initialize_ctx};
  emel::text::generator::initializer::action::context initializer_context{
      context};
  const emel::text::generator::initializer::event::run initializer_run{
      initialize_run.request,
      initialize_run.ctx,
  };

  initialize_ctx.phase_accepted = false;
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::text::conditioner::error::invalid_argument);
  CHECK(emel::text::generator::initializer::guard::
            conditioner_bind_invalid_request{}(initializer_run,
                                               initializer_context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::text::conditioner::error::backend);
  CHECK(emel::text::generator::initializer::guard::
            conditioner_bind_backend_error{}(initializer_run,
                                             initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::text::renderer::error::invalid_request));
  CHECK(emel::text::generator::initializer::guard::
            renderer_initialize_invalid_request{}(initializer_run,
                                                  initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::text::renderer::error::backend_error));
  CHECK(emel::text::generator::initializer::guard::
            renderer_initialize_backend_error{}(initializer_run,
                                                initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::invalid_request));
  CHECK(emel::text::generator::initializer::guard::
            memory_reserve_invalid_request{}(initializer_run,
                                             initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::backend_error));
  CHECK(
      emel::text::generator::initializer::guard::memory_reserve_backend_error{}(
          initializer_run, initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::invalid_request));
  CHECK(emel::text::generator::initializer::guard::
            graph_reserve_invalid_request{}(initializer_run,
                                            initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::assembler_failed));
  CHECK(
      emel::text::generator::initializer::guard::graph_reserve_backend_error{}(
          initializer_run, initializer_context));
  initialize_ctx.buffers_ready = true;
  CHECK(emel::text::generator::initializer::guard::sampler_configured{}(
      initializer_run, initializer_context));
  initialize_ctx.buffers_ready = false;
  CHECK(emel::text::generator::initializer::guard::sampler_config_failed{}(
      initializer_run, initializer_context));

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.err = emel::error::cast(emel::text::generator::error::none);
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::invalid_request));
  CHECK(emel::text::generator::guard::reset_sequence_invalid_request{}(
      generate_run, context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::backend_error));
  CHECK(emel::text::generator::guard::reset_sequence_backend_error{}(
      generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::text::conditioner::error::capacity);
  CHECK(emel::text::generator::guard::conditioning_invalid_request{}(
      generate_run, context));
  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  generate_ctx.prompt_token_count = 0;
  CHECK(emel::text::generator::guard::conditioning_backend_error{}(generate_run,
                                                                   context));
  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::batch::planner::error::invalid_step_size));
  CHECK(emel::text::generator::guard::planning_invalid_request{}(generate_run,
                                                                 context));
  generate_ctx.phase_code = static_cast<int32_t>(emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_sequence_metadata));
  CHECK(emel::text::generator::guard::planning_invalid_request{}(generate_run,
                                                                 context));
  CHECK_FALSE(emel::text::generator::guard::planning_backend_error{}(
      generate_run, context));
  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  generate_ctx.plan_step_count = 0;
  CHECK(emel::text::generator::guard::planning_backend_error{}(generate_run,
                                                               context));
  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::processor_failed));
  CHECK(emel::text::generator::guard::decode_compute_backend_error{}(
      generate_run, context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(emel::text::generator::guard::decode_sample_invalid_request{}(
      generate_run, context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::logits::sampler::error::backend_error));
  CHECK(emel::text::generator::guard::decode_sample_backend_error{}(
      generate_run, context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::text::renderer::error::invalid_request));
  CHECK(emel::text::generator::guard::decode_render_invalid_request{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::flush_invalid_request{}(generate_run,
                                                              context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::text::renderer::error::backend_error));
  CHECK(emel::text::generator::guard::decode_render_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::flush_backend_error{}(generate_run,
                                                            context));

  generate_ctx.err = emel::error::cast(emel::text::generator::error::none);
  CHECK(emel::text::generator::guard::generate_result_none{}(generate_run,
                                                             context));
  initialize_ctx.err =
      emel::error::cast(emel::text::generator::error::invalid_request);
  CHECK(emel::text::generator::guard::initialize_result_invalid_request{}(
      initialize_run, context));
  generate_ctx.prefill_contract = emel::text::generator::
      prefill_compute_contract::flash_materialized_scalar;
  CHECK(emel::text::generator::guard::
            prefill_result_ok_with_materialized_logits_contract{}(generate_run,
                                                                  context));
  generate_ctx.prefill_contract =
      emel::text::generator::prefill_compute_contract::flash_preselected_scalar;
  CHECK(emel::text::generator::guard::
            prefill_result_ok_with_preselected_argmax_contract{}(generate_run,
                                                                 context));
  generate_ctx.err =
      emel::error::cast(emel::text::generator::error::invalid_request);
  CHECK(emel::text::generator::guard::prefill_result_invalid_request{}(
      generate_run, context));
  generate_ctx.err = emel::error::cast(emel::text::generator::error::backend);
  CHECK(emel::text::generator::guard::prefill_result_backend_error{}(
      generate_run, context));
  context.prefill_actor = &dummy_tokenizer_actor;
  context.dispatch_prefill =
      [](void *, const emel::text::generator::prefill::event::run &) {
        return true;
      };
  CHECK(emel::text::generator::guard::prefill_dispatch_available{}(generate_run,
                                                                   context));
  context.prefill_actor = nullptr;
  context.dispatch_prefill = nullptr;
  CHECK(emel::text::generator::guard::prefill_dispatch_unavailable{}(
      generate_run, context));
  context.buffers.vocab_size = 4;
  static float logits[4] = {1.0f, 2.0f, 0.5f, -1.0f};
  context.buffers.logits.reset();
  context.buffers.logits = std::make_unique<float[]>(4);
  std::copy(std::begin(logits), std::end(logits), context.buffers.logits.get());
  CHECK(emel::text::generator::guard::decode_argmax_ready{}(generate_run,
                                                            context));
  context.buffers.logits.reset();
  CHECK(emel::text::generator::guard::decode_argmax_invalid_request{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::generate_result_backend{}(generate_run,
                                                                context));
}

TEST_CASE("generator guard detail predicates cover negative branch cases") {
  namespace guard_detail = emel::text::generator::guard::detail;
  using prefill_contract = emel::text::generator::prefill_compute_contract;

  CHECK(guard_detail::conditioner_invalid_code(guard_detail::conditioner_code(
      emel::text::conditioner::error::model_invalid)));
  CHECK(guard_detail::conditioner_invalid_code(guard_detail::conditioner_code(
      emel::text::conditioner::error::capacity)));
  CHECK_FALSE(guard_detail::conditioner_invalid_code(-77));
  CHECK(guard_detail::conditioner_backend_code(guard_detail::conditioner_code(
      emel::text::conditioner::error::untracked)));
  CHECK_FALSE(guard_detail::conditioner_backend_code(-77));
  CHECK(guard_detail::renderer_invalid_code(
      guard_detail::renderer_code(emel::text::renderer::error::model_invalid)));
  CHECK_FALSE(guard_detail::renderer_invalid_code(-77));
  CHECK(guard_detail::renderer_backend_code(guard_detail::renderer_code(
      emel::text::renderer::error::internal_error)));
  CHECK(guard_detail::renderer_backend_code(
      guard_detail::renderer_code(emel::text::renderer::error::untracked)));
  CHECK_FALSE(guard_detail::renderer_backend_code(-77));
  CHECK(guard_detail::memory_backend_code(
      guard_detail::memory_code(emel::memory::hybrid::error::internal_error)));
  CHECK(guard_detail::memory_backend_code(
      guard_detail::memory_code(emel::memory::hybrid::error::out_of_memory)));
  CHECK(guard_detail::memory_backend_code(
      guard_detail::memory_code(emel::memory::hybrid::error::untracked)));
  CHECK_FALSE(guard_detail::memory_backend_code(-77));
  CHECK(guard_detail::graph_backend_code(
      guard_detail::graph_code(emel::graph::error::busy)));
  CHECK(guard_detail::graph_backend_code(
      guard_detail::graph_code(emel::graph::error::internal_error)));
  CHECK(guard_detail::graph_backend_code(
      guard_detail::graph_code(emel::graph::error::untracked)));
  CHECK_FALSE(guard_detail::graph_backend_code(-77));
  CHECK(guard_detail::sampler_backend_code(guard_detail::sampler_code(
      emel::logits::sampler::error::internal_error)));
  CHECK(guard_detail::sampler_backend_code(
      guard_detail::sampler_code(emel::logits::sampler::error::untracked)));
  CHECK_FALSE(guard_detail::sampler_backend_code(-77));

  CHECK(guard_detail::planner_invalid_code(
      static_cast<int32_t>(emel::batch::planner::error::invalid_token_data)));
  CHECK(guard_detail::planner_invalid_code(
      static_cast<int32_t>(emel::batch::planner::error::missing_mode)));
  CHECK(guard_detail::planner_invalid_code(
      static_cast<int32_t>(emel::batch::planner::error::unsupported_layout)));
  CHECK_FALSE(guard_detail::planner_invalid_code(0));
  CHECK(guard_detail::planner_backend_code(
      static_cast<int32_t>(emel::batch::planner::error::algorithm_failed)));
  CHECK(guard_detail::planner_backend_code(
      static_cast<int32_t>(emel::batch::planner::error::untracked)));
  CHECK_FALSE(guard_detail::planner_backend_code(0));

  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::flash_materialized_chunk8_q8_k));
  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::flash_materialized_chunk4_packed_q8_0));
  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::flash_materialized_chunk4_q8_k));
  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::nonflash_materialized_scalar));
  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::nonflash_materialized_chunk8_q8_k));
  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::nonflash_materialized_chunk4_packed_q8_0));
  CHECK(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::nonflash_materialized_chunk4_q8_k));
  CHECK_FALSE(guard_detail::prefill_contract_uses_materialized_logits(
      prefill_contract::flash_preselected_scalar));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::flash_preselected_chunk8_q8_k));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::flash_preselected_chunk4_packed_q8_0));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::flash_preselected_chunk4_q8_k));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::nonflash_preselected_scalar));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::nonflash_preselected_chunk8_q8_k));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::nonflash_preselected_chunk4_packed_q8_0));
  CHECK(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::nonflash_preselected_chunk4_q8_k));
  CHECK_FALSE(guard_detail::prefill_contract_uses_preselected_argmax(
      prefill_contract::flash_materialized_scalar));

  auto fixture = std::make_unique<compute_guard_fixture>();
  auto &backend = fixture->context.compute.backend;
  CHECK(guard_detail::guard_compute_backend_shape_ready(backend));
  backend.n_embd = 0;
  CHECK_FALSE(guard_detail::guard_compute_backend_shape_ready(backend));
  backend.n_embd = 4;
  backend.n_head = 0;
  CHECK_FALSE(guard_detail::guard_compute_backend_shape_ready(backend));
  backend.n_head = 1;
  backend.n_vocab = 0;
  CHECK_FALSE(guard_detail::guard_compute_backend_shape_ready(backend));
  backend.n_vocab = 4;
  backend.blocks.clear();
  CHECK_FALSE(guard_detail::guard_compute_backend_shape_ready(backend));
  backend.blocks.resize(1u);
  CHECK(guard_detail::guard_compute_backend_shape_ready(backend));
  CHECK(guard_detail::guard_compute_backend_ready(fixture->context));
  fixture->context.compute.backend_ready = false;
  CHECK_FALSE(guard_detail::guard_compute_backend_ready(fixture->context));
  fixture->context.compute.backend_ready = true;
  auto *matmul_actor = backend.matmul_actor;
  backend.matmul_actor = nullptr;
  CHECK_FALSE(guard_detail::guard_compute_backend_ready(fixture->context));
  backend.matmul_actor = matmul_actor;

  CHECK(guard_detail::guard_graph_reservation_ready(fixture->context));
  fixture->context.state.graph_reservation.tensor_count = 0u;
  CHECK_FALSE(guard_detail::guard_graph_reservation_ready(fixture->context));
  fixture->context.state.graph_reservation.tensor_count = 1u;
  CHECK(guard_detail::guard_compute_lifecycle_ready(backend.prefill_lifecycle));
  backend.prefill_lifecycle.tensor_count = 0;
  CHECK_FALSE(
      guard_detail::guard_compute_lifecycle_ready(backend.prefill_lifecycle));
  backend.prefill_lifecycle.tensor_count = 1;

  CHECK(guard_detail::guard_step_plan_ready(
      backend.prefill_plan, emel::text::generator::detail::step_kind::prefill));
  backend.prefill_plan.max_step_tokens = 0;
  CHECK_FALSE(guard_detail::guard_step_plan_ready(
      backend.prefill_plan, emel::text::generator::detail::step_kind::prefill));
  backend.prefill_plan.max_step_tokens = 4;
  CHECK(guard_detail::guard_materialized_logits_ready(fixture->context));
  fixture->context.buffers.vocab_size = 0;
  CHECK_FALSE(guard_detail::guard_materialized_logits_ready(fixture->context));
  fixture->context.buffers.vocab_size = 4;
  CHECK(guard_detail::guard_materialized_output_ready(fixture->context));
  backend.bound_logits.clear();
  CHECK_FALSE(guard_detail::guard_materialized_output_ready(fixture->context));
  backend.bound_logits.resize(4u);

  CHECK(guard_detail::guard_bound_request_capacity_ready(fixture->context, 1));
  CHECK_FALSE(
      guard_detail::guard_bound_request_capacity_ready(fixture->context, 0));
  backend.bound_tokens.clear();
  CHECK_FALSE(
      guard_detail::guard_bound_request_capacity_ready(fixture->context, 1));
  backend.bound_tokens.resize(4u);
  backend.bound_positions.clear();
  CHECK_FALSE(
      guard_detail::guard_bound_request_capacity_ready(fixture->context, 1));
  backend.bound_positions.resize(4u);

  emel::text::generator::event::generate_ctx runtime{};
  runtime.prompt_token_count = 1;
  runtime.prefill_step_size = 1;
  runtime.plan_outputs = 1;
  runtime.selected_token = 0;
  runtime.kv_tokens = 0;
  backend.kv_cache_tokens = 0;
  CHECK(guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  runtime.plan_outputs = 2;
  CHECK_FALSE(
      guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  runtime.plan_outputs = 1;
  runtime.prefill_step_size = 0;
  CHECK_FALSE(
      guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  runtime.prefill_step_size = 5;
  CHECK_FALSE(
      guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  runtime.prefill_step_size = 1;
  runtime.prompt_token_count = 0;
  CHECK_FALSE(
      guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  runtime.prompt_token_count = 5;
  CHECK_FALSE(
      guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  runtime.prompt_token_count = 1;

  CHECK(guard_detail::guard_decode_request_ready(runtime, fixture->context));
  runtime.selected_token = -1;
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  runtime.selected_token = backend.token_embedding.rows;
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  runtime.selected_token = 0;
  runtime.kv_tokens = -1;
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  runtime.kv_tokens = backend.n_ctx;
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  runtime.kv_tokens = 0;
  backend.kv_cache_tokens = 1;
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  backend.kv_cache_tokens = 0;

  // Snapshot addressing coherence (KVM-03): geometry drift, inactive
  // sequence, length drift, and missing block mapping each fail the ready
  // predicates and route through the explicit invalid transitions.
  auto &snapshot = fixture->context.state.memory_snapshot;
  CHECK(guard_detail::guard_snapshot_geometry_coherent(fixture->context));
  CHECK(guard_detail::guard_snapshot_covers_tokens(fixture->context, 1));
  CHECK_FALSE(guard_detail::guard_snapshot_covers_tokens(fixture->context, 0));
  CHECK_FALSE(guard_detail::guard_snapshot_covers_tokens(fixture->context, 2));

  snapshot.block_tokens = 4;
  CHECK_FALSE(guard_detail::guard_snapshot_geometry_coherent(fixture->context));
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  CHECK_FALSE(
      guard_detail::guard_prefill_request_ready(runtime, fixture->context));
  snapshot.block_tokens = 8;

  snapshot.sequence_active[0] = 0;
  CHECK_FALSE(guard_detail::guard_snapshot_geometry_coherent(fixture->context));
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  snapshot.sequence_active[0] = 1;

  snapshot.sequence_recurrent_slot[0] = -1;
  CHECK_FALSE(guard_detail::guard_snapshot_geometry_coherent(fixture->context));
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  snapshot.sequence_recurrent_slot[0] = 0;

  snapshot.sequence_length_values[0] = 3;
  CHECK_FALSE(guard_detail::guard_snapshot_covers_tokens(fixture->context, 1));
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  snapshot.sequence_length_values[0] = 1;

  snapshot.sequence_kv_block_count[0] = 0;
  CHECK_FALSE(guard_detail::guard_snapshot_covers_tokens(fixture->context, 1));
  snapshot.sequence_kv_block_count[0] = 1;

  snapshot.sequence_kv_blocks[0][0] = 1;
  CHECK_FALSE(guard_detail::guard_snapshot_covers_tokens(fixture->context, 1));
  CHECK_FALSE(
      guard_detail::guard_decode_request_ready(runtime, fixture->context));
  snapshot.sequence_kv_blocks[0][0] = 0;

  CHECK(guard_detail::guard_decode_request_ready(runtime, fixture->context));
  CHECK(guard_detail::guard_prefill_request_ready(runtime, fixture->context));

  // Flash route identity requirement (KVR-02): the identity map keeps flash
  // eligible; a permuted block map routes to the scalar path.
  CHECK(guard_detail::guard_flash_kv_map_identity(fixture->context, 1));
  snapshot.block_tokens = 4;
  snapshot.sequence_length_values[0] = 8;
  snapshot.sequence_kv_block_count[0] = 2;
  snapshot.sequence_kv_blocks[0][0] = 1;
  snapshot.sequence_kv_blocks[0][1] = 0;
  CHECK_FALSE(guard_detail::guard_flash_kv_map_identity(fixture->context, 8));
  snapshot.sequence_kv_blocks[0][0] = 0;
  snapshot.sequence_kv_blocks[0][1] = 1;
  CHECK(guard_detail::guard_flash_kv_map_identity(fixture->context, 8));
  snapshot.block_tokens = 8;
  snapshot.sequence_length_values[0] = 1;
  snapshot.sequence_kv_block_count[0] = 1;
  snapshot.sequence_kv_blocks[0][1] = 0;
}

TEST_CASE(
    "generator compute readiness guards classify request and backend gaps") {
  auto fixture = std::make_unique<compute_guard_fixture>();
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto request = make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx runtime{};
  auto generate_run = fixture->make_generate_run(request, runtime);
  emel::text::generator::prefill::action::context prefill_context{
      fixture->context};
  const emel::text::generator::prefill::event::run prefill_run{
      generate_run.request,
      generate_run.ctx,
  };

  CHECK(emel::text::generator::guard::
            guard_decode_materialized_scalar_kernel_ready{}(generate_run,
                                                            fixture->context));
  CHECK_FALSE(
      emel::text::generator::guard::guard_decode_compute_invalid_request{}(
          generate_run, fixture->context));
  CHECK_FALSE(
      emel::text::generator::guard::guard_decode_compute_backend_unavailable{}(
          generate_run, fixture->context));
  CHECK(emel::text::generator::prefill::guard::
            guard_materialized_logits_with_scalar_kernel_ready{}(
                prefill_run, prefill_context));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::guard_compute_invalid_request{}(
          prefill_run, prefill_context));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::
          guard_compute_backend_unavailable{}(prefill_run, prefill_context));

  runtime.selected_token =
      fixture->context.compute.backend.token_embedding.rows;
  CHECK(emel::text::generator::guard::guard_decode_compute_invalid_request{}(
      generate_run, fixture->context));
  CHECK_FALSE(
      emel::text::generator::guard::guard_decode_compute_backend_unavailable{}(
          generate_run, fixture->context));
  runtime.selected_token = 0;

  runtime.prompt_token_count = 0;
  CHECK(emel::text::generator::prefill::guard::guard_compute_invalid_request{}(
      prefill_run, prefill_context));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::
          guard_compute_backend_unavailable{}(prefill_run, prefill_context));
  runtime.prompt_token_count = 1;

  fixture->context.compute.backend.bound_logits.clear();
  CHECK(emel::text::generator::guard::guard_decode_compute_invalid_request{}(
      generate_run, fixture->context));
  CHECK(emel::text::generator::prefill::guard::guard_compute_invalid_request{}(
      prefill_run, prefill_context));
  fixture->context.compute.backend.bound_logits.resize(4u);

  fixture->context.compute.backend.decode_lifecycle.phase = nullptr;
  CHECK(
      emel::text::generator::guard::guard_decode_compute_backend_unavailable{}(
          generate_run, fixture->context));
  CHECK_FALSE(
      emel::text::generator::guard::guard_decode_compute_invalid_request{}(
          generate_run, fixture->context));
  fixture->context.compute.backend.decode_lifecycle.phase =
      &fixture->decode_phase;

  fixture->context.compute.backend.prefill_lifecycle.phase = nullptr;
  CHECK(emel::text::generator::prefill::guard::
            guard_compute_backend_unavailable{}(prefill_run, prefill_context));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::guard_compute_invalid_request{}(
          prefill_run, prefill_context));
  fixture->context.compute.backend.prefill_lifecycle.phase =
      &fixture->prefill_phase;

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    (defined(__ARM_FEATURE_DOTPROD) || defined(__ARM_FEATURE_MATMUL_INT8))
  fixture->enable_direct_preselected_argmax();
  CHECK(emel::text::generator::guard::guard_decode_preselected_direct_ready{}(
      generate_run, fixture->context));
  CHECK_FALSE(emel::text::generator::guard::
                  guard_decode_preselected_compute_invalid_request{}(
                      generate_run, fixture->context));

  runtime.selected_token =
      fixture->context.compute.backend.token_embedding.rows;
  CHECK(emel::text::generator::guard::
            guard_decode_preselected_compute_invalid_request{}(
                generate_run, fixture->context));
  runtime.selected_token = 0;
  CHECK(emel::text::generator::prefill::guard::
            guard_preselected_argmax_with_scalar_kernel_ready{}(
                prefill_run, prefill_context));
#endif
}

TEST_CASE("generator core actions cover reset, decode, and fallback channels") {
  emel::text::generator::action::context context{};
  context.model = &test_model();
  context.limits.prompt_capacity = 4;
  context.buffers.vocab_size = 4;
  context.buffers.logits = std::make_unique<float[]>(4);
  context.buffers.logits[0] = 0.25f;
  context.buffers.logits[1] = 3.0f;
  context.buffers.logits[2] = 1.0f;
  context.buffers.logits[3] = -4.0f;
  context.buffers.candidate_capacity = 4;
  context.buffers.candidate_ids = std::make_unique<int32_t[]>(4);
  context.buffers.candidate_scores = std::make_unique<float[]>(4);
  context.compute.backend.input_tokens_tensor_id = 0;
  context.compute.backend.positions_tensor_id = 1;
  context.compute.backend.logits_tensor_id = 2;
  context.compute.backend.key_cache_tensor_id = 3;
  context.compute.backend.value_cache_tensor_id = 4;
  context.compute.backend.lifecycle_tensors.resize(5u);
  context.state.graph_reservation.node_count = 1;
  context.state.graph_reservation.tensor_count = 1;

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 7;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.err = emel::error::cast(emel::text::generator::error::backend);
  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 44;
  generate_ctx.tokens_generated = 5;
  generate_ctx.output_length = 6;
  generate_ctx.selected_token = 2;
  generate_ctx.kv_tokens = 1;
  generate_ctx.prompt_token_count = 2;
  generate_ctx.prefill_step_size = 1;
  generate_ctx.plan_outputs = 1;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  emel::text::generator::action::begin_generate(generate_run, context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::none));
  CHECK(generate_ctx.target_tokens == 2);
  CHECK(output_length_out == 0u);

  generate_ctx.err = emel::error::cast(emel::text::generator::error::none);
  emel::text::generator::action::reject_invalid_generate(generate_run, context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::invalid_request));
  emel::text::generator::action::reject_uninitialized_generate(generate_run,
                                                               context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::invalid_request));

  emel::text::generator::action::mark_invalid_request(generate_run, context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::invalid_request));
  emel::text::generator::action::mark_backend_error(generate_run, context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::backend));
  emel::text::generator::action::on_unexpected(generate_run, context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::backend));

  generate_ctx.selected_token = -1;
  emel::text::generator::action::request_decode_select_argmax(generate_run,
                                                              context);
  CHECK(generate_ctx.selected_token == 1);
  CHECK(generate_ctx.phase_accepted);
  CHECK(generate_ctx.phase_code == 0);

  emel::text::generator::action::request_decode_sample(generate_run, context);
  CHECK(generate_ctx.phase_code >= 0);
  emel::text::generator::action::request_decode_sample_preselected(generate_run,
                                                                   context);
  CHECK(generate_ctx.phase_code >= 0);

  emel::text::generator::action::request_decode_compute_flash_kernel(
      generate_run, context);
  emel::text::generator::action::request_decode_compute_flash_packed_q8_0(
      generate_run, context);
  emel::text::generator::action::request_decode_compute_flash_q8_k(generate_run,
                                                                   context);
  emel::text::generator::action::request_decode_compute_flash_native_quantized(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_native_quantized_q8_k_logits(generate_run,
                                                                context);
  emel::text::generator::action::request_decode_compute_flash_kernel_streamed(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_packed_q8_0_streamed(generate_run, context);
  emel::text::generator::action::request_decode_compute_flash_q8_k_streamed(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_native_quantized_streamed(generate_run,
                                                             context);
  emel::text::generator::action::
      request_decode_compute_flash_native_quantized_q8_k_logits_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_kernel(generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_packed_q8_0(generate_run, context);
  emel::text::generator::action::request_decode_compute_flash_parallel_q8_k(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_native_quantized(generate_run,
                                                             context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_native_quantized_q8_k_logits(
          generate_run, context);
  CHECK(generate_ctx.io.selected_attention_mode ==
        emel::text::generator::attention_mode::flash);
  emel::text::generator::action::request_decode_compute_nonflash_kernel(
      generate_run, context);
  emel::text::generator::action::request_decode_compute_nonflash_packed_q8_0(
      generate_run, context);
  emel::text::generator::action::request_decode_compute_nonflash_q8_k(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_native_quantized(generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_native_quantized_q8_k_logits(generate_run,
                                                                   context);
  emel::text::generator::action::request_decode_compute_nonflash_kernel_streamed(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_packed_q8_0_streamed(generate_run,
                                                           context);
  emel::text::generator::action::request_decode_compute_nonflash_q8_k_streamed(
      generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_native_quantized_streamed(generate_run,
                                                                context);
  emel::text::generator::action::
      request_decode_compute_nonflash_native_quantized_q8_k_logits_streamed(
          generate_run, context);
  CHECK(generate_ctx.io.selected_attention_mode ==
        emel::text::generator::attention_mode::nonflash);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_kernel(generate_run,
                                                             context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_q8_k(generate_run,
                                                           context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_native_quantized_q8_k(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_native_quantized_kernel(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_kernel_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_q8_k_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_native_quantized_q8_k_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_preselected_argmax_native_quantized_kernel_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_preselected_argmax_kernel(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_preselected_argmax_q8_k(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_preselected_argmax_native_quantized_q8_k(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_flash_parallel_preselected_argmax_native_quantized_kernel(
          generate_run, context);
  CHECK(generate_ctx.io.selected_attention_mode ==
        emel::text::generator::attention_mode::flash);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_kernel(generate_run,
                                                                context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_q8_k(generate_run,
                                                              context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_kernel_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_q8_k_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k_streamed(
          generate_run, context);
  emel::text::generator::action::
      request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel_streamed(
          generate_run, context);
  CHECK(generate_ctx.io.selected_attention_mode ==
        emel::text::generator::attention_mode::nonflash);

  const std::array<int32_t, 1> step_sizes{3};
  emel::batch::planner::events::plan_done plan_done{};
  plan_done.step_sizes = step_sizes.data();
  plan_done.step_count = 4;
  plan_done.total_outputs = 5;
  emel::text::generator::action::capture_plan_done(generate_ctx, plan_done);
  CHECK(generate_ctx.prefill_step_size == 3);
  CHECK(generate_ctx.plan_step_count == 4);
  CHECK(generate_ctx.plan_outputs == 5);

  emel::batch::planner::events::plan_error plan_error{
      emel::error::cast(emel::batch::planner::error::invalid_request)};
  emel::text::generator::action::capture_plan_error(generate_ctx, plan_error);
  CHECK(generate_ctx.phase_code == static_cast<int32_t>(plan_error.err));

  emel::graph::event::reserve_output reserve_output{};
  emel::graph::events::reserve_done reserve_done{reserve_output};
  CHECK(
      emel::text::generator::action::capture_graph_reserve_done(reserve_done));
  emel::text::generator::event::initialize_ctx initialize_ctx{};
  emel::graph::events::reserve_error reserve_error{reserve_output, 73};
  CHECK(emel::text::generator::action::capture_graph_reserve_error(
      initialize_ctx, reserve_error));
  CHECK(initialize_ctx.phase_code == 73);
  emel::graph::event::compute_output compute_output{};
  emel::graph::events::compute_done compute_done{compute_output};
  CHECK(
      emel::text::generator::action::capture_graph_compute_done(compute_done));
  emel::graph::events::compute_error compute_error{compute_output, 91};
  emel::text::generator::action::capture_graph_compute_error(generate_ctx,
                                                             compute_error);
  CHECK(generate_ctx.phase_code == 91);

  emel::text::generator::graph_lifecycle_snapshot graph_snapshot{};
  const emel::text::generator::event::capture_graph_lifecycle lifecycle_ev{
      graph_snapshot};
  emel::text::generator::action::capture_graph_lifecycle_without_runtime_tensor(
      lifecycle_ev, context);
  CHECK(graph_snapshot.reservation == &context.state.graph_reservation);
  CHECK(graph_snapshot.first_tensor_captured);
  CHECK_FALSE(graph_snapshot.runtime_tensor_captured);
}

TEST_CASE(
    "generator guards cover success, rejected, and callback matrix branches") {
  emel::text::generator::action::context context{};
  context.model = &test_model();
  context.limits.decode_capacity = 4;
  context.buffers.vocab_size = 2;
  context.buffers.logits = std::make_unique<float[]>(2);
  context.state.selection_mode =
      emel::text::generator::selection_mode::sample_logits;

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  generate_ctx.prompt_token_count = 1;
  generate_ctx.plan_step_count = 1;
  generate_ctx.prefill_step_size = 1;
  generate_ctx.plan_outputs = 0;
  CHECK(
      emel::text::generator::guard::reset_sequence_ok{}(generate_run, context));
  CHECK(emel::text::generator::guard::conditioning_ok{}(generate_run, context));
  CHECK(emel::text::generator::guard::conditioning_ok_with_scalar_prefill{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::planning_ok{}(generate_run, context));
  CHECK(emel::text::generator::guard::allocate_sequence_ok{}(generate_run,
                                                             context));
  CHECK(emel::text::generator::guard::decode_slots_ok{}(generate_run, context));
  CHECK(emel::text::generator::guard::decode_snapshot_ok{}(generate_run,
                                                           context));
  CHECK(
      emel::text::generator::guard::decode_compute_ok{}(generate_run, context));
  CHECK(
      emel::text::generator::guard::decode_sample_ok{}(generate_run, context));
  CHECK(
      emel::text::generator::guard::decode_render_ok{}(generate_run, context));
  CHECK(emel::text::generator::guard::flush_ok{}(generate_run, context));

  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = 1 << 20;
  CHECK(emel::text::generator::guard::reset_sequence_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::conditioning_backend_error{}(generate_run,
                                                                   context));
  CHECK(emel::text::generator::guard::planning_backend_error{}(generate_run,
                                                               context));
  CHECK(emel::text::generator::guard::allocate_sequence_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_slots_backend_error{}(generate_run,
                                                                   context));
  CHECK(emel::text::generator::guard::decode_snapshot_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_compute_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_sample_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_render_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::flush_backend_error{}(generate_run,
                                                            context));

  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::invalid_request));
  CHECK(emel::text::generator::guard::allocate_sequence_invalid_request{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_slots_invalid_request{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_snapshot_invalid_request{}(
      generate_run, context));

  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::out_of_memory));
  CHECK(emel::text::generator::guard::allocate_sequence_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_slots_backend_error{}(generate_run,
                                                                   context));
  CHECK(emel::text::generator::guard::decode_snapshot_backend_error{}(
      generate_run, context));

  generate_ctx.phase_code = 0;
  CHECK(emel::text::generator::guard::allocate_sequence_backend_error{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_slots_backend_error{}(generate_run,
                                                                   context));
  CHECK(emel::text::generator::guard::decode_snapshot_backend_error{}(
      generate_run, context));

  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::logits::sampler::error::internal_error));
  CHECK(emel::text::generator::guard::decode_sample_backend_error{}(
      generate_run, context));

  context.state.selection_mode =
      emel::text::generator::selection_mode::preselected_argmax;
  CHECK(emel::text::generator::guard::decode_uses_preselected_argmax{}(
      generate_run, context));
  CHECK_FALSE(emel::text::generator::guard::decode_uses_materialized_logits{}(
      generate_run, context));

  generate.on_done = {};
  generate.on_error = {};
  generate.error_out = &error_out;
  CHECK(
      emel::text::generator::guard::generate_no_done_callback_with_error_out{}(
          generate_run, context));
  CHECK(
      emel::text::generator::guard::generate_no_error_callback_with_error_out{}(
          generate_run, context));

  generate.error_out = nullptr;
  CHECK(
      emel::text::generator::guard::
          generate_no_done_callback_without_error_out{}(generate_run, context));
  CHECK(emel::text::generator::guard::
            generate_no_error_callback_without_error_out{}(generate_run,
                                                           context));

  generate.on_done = {&tracker, on_generate_done};
  generate.on_error = {&tracker, on_generate_error};
  CHECK(
      emel::text::generator::guard::generate_done_callback_without_error_out{}(
          generate_run, context));
  CHECK(
      emel::text::generator::guard::generate_error_callback_without_error_out{}(
          generate_run, context));
}

TEST_CASE(
    "generator guards cover malformed planning and stop-token fallbacks") {
  emel::text::generator::action::context context{};
  context.limits.decode_capacity = 4;

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.err = emel::error::cast(emel::text::generator::error::none);
  generate_ctx.render_status = emel::text::renderer::sequence_status::running;
  generate_ctx.tokens_generated = 0;
  generate_ctx.target_tokens = 2;
  generate_ctx.selected_token = 7;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};

  CHECK(emel::text::generator::guard::decode_should_continue{}(generate_run,
                                                               context));

  context.model = &test_model();
  CHECK_FALSE(emel::text::generator::guard::decode_should_continue{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_complete{}(generate_run, context));

  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  generate_ctx.plan_step_count = 1;
  generate_ctx.prefill_step_size = 1;
  generate_ctx.plan_outputs = 0;
  CHECK_FALSE(emel::text::generator::guard::planning_backend_error{}(
      generate_run, context));

  generate_ctx.plan_step_count = 0;
  CHECK(emel::text::generator::guard::planning_backend_error{}(generate_run,
                                                               context));
  generate_ctx.plan_step_count = 1;
  generate_ctx.prefill_step_size = 0;
  CHECK(emel::text::generator::guard::planning_backend_error{}(generate_run,
                                                               context));
  generate_ctx.prefill_step_size = 1;
  generate_ctx.plan_outputs = -1;
  CHECK(emel::text::generator::guard::planning_backend_error{}(generate_run,
                                                               context));

  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = static_cast<int32_t>(emel::error::cast(
      emel::batch::planner::error::planning_progress_stalled));
  CHECK(emel::text::generator::guard::planning_backend_error{}(generate_run,
                                                               context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::invalid_request));
  CHECK(emel::text::generator::guard::decode_compute_invalid_request{}(
      generate_run, context));
}

TEST_CASE("generator guard detail helpers cover backend code alternatives") {
  CHECK(emel::text::generator::guard::detail::conditioner_backend_code(
      static_cast<int32_t>(emel::text::conditioner::error::untracked)));
  CHECK(emel::text::generator::guard::detail::renderer_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::text::renderer::error::internal_error))));
  CHECK(emel::text::generator::guard::detail::renderer_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::text::renderer::error::untracked))));
  CHECK(emel::text::generator::guard::detail::memory_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::memory::hybrid::error::internal_error))));
  CHECK(emel::text::generator::guard::detail::memory_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::memory::hybrid::error::untracked))));
  CHECK(emel::text::generator::guard::detail::graph_backend_code(
      static_cast<int32_t>(emel::error::cast(emel::graph::error::busy))));
  CHECK(emel::text::generator::guard::detail::graph_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::graph::error::internal_error))));
  CHECK(emel::text::generator::guard::detail::graph_backend_code(
      static_cast<int32_t>(emel::error::cast(emel::graph::error::untracked))));
  CHECK(emel::text::generator::guard::detail::sampler_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::logits::sampler::error::untracked))));

  CHECK(emel::text::generator::initializer::guard::detail::loader_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::model::loader::error::untracked))));
  CHECK(emel::text::generator::initializer::guard::detail::
            conditioner_backend_code(static_cast<int32_t>(
                emel::text::conditioner::error::untracked)));
  CHECK(
      emel::text::generator::initializer::guard::detail::renderer_backend_code(
          static_cast<int32_t>(
              emel::error::cast(emel::text::renderer::error::internal_error))));
  CHECK(
      emel::text::generator::initializer::guard::detail::renderer_backend_code(
          static_cast<int32_t>(
              emel::error::cast(emel::text::renderer::error::untracked))));
  CHECK(emel::text::generator::initializer::guard::detail::memory_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::memory::hybrid::error::internal_error))));
  CHECK(emel::text::generator::initializer::guard::detail::memory_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::memory::hybrid::error::untracked))));
  CHECK(emel::text::generator::initializer::guard::detail::graph_backend_code(
      static_cast<int32_t>(emel::error::cast(emel::graph::error::busy))));
  CHECK(emel::text::generator::initializer::guard::detail::graph_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::graph::error::internal_error))));
  CHECK(emel::text::generator::initializer::guard::detail::graph_backend_code(
      static_cast<int32_t>(emel::error::cast(emel::graph::error::untracked))));

  CHECK(emel::text::generator::prefill::guard::detail::memory_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::memory::hybrid::error::internal_error))));
  CHECK(emel::text::generator::prefill::guard::detail::memory_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::memory::hybrid::error::untracked))));
  CHECK(emel::text::generator::prefill::guard::detail::graph_backend_code(
      static_cast<int32_t>(
          emel::error::cast(emel::graph::error::internal_error))));
  CHECK(emel::text::generator::prefill::guard::detail::graph_backend_code(
      static_cast<int32_t>(emel::error::cast(emel::graph::error::untracked))));

  constexpr int32_t unrelated_code = 123456;
  CHECK_FALSE(emel::text::generator::guard::detail::conditioner_backend_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::renderer_invalid_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::renderer_backend_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::memory_invalid_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::memory_backend_code(
      unrelated_code));
  CHECK_FALSE(
      emel::text::generator::guard::detail::graph_invalid_code(unrelated_code));
  CHECK_FALSE(
      emel::text::generator::guard::detail::graph_backend_code(unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::sampler_invalid_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::sampler_backend_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::guard::detail::planner_invalid_code(0));
  CHECK_FALSE(emel::text::generator::guard::detail::planner_backend_code(0));

  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::loader_invalid_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::loader_backend_code(
          unrelated_code));
  CHECK_FALSE(emel::text::generator::initializer::guard::detail::
                  conditioner_invalid_code(unrelated_code));
  CHECK_FALSE(emel::text::generator::initializer::guard::detail::
                  conditioner_backend_code(unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::renderer_invalid_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::renderer_backend_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::memory_invalid_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::memory_backend_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::graph_invalid_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::initializer::guard::detail::graph_backend_code(
          unrelated_code));

  CHECK_FALSE(
      emel::text::generator::prefill::guard::detail::memory_invalid_code(
          unrelated_code));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::detail::memory_backend_code(
          unrelated_code));
  CHECK_FALSE(emel::text::generator::prefill::guard::detail::graph_invalid_code(
      unrelated_code));
  CHECK_FALSE(emel::text::generator::prefill::guard::detail::graph_backend_code(
      unrelated_code));
}

TEST_CASE(
    "generator initialize guards cover all callback channel combinations") {
  emel::text::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);

  auto initialize = make_initialize_request(&tracker, &error_out);
  emel::text::generator::event::initialize_ctx initialize_ctx{};
  initialize_ctx.err = emel::error::cast(emel::text::generator::error::backend);
  emel::text::generator::event::initialize_run initialize_run{initialize,
                                                              initialize_ctx};

  CHECK(emel::text::generator::guard::initialize_result_backend{}(
      initialize_run, context));
  CHECK(emel::text::generator::guard::initialize_done_callback_with_error_out{}(
      initialize_run, context));
  CHECK(
      emel::text::generator::guard::initialize_error_callback_with_error_out{}(
          initialize_run, context));

  initialize.error_out = nullptr;
  CHECK(emel::text::generator::guard::
            initialize_done_callback_without_error_out{}(initialize_run,
                                                         context));
  CHECK(emel::text::generator::guard::
            initialize_error_callback_without_error_out{}(initialize_run,
                                                          context));

  initialize.error_out = &error_out;
  initialize.on_done = {};
  initialize.on_error = {};
  CHECK(emel::text::generator::guard::
            initialize_no_done_callback_with_error_out{}(initialize_run,
                                                         context));
  CHECK(emel::text::generator::guard::
            initialize_no_error_callback_with_error_out{}(initialize_run,
                                                          context));

  initialize.error_out = nullptr;
  CHECK(emel::text::generator::guard::
            initialize_no_done_callback_without_error_out{}(initialize_run,
                                                            context));
  CHECK(emel::text::generator::guard::
            initialize_no_error_callback_without_error_out{}(initialize_run,
                                                             context));
}

TEST_CASE("generator runtime guards model explicit flash and nonflash compute "
          "selection") {
  emel::text::generator::action::context context{};
  auto &backend = context.compute.backend;
  backend.n_layer = 1;
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 4;
  backend.kv_block_tokens = 4;
  backend.kv_positions_capacity = 4;
  context.state.memory_snapshot.max_sequences = 1;
  context.state.memory_snapshot.block_tokens = 4;
  context.state.memory_snapshot.sequence_active[0] = 1;
  context.state.memory_snapshot.sequence_length_values[0] = 2;
  context.state.memory_snapshot.sequence_kv_block_count[0] = 1;
  context.state.memory_snapshot.sequence_kv_blocks[0][0] = 0;
  backend.q_attn.resize(4, 0.0f);
  backend.key_cache.resize(16, 0.0f);
  backend.value_cache.resize(16, 0.0f);
  backend.flash_key_cache.resize(16, 0.0f);
  backend.flash_value_cache.resize(16, 0.0f);
  backend.attn_ctx.resize(4, 0.0f);

  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 2;
  generate_ctx.kv_tokens = 1;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};
  CHECK(emel::text::generator::guard::decode_flash_runtime_supported{}(
      generate_run, context));
  CHECK_FALSE(emel::text::generator::guard::decode_nonflash_runtime_required{}(
      generate_run, context));

  backend.head_dim_kv = 1;
  backend.key_cache.resize(8, 0.0f);
  backend.value_cache.resize(8, 0.0f);
  backend.flash_key_cache.resize(8, 0.0f);
  backend.flash_value_cache.resize(8, 0.0f);

  CHECK_FALSE(emel::text::generator::guard::decode_flash_runtime_supported{}(
      generate_run, context));
  CHECK(emel::text::generator::guard::decode_nonflash_runtime_required{}(
      generate_run, context));
}

TEST_CASE("generator prefill runtime guards model explicit flash and compute "
          "result selection") {
  emel::text::generator::action::context generator_context{};
  auto &backend = generator_context.compute.backend;
  backend.n_layer = 1;
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 4;
  backend.kv_block_tokens = 4;
  backend.kv_positions_capacity = 4;
  generator_context.state.memory_snapshot.max_sequences = 1;
  generator_context.state.memory_snapshot.block_tokens = 4;
  generator_context.state.memory_snapshot.sequence_active[0] = 1;
  generator_context.state.memory_snapshot.sequence_length_values[0] = 2;
  generator_context.state.memory_snapshot.sequence_kv_block_count[0] = 1;
  generator_context.state.memory_snapshot.sequence_kv_blocks[0][0] = 0;
  backend.q_attn.resize(4, 0.0f);
  backend.key_cache.resize(16, 0.0f);
  backend.value_cache.resize(16, 0.0f);
  backend.flash_key_cache.resize(16, 0.0f);
  backend.flash_value_cache.resize(16, 0.0f);
  backend.attn_ctx.resize(4, 0.0f);

  emel::text::generator::prefill::action::context prefill_context{
      generator_context};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 2;
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  CHECK(emel::text::generator::prefill::guard::flash_runtime_supported{}(
      prefill_run, prefill_context));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::nonflash_runtime_required{}(
          prefill_run, prefill_context));

  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::invalid_request));
  CHECK(emel::text::generator::prefill::guard::compute_invalid_request{}(
      prefill_run, prefill_context));
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::processor_failed));
  CHECK(emel::text::generator::prefill::guard::compute_backend_error{}(
      prefill_run, prefill_context));

  backend.head_dim_kv = 1;
  backend.key_cache.resize(8, 0.0f);
  backend.value_cache.resize(8, 0.0f);
  backend.flash_key_cache.resize(8, 0.0f);
  backend.flash_value_cache.resize(8, 0.0f);

  CHECK_FALSE(emel::text::generator::prefill::guard::flash_runtime_supported{}(
      prefill_run, prefill_context));
  CHECK(emel::text::generator::prefill::guard::nonflash_runtime_required{}(
      prefill_run, prefill_context));
}

TEST_CASE("generator prefill parallel route guard exposes unavailable causes") {
  auto fixture = std::make_unique<compute_guard_fixture>();
  auto &backend = fixture->context.compute.backend;
  backend.routes = emel::text::generator::test::k_generation_route_policy;

  emel::text::generator::prefill::action::context prefill_context{
      fixture->context};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = backend.routes.parallel_min_prefill_tokens;
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  CHECK(emel::text::generator::prefill::guard::detail::uses_parallel_matmul_lanes(prefill_run, prefill_context));

  generate_ctx.prompt_token_count =
      backend.routes.parallel_min_prefill_tokens - 1;
  CHECK_FALSE(
      emel::text::generator::prefill::guard::detail::uses_parallel_matmul_lanes(
          prefill_run, prefill_context));
  generate_ctx.prompt_token_count = backend.routes.parallel_min_prefill_tokens;

  backend.parallel_lanes_enabled = false;
  CHECK_FALSE(
      emel::text::generator::prefill::guard::detail::uses_parallel_matmul_lanes(
          prefill_run, prefill_context));
  backend.parallel_lanes_enabled = true;

  auto *matmul_actor = backend.matmul_actor;
  backend.matmul_actor = nullptr;
  CHECK_FALSE(
      emel::text::generator::prefill::guard::detail::uses_parallel_matmul_lanes(
          prefill_run, prefill_context));
  backend.matmul_actor = matmul_actor;
}

TEST_CASE("generator prefill guards keep sub-chunk prompts on scalar route") {
  chunk4_planning_backend_fixture fixture{};
  auto &backend = fixture.context.compute.backend;
  backend.routes.prefill_chunk4_min_tokens = 0;
  backend.routes.prefill_chunk8_min_tokens = 0;

  emel::text::generator::prefill::action::context prefill_context{
      fixture.context};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count =
      emel::text::generator::detail::k_prefill_q8_chunk_rows - 1;
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  CHECK_FALSE(emel::text::generator::prefill::guard::
                  uses_materialized_logits_with_chunk8_q8_k{}(
                      prefill_run, prefill_context));
  CHECK_FALSE(emel::text::generator::prefill::guard::
                  uses_materialized_logits_with_chunk4_q8_k{}(
                      prefill_run, prefill_context));
  CHECK(emel::text::generator::prefill::guard::
            uses_materialized_logits_with_scalar{}(prefill_run,
                                                  prefill_context));
}

TEST_CASE(
    "generator prefill guards classify explicit chunk8 routes and contracts") {
  chunk4_planning_backend_fixture fixture{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count =
      emel::text::generator::detail::k_prefill_q8_chunk8_rows;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};
  emel::text::generator::prefill::action::context prefill_context{
      fixture.context};
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  generate_ctx.prefill_contract = emel::text::generator::
      prefill_compute_contract::flash_materialized_chunk8_q8_k;
  CHECK(emel::text::generator::guard::detail::
            prefill_contract_uses_materialized_logits(
                generate_ctx.prefill_contract));
  CHECK_FALSE(emel::text::generator::guard::detail::
                  prefill_contract_uses_preselected_argmax(
                      generate_ctx.prefill_contract));
  CHECK(emel::text::generator::guard::
            prefill_result_ok_with_materialized_logits_contract{}(
                generate_run, fixture.context));

  generate_ctx.prefill_contract = emel::text::generator::
      prefill_compute_contract::flash_preselected_chunk8_q8_k;
  CHECK_FALSE(emel::text::generator::guard::detail::
                  prefill_contract_uses_materialized_logits(
                      generate_ctx.prefill_contract));
  CHECK(emel::text::generator::guard::detail::
            prefill_contract_uses_preselected_argmax(
                generate_ctx.prefill_contract));
  CHECK(emel::text::generator::guard::
            prefill_result_ok_with_preselected_argmax_contract{}(
                generate_run, fixture.context));

  generate_ctx.prefill_contract = emel::text::generator::
      prefill_compute_contract::nonflash_materialized_chunk8_q8_k;
  CHECK(emel::text::generator::guard::detail::
            prefill_contract_uses_materialized_logits(
                generate_ctx.prefill_contract));

  generate_ctx.prefill_contract = emel::text::generator::
      prefill_compute_contract::nonflash_preselected_chunk8_q8_k;
  CHECK(emel::text::generator::guard::detail::
            prefill_contract_uses_preselected_argmax(
                generate_ctx.prefill_contract));

  if (host_supports_chunk8_prefill_q8_route()) {
    prefill_context.generator.state.selection_mode =
        emel::text::generator::selection_mode::sample_logits;
    CHECK(emel::text::generator::prefill::guard::
              uses_materialized_logits_with_chunk8_q8_k{}(prefill_run,
                                                          prefill_context));
    CHECK_FALSE(emel::text::generator::prefill::guard::
                    uses_materialized_logits_with_chunk4_q8_k{}(
                        prefill_run, prefill_context));
    CHECK_FALSE(emel::text::generator::prefill::guard::
                    uses_materialized_logits_with_scalar{}(prefill_run,
                                                           prefill_context));

    prefill_context.generator.state.selection_mode =
        emel::text::generator::selection_mode::preselected_argmax;
    if (emel::text::generator::guard::detail::
            preselected_argmax_direct_supported(
                prefill_context.generator.compute.backend)) {
      CHECK(emel::text::generator::prefill::guard::
                uses_preselected_argmax_with_chunk8_q8_k{}(prefill_run,
                                                           prefill_context));
      CHECK_FALSE(emel::text::generator::prefill::guard::
                      uses_preselected_argmax_with_chunk4_q8_k{}(
                          prefill_run, prefill_context));
      CHECK_FALSE(emel::text::generator::prefill::guard::
                      uses_preselected_argmax_with_scalar{}(prefill_run,
                                                            prefill_context));
    } else {
      CHECK_FALSE(emel::text::generator::prefill::guard::
                      uses_preselected_argmax_with_chunk8_q8_k{}(
                          prefill_run, prefill_context));
    }
  }
}

TEST_CASE("generator guards classify explicit native quantized materialized q8 "
          "logits routes") {
  native_quantized_route_fixture fixture{};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;

  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 1;
  emel::text::generator::event::generate_run generate_run{generate,
                                                          generate_ctx};
  emel::text::generator::prefill::action::context prefill_context{
      fixture.context};
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  fixture.context.state.selection_mode =
      emel::text::generator::selection_mode::sample_logits;
  CHECK(
      emel::text::generator::guard::compute_scalar_native_quantized_supported{}(
          generate_run, fixture.context));

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
    defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(
      emel::text::generator::guard::detail::materialized_output_q8_k_supported(
          fixture.context.compute.backend));
  CHECK(emel::text::generator::guard::
            compute_materialized_scalar_native_quantized_q8_k_supported{}(
                generate_run, fixture.context));
  CHECK_FALSE(
      emel::text::generator::guard::
          compute_materialized_scalar_native_quantized_kernel_required{}(
              generate_run, fixture.context));
  CHECK(emel::text::generator::prefill::guard::
            uses_materialized_logits_with_scalar_native_quantized_q8_k{}(
                prefill_run, prefill_context));
  CHECK_FALSE(
      emel::text::generator::prefill::guard::
          uses_materialized_logits_with_scalar_native_quantized_kernel{}(
              prefill_run, prefill_context));
#else
  CHECK_FALSE(
      emel::text::generator::guard::detail::materialized_output_q8_k_supported(
          fixture.context.compute.backend));
  CHECK_FALSE(emel::text::generator::guard::
                  compute_materialized_scalar_native_quantized_q8_k_supported{}(
                      generate_run, fixture.context));
  CHECK(emel::text::generator::guard::
            compute_materialized_scalar_native_quantized_kernel_required{}(
                generate_run, fixture.context));
  CHECK_FALSE(emel::text::generator::prefill::guard::
                  uses_materialized_logits_with_scalar_native_quantized_q8_k{}(
                      prefill_run, prefill_context));
  CHECK(emel::text::generator::prefill::guard::
            uses_materialized_logits_with_scalar_native_quantized_kernel{}(
                prefill_run, prefill_context));
#endif

  fixture.context.compute.backend.output.tensor = &fixture.body_tensor;
  CHECK_FALSE(
      emel::text::generator::guard::detail::materialized_output_q8_k_supported(
          fixture.context.compute.backend));
  CHECK(emel::text::generator::guard::
            compute_materialized_scalar_native_quantized_kernel_required{}(
                generate_run, fixture.context));
}

TEST_CASE("generator prefill actions publish every explicit compute contract") {
  emel::text::generator::action::context generator_context{};
  generator_context.limits.prompt_capacity = 4;
  generator_context.buffers.vocab_size = 4;
  generator_context.buffers.logits = std::make_unique<float[]>(4);
  generator_context.compute.backend.input_tokens_tensor_id = 0;
  generator_context.compute.backend.positions_tensor_id = 1;
  generator_context.compute.backend.logits_tensor_id = 2;
  generator_context.compute.backend.key_cache_tensor_id = 3;
  generator_context.compute.backend.value_cache_tensor_id = 4;
  generator_context.compute.backend.lifecycle_tensors.resize(5u);
  generator_context.state.graph_reservation.node_count = 1;
  generator_context.state.graph_reservation.tensor_count = 1;

  emel::text::generator::prefill::action::context prefill_context{
      generator_context};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 2;
  generate_ctx.prefill_step_size = 1;
  generate_ctx.plan_outputs = 1;
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  const auto check_contract =
      [&](const auto &action,
          const emel::text::generator::prefill_compute_contract expected) {
        generate_ctx.prefill_contract =
            emel::text::generator::prefill_compute_contract::none;
        action(prefill_run, prefill_context);
        CHECK(generate_ctx.prefill_contract == expected);
      };

  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_scalar_kernel,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_scalar_packed_q8_0,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_scalar_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_flash_materialized_scalar_native_quantized,
      emel::text::generator::prefill_compute_contract::
          flash_materialized_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_flash_materialized_scalar_native_quantized_q8_k,
      emel::text::generator::prefill_compute_contract::
          flash_materialized_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_chunk8_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_chunk8_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_parallel_chunk8_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_chunk8_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_chunk4_packed_q8_0,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_chunk4_packed_q8_0);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_flash_materialized_parallel_chunk4_packed_q8_0,
      emel::text::generator::prefill_compute_contract::
          flash_materialized_chunk4_packed_q8_0);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_chunk4_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_chunk4_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_materialized_parallel_chunk4_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_materialized_chunk4_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_scalar_kernel,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_scalar_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_flash_preselected_scalar_native_quantized_q8_k,
      emel::text::generator::prefill_compute_contract::
          flash_preselected_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_flash_preselected_scalar_native_quantized_kernel,
      emel::text::generator::prefill_compute_contract::
          flash_preselected_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_chunk8_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_chunk8_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_parallel_chunk8_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_chunk8_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_chunk4_packed_q8_0,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_chunk4_packed_q8_0);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_flash_preselected_parallel_chunk4_packed_q8_0,
      emel::text::generator::prefill_compute_contract::
          flash_preselected_chunk4_packed_q8_0);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_chunk4_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_chunk4_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_flash_preselected_parallel_chunk4_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     flash_preselected_chunk4_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_materialized_scalar_kernel,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_materialized_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_materialized_scalar_packed_q8_0,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_materialized_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_materialized_scalar_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_materialized_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_nonflash_materialized_scalar_native_quantized,
      emel::text::generator::prefill_compute_contract::
          nonflash_materialized_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_nonflash_materialized_scalar_native_quantized_q8_k,
      emel::text::generator::prefill_compute_contract::
          nonflash_materialized_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_materialized_chunk8_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_materialized_chunk8_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_materialized_chunk4_packed_q8_0,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_materialized_chunk4_packed_q8_0);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_materialized_chunk4_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_materialized_chunk4_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_preselected_scalar_kernel,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_preselected_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_preselected_scalar_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_preselected_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_nonflash_preselected_scalar_native_quantized_q8_k,
      emel::text::generator::prefill_compute_contract::
          nonflash_preselected_scalar);
  check_contract(
      emel::text::generator::prefill::action::
          request_contract_nonflash_preselected_scalar_native_quantized_kernel,
      emel::text::generator::prefill_compute_contract::
          nonflash_preselected_scalar);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_preselected_chunk8_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_preselected_chunk8_q8_k);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_preselected_chunk4_packed_q8_0,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_preselected_chunk4_packed_q8_0);
  check_contract(emel::text::generator::prefill::action::
                     request_contract_nonflash_preselected_chunk4_q8_k,
                 emel::text::generator::prefill_compute_contract::
                     nonflash_preselected_chunk4_q8_k);
}

TEST_CASE("generator prefill actions and guards classify phase outcomes") {
  emel::text::generator::action::context generator_context{};
  emel::text::generator::prefill::action::context prefill_context{
      generator_context};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  size_t output_length_out = 0;
  auto generate =
      make_generate_request(&tracker, &error_out, output_length_out);
  emel::text::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 3;
  emel::text::generator::prefill::event::run prefill_run{generate,
                                                         generate_ctx};

  emel::text::generator::prefill::action::mark_prefill_cached(prefill_run,
                                                              prefill_context);
  CHECK(generate_ctx.kv_tokens == 3);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::none));

  emel::text::generator::prefill::action::mark_invalid_request(prefill_run,
                                                               prefill_context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::invalid_request));

  emel::text::generator::prefill::action::mark_backend_error(prefill_run,
                                                             prefill_context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::backend));

  emel::text::generator::prefill::action::on_unexpected(prefill_run,
                                                        prefill_context);
  CHECK(generate_ctx.err ==
        emel::error::cast(emel::text::generator::error::backend));

  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  CHECK(emel::text::generator::prefill::guard::slots_ok{}(prefill_run,
                                                          prefill_context));
  CHECK(emel::text::generator::prefill::guard::snapshot_ok{}(prefill_run,
                                                             prefill_context));
  CHECK(emel::text::generator::prefill::guard::compute_ok{}(prefill_run,
                                                            prefill_context));

  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::invalid_request));
  CHECK(emel::text::generator::prefill::guard::slots_invalid_request{}(
      prefill_run, prefill_context));
  CHECK(emel::text::generator::prefill::guard::snapshot_invalid_request{}(
      prefill_run, prefill_context));

  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::memory::hybrid::error::out_of_memory));
  CHECK(emel::text::generator::prefill::guard::slots_backend_error{}(
      prefill_run, prefill_context));
  CHECK(emel::text::generator::prefill::guard::snapshot_backend_error{}(
      prefill_run, prefill_context));

  generate_ctx.phase_code = 0;
  CHECK(emel::text::generator::prefill::guard::slots_backend_error{}(
      prefill_run, prefill_context));

  generate_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::graph::error::invalid_request));
  CHECK(emel::text::generator::prefill::guard::compute_invalid_request{}(
      prefill_run, prefill_context));
  CHECK_FALSE(emel::text::generator::prefill::guard::compute_backend_error{}(
      prefill_run, prefill_context));

  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::error::busy));
  CHECK(emel::text::generator::prefill::guard::compute_backend_error{}(
      prefill_run, prefill_context));

  generate_ctx.phase_code = 0;
  CHECK(emel::text::generator::prefill::guard::compute_backend_error{}(
      prefill_run, prefill_context));
}

TEST_CASE(
    "generator initializer guards cover readiness and phase outcome branches") {
  emel::text::generator::action::context generator_context{};
  emel::text::generator::initializer::action::context initializer_context{
      generator_context};
  callback_tracker tracker{};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  auto initialize = make_initialize_request(&tracker, &error_out);
  emel::text::generator::event::initialize_ctx initialize_ctx{};
  const emel::text::generator::initializer::event::run initializer_run{
      initialize,
      initialize_ctx,
  };

  CHECK(emel::text::generator::initializer::guard::backend_prepare_needed{}(
      initializer_run, initializer_context));
  generator_context.compute.backend_ready = true;
  generator_context.limits.block_tokens = initialize.block_tokens;
  generator_context.compute.backend.n_ctx = 8;
  generator_context.compute.backend.kv_block_tokens = initialize.block_tokens;
  generator_context.compute.backend.kv_positions_capacity = 8;
  CHECK(emel::text::generator::initializer::guard::backend_already_ready{}(
      initializer_run, initializer_context));
  generator_context.compute.backend.kv_block_tokens =
      initialize.block_tokens + 1;
  CHECK(emel::text::generator::initializer::guard::backend_prepare_needed{}(
      initializer_run, initializer_context));
  generator_context.compute.backend.kv_block_tokens = initialize.block_tokens;

  initialize_ctx.phase_accepted = true;
  initialize_ctx.phase_code = 0;
  CHECK(emel::text::generator::initializer::guard::backend_prepare_ok{}(
      initializer_run, initializer_context));
  CHECK(emel::text::generator::initializer::guard::conditioner_bind_ok{}(
      initializer_run, initializer_context));
  CHECK(emel::text::generator::initializer::guard::renderer_initialize_ok{}(
      initializer_run, initializer_context));
  CHECK(emel::text::generator::initializer::guard::memory_reserve_ok{}(
      initializer_run, initializer_context));
  CHECK(emel::text::generator::initializer::guard::graph_reserve_ok{}(
      initializer_run, initializer_context));

  generator_context.state.graph_reservation.node_count = 1;
  CHECK(emel::text::generator::initializer::guard::graph_reservation_present{}(
      initializer_run, initializer_context));
  CHECK(emel::text::generator::initializer::guard::
            memory_reserve_with_existing_graph{}(initializer_run,
                                                 initializer_context));
  generator_context.state.graph_reservation.node_count = 0;
  CHECK(emel::text::generator::initializer::guard::graph_reservation_missing{}(
      initializer_run, initializer_context));
  CHECK(emel::text::generator::initializer::guard::
            memory_reserve_with_missing_graph{}(initializer_run,
                                                initializer_context));

  initialize_ctx.phase_accepted = false;
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(emel::text::generator::initializer::guard::
            backend_prepare_invalid_request{}(initializer_run,
                                              initializer_context));
  initialize_ctx.phase_code = static_cast<int32_t>(
      emel::error::cast(emel::model::loader::error::parse_failed));
  CHECK(emel::text::generator::initializer::guard::
            backend_prepare_backend_error{}(initializer_run,
                                            initializer_context));
  initialize_ctx.phase_code = 0;
  CHECK(emel::text::generator::initializer::guard::
            backend_prepare_backend_error{}(initializer_run,
                                            initializer_context));
}
