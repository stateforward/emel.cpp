#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "emel/error/error.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/speech/predictor/moshi/any.hpp"
#include "emel/speech/predictor/moshi/executor/any.hpp"
#include "moshi_fixture.hpp"

namespace {

namespace memory_streaming = emel::memory::streaming;
namespace moshi = emel::speech::predictor::moshi;
namespace executor = emel::speech::predictor::moshi::executor;
using emel::speech::predictor::moshi::test::load_fixture_or_skip;

struct cache_storage {
  cache_storage(const int32_t layer_count, const int32_t capacity,
                const int32_t dimension)
      : key(static_cast<size_t>(layer_count) * static_cast<size_t>(capacity) *
            static_cast<size_t>(dimension)),
        value(key.size()), offsets(static_cast<size_t>(layer_count)) {
    const size_t per_layer =
        static_cast<size_t>(capacity) * static_cast<size_t>(dimension);
    for (int32_t layer = 0; layer < layer_count; ++layer) {
      offsets[static_cast<size_t>(layer)] =
          static_cast<size_t>(layer) * per_layer;
    }
  }

  std::vector<uint16_t> key;
  std::vector<uint16_t> value;
  std::vector<size_t> offsets;
};

bool bind_temporal_cache(void *storage_ptr, const emel::model::data &model,
                         const emel::memory::view::snapshot &, int32_t,
                         executor::detail::temporal_kv_view &view) noexcept {
  auto &storage = *static_cast<cache_storage *>(storage_ptr);
  view.key_cache = storage.key;
  view.value_cache = storage.value;
  view.layer_cache_offsets = storage.offsets;
  view.layer_count = model.moshi_lm.num_layers;
  view.position_capacity = model.moshi_lm.context;
  view.kv_dim = model.moshi_lm.dim;
  return true;
}

bool bind_depformer_cache(void *storage_ptr, const emel::model::data &model,
                          const emel::memory::view::snapshot &, int32_t,
                          executor::detail::depformer_kv_view &view) noexcept {
  auto &storage = *static_cast<cache_storage *>(storage_ptr);
  view.key_cache = storage.key;
  view.value_cache = storage.value;
  view.layer_cache_offsets = storage.offsets;
  view.layer_count = model.moshi_lm.depformer_num_layers;
  view.position_capacity = model.moshi_lm.depformer_context;
  view.kv_dim = model.moshi_lm.depformer_dim;
  return true;
}

memory_streaming::window_view capture(memory_streaming::sm &machine) {
  memory_streaming::window_view view = {};
  int32_t err =
      static_cast<int32_t>(emel::error::cast(memory_streaming::error::none));
  REQUIRE(machine.process_event(memory_streaming::event::capture_view{
      .view_out = view, .error_out = err}));
  REQUIRE(err == static_cast<int32_t>(
                     emel::error::cast(memory_streaming::error::none)));
  return view;
}

} // namespace

TEST_CASE(
    "speech Moshi executor delegates ring positions to streaming actors") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }
  const auto &lm = fixture.model->moshi_lm;

  cache_storage temporal_cache{lm.num_layers, lm.context, lm.dim};
  cache_storage depformer_cache{lm.depformer_num_layers, lm.depformer_context,
                                lm.depformer_dim};
  memory_streaming::sm temporal_positions{
      memory_streaming::dependencies{.capacity = lm.context}};
  memory_streaming::sm depformer_positions{
      memory_streaming::dependencies{.capacity = lm.depformer_context}};
  int32_t memory_err =
      static_cast<int32_t>(emel::error::cast(memory_streaming::error::none));
  REQUIRE(temporal_positions.process_event(
      memory_streaming::event::initialize{.error_out = memory_err}));
  REQUIRE(memory_err == static_cast<int32_t>(
                            emel::error::cast(memory_streaming::error::none)));
  REQUIRE(depformer_positions.process_event(
      memory_streaming::event::initialize{.error_out = memory_err}));
  REQUIRE(memory_err == static_cast<int32_t>(
                            emel::error::cast(memory_streaming::error::none)));

  const auto temporal_binding =
      executor::bind_temporal_kv_cache(&temporal_cache, bind_temporal_cache);
  const auto depformer_binding =
      executor::bind_depformer_kv_cache(&depformer_cache, bind_depformer_cache);
  executor::sm machine{
      executor::bind_kv_caches(temporal_binding, depformer_binding,
                               temporal_positions, depformer_positions)};

  emel::error::type err = emel::error::cast(executor::error::none);
  executor::event::initialize initialize{*fixture.model};
  initialize.error_out = &err;
  REQUIRE(machine.process_event(initialize));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  input.fill(-1);
  input[0] = lm.text_padding_id;
  emel::memory::view::snapshot unused_snapshot = {};
  int32_t text_token = -1;
  moshi::event::graph_step step{
      *fixture.model, unused_snapshot,
      std::span<const int32_t>{input.data(), static_cast<size_t>(lm.n_q + 1)},
      std::span<int32_t>{output.data(), static_cast<size_t>(lm.dep_q)},
      text_token};
  step.error_out = &err;

  REQUIRE(machine.process_event(step));
  CHECK(capture(temporal_positions).logical_end == 1);
  CHECK(capture(depformer_positions).logical_end == lm.dep_q);

  REQUIRE(machine.process_event(step));
  CHECK(capture(temporal_positions).logical_end == 2);
  CHECK(capture(depformer_positions).logical_end == lm.dep_q);
}
