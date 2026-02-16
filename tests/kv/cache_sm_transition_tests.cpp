#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"

namespace {

TEST_CASE("kv_cache_sm_prepare_populates_outputs_and_capacity_error") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 2> ubatch_sizes = {{2, 1}};
  std::array<int32_t, 1> slot_offsets = {{0}};
  int32_t ubatch_count = 0;
  int32_t err = EMEL_OK;

  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 8,
    .slot_offsets_out = slot_offsets.data(),
    .slot_offsets_capacity = static_cast<int32_t>(slot_offsets.size()),
    .ubatch_count_out = &ubatch_count,
    .error_out = &err,
  });
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("kv_cache_sm_prepare_apply_rollback_optional_outputs") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 2> ubatch_sizes = {{1, 1}};
  std::array<int32_t, 2> slot_offsets = {{0, 0}};
  int32_t ubatch_count = 0;
  int32_t err = EMEL_OK;

  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 8,
    .slot_offsets_out = slot_offsets.data(),
    .slot_offsets_capacity = static_cast<int32_t>(slot_offsets.size()),
    .ubatch_count_out = &ubatch_count,
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
  CHECK(ubatch_count == 2);

  int32_t kv_tokens = 0;
  err = EMEL_OK;
  machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 0,
    .kv_tokens_out = &kv_tokens,
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
  CHECK(kv_tokens == 1);

  machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 1,
  });

  machine.process_event(emel::kv::cache::event::rollback{
    .from_ubatch_index = 1,
  });
}

TEST_CASE("kv_cache_sm_prepare_with_missing_outputs_and_validation_error") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 1> ubatch_sizes = {{1}};

  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 4,
  });

  int32_t err = EMEL_OK;
  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = 0,
    .requested_capacity = 4,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
