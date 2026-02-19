#include <array>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"

namespace {

TEST_CASE("kv_cache_sm_invalid_prepare_sets_error") {
  emel::kv::cache::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = nullptr,
    .ubatch_count = 0,
    .requested_capacity = 0,
    .error_out = &err,
  });

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("kv_cache_sm_prepare_recovers_after_error") {
  emel::kv::cache::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = nullptr,
    .ubatch_count = 0,
    .requested_capacity = 0,
    .error_out = &err,
  });
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 1> sizes = {{1}};
  machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = sizes.data(),
    .ubatch_count = static_cast<int32_t>(sizes.size()),
    .requested_capacity = 4,
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
}

TEST_CASE("kv_cache_sm_apply_before_prepare_reports_backend_error") {
  emel::kv::cache::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 0,
    .error_out = &err,
  });

  CHECK(err == EMEL_ERR_BACKEND);
}

}  // namespace
