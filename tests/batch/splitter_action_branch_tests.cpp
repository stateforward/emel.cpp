#include <array>
#include <doctest/doctest.h>

#include "emel/batch/splitter/guards.hpp"
#include "emel/callback.hpp"

namespace {

inline void noop_done(const emel::batch::splitter::events::splitting_done &) noexcept {}
inline void noop_error(const emel::batch::splitter::events::splitting_error &) noexcept {}

}  // namespace

TEST_CASE("batch_splitter_guard_callbacks_are_valid") {
  emel::batch::splitter::event::split request{};
  CHECK_FALSE(emel::batch::splitter::guard::callbacks_are_valid(request));

  request.on_done =
    emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
      &noop_done>();
  request.on_error =
    emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
      &noop_error>();
  CHECK(emel::batch::splitter::guard::callbacks_are_valid(request));
}

TEST_CASE("batch_splitter_guard_inputs_are_valid") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 1> tokens = {{1}};

  ctx.token_ids = nullptr;
  ctx.n_tokens = 0;
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  ctx.token_ids = tokens.data();
  ctx.n_tokens = emel::batch::splitter::action::MAX_UBATCHES + 1;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  ctx.n_tokens = 1;
  ctx.mode = static_cast<emel::batch::splitter::event::split_mode>(99);
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  CHECK(emel::batch::splitter::guard::inputs_are_valid(ctx));
}

TEST_CASE("batch_splitter_guard_mode_selection") {
  emel::batch::splitter::action::context ctx{};

  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  CHECK(emel::batch::splitter::guard::mode_is_simple(ctx));
  CHECK_FALSE(emel::batch::splitter::guard::mode_is_equal(ctx));
  CHECK_FALSE(emel::batch::splitter::guard::mode_is_seq(ctx));

  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  CHECK(emel::batch::splitter::guard::mode_is_equal(ctx));

  ctx.mode = emel::batch::splitter::event::split_mode::seq;
  CHECK(emel::batch::splitter::guard::mode_is_seq(ctx));
}

TEST_CASE("batch_splitter_guard_split_failed") {
  emel::batch::splitter::action::context ctx{};

  ctx.ubatch_count = 0;
  ctx.total_outputs = 1;
  CHECK(emel::batch::splitter::guard::split_failed(ctx));

  ctx.ubatch_count = 1;
  ctx.total_outputs = 1;
  CHECK_FALSE(emel::batch::splitter::guard::split_failed(ctx));
}
