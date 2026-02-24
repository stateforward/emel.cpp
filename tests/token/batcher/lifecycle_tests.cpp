#include <array>
#include <boost/sml.hpp>
#include <doctest/doctest.h>
#include <limits>

#include "emel/token/batcher/actions.hpp"
#include "emel/token/batcher/sm.hpp"
#include "emel/batch/planner/context.hpp"
#include "emel/emel.h"

namespace {

struct seed_lookup_state {
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seeds = {};
  bool fail = false;
};

struct batch_callback_capture {
  int done = 0;
  int error = 0;
  int32_t last_err = EMEL_OK;

  void on_done(const emel::token::batcher::events::batch_done &) noexcept { done += 1; }

  void on_error(const emel::token::batcher::events::batch_error & ev) noexcept {
    error += 1;
    last_err = ev.err;
  }
};

bool resolve_seed_position(
    void * ctx,
    const int32_t seq_id,
    int32_t * position_out) noexcept {
  if (ctx == nullptr || position_out == nullptr || seq_id < 0 ||
      seq_id >= emel::batch::planner::action::MAX_SEQ) {
    return false;
  }
  const auto * state = static_cast<const seed_lookup_state *>(ctx);
  if (state->fail) {
    return false;
  }
  *position_out = state->seeds[static_cast<size_t>(seq_id)];
  return true;
}

struct unknown_event {};

}  // namespace

TEST_CASE("token_batcher_starts_initialized") {
  emel::token::batcher::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::token::batcher::initialized>));
}

TEST_CASE("token_batcher_generates_defaults") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<int32_t, 3> seq_primary = {};
  std::array<uint64_t, 3 * emel::batch::planner::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 9> positions = {};
  std::array<int8_t, 3> output_mask = {};
  int32_t outputs_total = 0;
  int32_t mask_words = 0;
  int32_t pos_count = 0;
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary.size());
  request.seq_masks_out = seq_masks.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks.size());
  request.positions_out = positions.data();
  request.positions_capacity = static_cast<int32_t>(positions.size());
  request.output_mask_out = output_mask.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask.size());
  request.outputs_total_out = &outputs_total;
  request.seq_mask_words_out = &mask_words;
  request.positions_count_out = &pos_count;
  request.error_out = &err;

  CHECK(machine.process_event(request));

  CHECK(err == EMEL_OK);
  CHECK(outputs_total == 1);
  CHECK(output_mask[0] == 0);
  CHECK(output_mask[1] == 0);
  CHECK(output_mask[2] == 1);
  CHECK(seq_primary[0] == 0);
  CHECK(seq_primary[1] == 0);
  CHECK(seq_primary[2] == 0);
  CHECK(mask_words == 1);
  CHECK(pos_count == 3);
  CHECK(positions[0] == 0);
  CHECK(positions[1] == 1);
  CHECK(positions[2] == 2);
}

TEST_CASE("token_batcher_rejects_token_out_of_vocab_bounds") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 9}};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.vocab_size = 8;
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_autopopulates_positions_from_seed_lookup") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<int32_t, 3> seq_primary_in = {{5, 5, 5}};
  std::array<int32_t, 3> seq_primary_out = {};
  std::array<uint64_t, 3 * emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 3> positions_out = {};
  std::array<int8_t, 3> output_mask_out = {};
  seed_lookup_state seed_state{};
  int32_t err = EMEL_OK;

  seed_state.seeds[5] = 41;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_mask_words = 1;
  request.position_seed_ctx = &seed_state;
  request.resolve_position_seed = resolve_seed_position;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(positions_out[0] == 41);
  CHECK(positions_out[1] == 42);
  CHECK(positions_out[2] == 43);
}

TEST_CASE("token_batcher_allows_first_seen_coupled_seq_without_seed_callback") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> seq_masks_in = {{uint64_t{1}, uint64_t{3}}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_masks = seq_masks_in.data();
  request.seq_mask_words = 1;
  request.seq_masks_count = static_cast<int32_t>(seq_masks_in.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(positions_out[0] == 0);
  CHECK(positions_out[1] == 1);
}

TEST_CASE("token_batcher_rejects_diverged_coupled_seed_positions") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<uint64_t, 1> seq_masks_in = {{(uint64_t{1} << 0) | (uint64_t{1} << 1)}};
  std::array<int32_t, 1> seq_primary_in = {{0}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  seed_lookup_state seed_state{};
  int32_t err = EMEL_OK;

  seed_state.seeds[0] = 7;
  seed_state.seeds[1] = 8;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_masks = seq_masks_in.data();
  request.seq_mask_words = 1;
  request.seq_masks_count = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.position_seed_ctx = &seed_state;
  request.resolve_position_seed = resolve_seed_position;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_reports_seed_lookup_failure") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> seq_primary_in = {{0}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  seed_lookup_state seed_state{};
  int32_t err = EMEL_OK;

  seed_state.fail = true;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.position_seed_ctx = &seed_state;
  request.resolve_position_seed = resolve_seed_position;
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("token_batcher_rejects_seed_position_overflow") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> seq_primary_in = {{0}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  seed_lookup_state seed_state{};
  int32_t err = EMEL_OK;

  seed_state.seeds[0] = std::numeric_limits<int32_t>::max();

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.position_seed_ctx = &seed_state;
  request.resolve_position_seed = resolve_seed_position;
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_rejects_invalid_seq_id") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{300, 0}};
  std::array<int32_t, 2> seq_primary = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 6> positions = {};
  std::array<int8_t, 2> output_mask = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary.size());
  request.seq_masks_out = seq_masks.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks.size());
  request.positions_out = positions.data();
  request.positions_capacity = static_cast<int32_t>(positions.size());
  request.output_mask_out = output_mask.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_rejects_decreasing_positions") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 2> positions_in = {{1, 0}};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_mask_words = 1;
  request.positions = positions_in.data();
  request.positions_count = static_cast<int32_t>(positions_in.size());
  request.seq_primary_ids_out = seq_primary.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary.size());
  request.seq_masks_out = seq_masks.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_overrides_output_mask_when_output_all") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 2> positions = {};
  std::array<int8_t, 2> output_mask_in = {{1, 0}};
  std::array<int8_t, 2> output_mask = {};
  int32_t outputs_total = 0;
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_mask_words = 1;
  request.output_mask = output_mask_in.data();
  request.output_mask_count = static_cast<int32_t>(output_mask_in.size());
  request.output_all = true;
  request.seq_primary_ids_out = seq_primary.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary.size());
  request.seq_masks_out = seq_masks.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks.size());
  request.positions_out = positions.data();
  request.positions_capacity = static_cast<int32_t>(positions.size());
  request.output_mask_out = output_mask.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask.size());
  request.outputs_total_out = &outputs_total;
  request.error_out = &err;

  CHECK(machine.process_event(request));

  CHECK(err == EMEL_OK);
  CHECK(output_mask[0] == 1);
  CHECK(output_mask[1] == 1);
  CHECK(outputs_total == 2);
}

TEST_CASE("token_batcher_accepts_stride_three_positions_with_multiword_masks") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 4> seq_masks_in = {
    0U,
    (uint64_t{1} << 3),
    0U,
    (uint64_t{1} << 4),
  };
  std::array<int32_t, 2> seq_primary_in = {{67, 68}};
  std::array<int32_t, 6> positions_in = {{0, 0, 0, 1, 1, 1}};
  std::array<int32_t, 6> positions_out = {};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int8_t, 2> output_mask_in = {{1, 0}};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t outputs_total = 0;
  int32_t seq_mask_words_out = 0;
  int32_t positions_count_out = 0;
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_masks = seq_masks_in.data();
  request.seq_mask_words = 2;
  request.seq_masks_count = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.positions = positions_in.data();
  request.positions_count = static_cast<int32_t>(positions_in.size());
  request.output_mask = output_mask_in.data();
  request.output_mask_count = static_cast<int32_t>(output_mask_in.size());
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.outputs_total_out = &outputs_total;
  request.seq_mask_words_out = &seq_mask_words_out;
  request.positions_count_out = &positions_count_out;
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(outputs_total == 1);
  CHECK(seq_mask_words_out == 2);
  CHECK(positions_count_out == 6);
  CHECK(positions_out[5] == 1);
  CHECK(seq_primary_out[0] == 67);
  CHECK(seq_primary_out[1] == 68);
  CHECK(output_mask_out[0] == 1);
  CHECK(output_mask_out[1] == 0);
}

TEST_CASE("token_batcher_rejects_primary_not_in_mask") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<uint64_t, 1> seq_masks_in = {{uint64_t{1} << 1}};
  std::array<int32_t, 1> seq_primary_in = {{2}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_masks = seq_masks_in.data();
  request.seq_mask_words = 1;
  request.seq_masks_count = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_rejects_empty_mask") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<uint64_t, 1> seq_masks_in = {{0U}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_masks = seq_masks_in.data();
  request.seq_mask_words = 1;
  request.seq_masks_count = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_enforces_single_output_per_seq") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask_in = {{1, 1}};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_mask_words = 1;
  request.output_mask = output_mask_in.data();
  request.output_mask_count = static_cast<int32_t>(output_mask_in.size());
  request.enforce_single_output_per_seq = true;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_rejects_non_contiguous_positions") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 2> positions_in = {{0, 2}};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_primary_ids = seq_primary_in.data();
  request.seq_primary_ids_count = static_cast<int32_t>(seq_primary_in.size());
  request.seq_mask_words = 1;
  request.positions = positions_in.data();
  request.positions_count = static_cast<int32_t>(positions_in.size());
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_handles_unexpected_event") {
  emel::token::batcher::action::context ctx{};
  int32_t err = EMEL_OK;
  struct unexpected_event {
    int32_t * error_out = nullptr;
  };

  ctx.error_out = &err;
  emel::token::batcher::action::on_unexpected(
    unexpected_event{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("token_batcher_rejects_missing_error_out") {
  emel::token::batcher::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());

  CHECK(machine.process_event(request));
  CHECK(machine.is(boost::sml::state<emel::token::batcher::errored>));
}

TEST_CASE("token_batcher_ensure_last_error_sets_backend") {
  emel::token::batcher::action::context ctx{};
  int32_t err = EMEL_OK;
  ctx.error_out = &err;
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;

  emel::token::batcher::action::ensure_last_error(ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("token_batcher_dispatches_callbacks_synchronously") {
  emel::token::batcher::sm machine{};
  batch_callback_capture capture{};

  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::planner::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::token::batcher::event::batch{};
  request.token_ids = tokens.data();
  request.n_tokens = static_cast<int32_t>(tokens.size());
  request.seq_mask_words = 1;
  request.seq_primary_ids_out = seq_primary_out.data();
  request.seq_primary_ids_capacity = static_cast<int32_t>(seq_primary_out.size());
  request.seq_masks_out = seq_masks_out.data();
  request.seq_masks_capacity = static_cast<int32_t>(seq_masks_out.size());
  request.positions_out = positions_out.data();
  request.positions_capacity = static_cast<int32_t>(positions_out.size());
  request.output_mask_out = output_mask_out.data();
  request.output_mask_capacity = static_cast<int32_t>(output_mask_out.size());
  request.error_out = &err;
  request.on_done =
      emel::callback<void(const emel::token::batcher::events::batch_done &)>::from<
          batch_callback_capture, &batch_callback_capture::on_done>(&capture);
  request.on_error =
      emel::callback<void(const emel::token::batcher::events::batch_error &)>::from<
          batch_callback_capture, &batch_callback_capture::on_error>(&capture);

  CHECK(machine.process_event(request));
  CHECK(capture.done == 1);
  CHECK(capture.error == 0);
  CHECK(err == EMEL_OK);

  request.error_out = nullptr;
  CHECK(machine.process_event(request));
  CHECK(capture.done == 1);
  CHECK(capture.error == 1);
  CHECK(capture.last_err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token_batcher_routes_unexpected_event") {
  emel::token::batcher::sm machine{};
  CHECK(machine.process_event(unknown_event{}));
  CHECK(machine.is(boost::sml::state<emel::token::batcher::unexpected>));
}
