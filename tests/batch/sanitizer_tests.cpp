#include <array>
#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/batch/sanitizer/actions.hpp"
#include "emel/batch/sanitizer/sm.hpp"
#include "emel/batch/splitter/context.hpp"
#include "emel/emel.h"

TEST_CASE("batch_sanitizer_starts_initialized") {
  emel::batch::sanitizer::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::batch::sanitizer::initialized>));
}

TEST_CASE("batch_sanitizer_generates_defaults") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<int32_t, 3> seq_primary = {};
  std::array<uint64_t, 3 * emel::batch::splitter::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 9> positions = {};
  std::array<int8_t, 3> output_mask = {};
  int32_t outputs_total = 0;
  int32_t mask_words = 0;
  int32_t pos_count = 0;
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_rejects_invalid_seq_id") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{300, 0}};
  std::array<int32_t, 2> seq_primary = {};
  std::array<uint64_t, 2 * emel::batch::splitter::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 6> positions = {};
  std::array<int8_t, 2> output_mask = {};
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_rejects_decreasing_positions") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary = {};
  std::array<uint64_t, 2 * emel::batch::splitter::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 2> positions_in = {{1, 0}};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask = {};
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_overrides_output_mask_when_output_all") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary = {};
  std::array<uint64_t, 2 * emel::batch::splitter::action::SEQ_WORDS> seq_masks = {};
  std::array<int32_t, 2> positions = {};
  std::array<int8_t, 2> output_mask_in = {{1, 0}};
  std::array<int8_t, 2> output_mask = {};
  int32_t outputs_total = 0;
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_accepts_stride_three_positions_with_multiword_masks") {
  emel::batch::sanitizer::sm machine{};
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
  std::array<uint64_t, 2 * emel::batch::splitter::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int8_t, 2> output_mask_in = {{1, 0}};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t outputs_total = 0;
  int32_t seq_mask_words_out = 0;
  int32_t positions_count_out = 0;
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_rejects_primary_not_in_mask") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<uint64_t, 1> seq_masks_in = {{uint64_t{1} << 1}};
  std::array<int32_t, 1> seq_primary_in = {{2}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::splitter::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_rejects_empty_mask") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<uint64_t, 1> seq_masks_in = {{0U}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::splitter::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_enforces_single_output_per_seq") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::splitter::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask_in = {{1, 1}};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_rejects_non_contiguous_positions") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> seq_primary_in = {{0, 0}};
  std::array<int32_t, 2> seq_primary_out = {};
  std::array<uint64_t, 2 * emel::batch::splitter::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 2> positions_in = {{0, 2}};
  std::array<int32_t, 2> positions_out = {};
  std::array<int8_t, 2> output_mask_out = {};
  int32_t err = EMEL_OK;

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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

TEST_CASE("batch_sanitizer_handles_unexpected_event") {
  emel::batch::sanitizer::action::context ctx{};
  int32_t err = EMEL_OK;
  struct unexpected_event {
    int32_t * error_out = nullptr;
  };

  ctx.error_out = &err;
  emel::batch::sanitizer::action::on_unexpected(
    unexpected_event{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("batch_sanitizer_rejects_missing_error_out") {
  emel::batch::sanitizer::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> seq_primary_out = {};
  std::array<uint64_t, emel::batch::splitter::action::SEQ_WORDS> seq_masks_out = {};
  std::array<int32_t, 1> positions_out = {};
  std::array<int8_t, 1> output_mask_out = {};

  auto request = emel::batch::sanitizer::event::sanitize_decode{};
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
  CHECK(machine.is(boost::sml::state<emel::batch::sanitizer::errored>));
}

TEST_CASE("batch_sanitizer_ensure_last_error_sets_backend") {
  emel::batch::sanitizer::action::context ctx{};
  int32_t err = EMEL_OK;
  ctx.error_out = &err;
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;

  emel::batch::sanitizer::action::ensure_last_error(ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}
