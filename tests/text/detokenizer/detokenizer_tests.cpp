#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/detokenizer/sm.hpp"

namespace {

int32_t add_token(emel::model::data::vocab & vocab,
                  const char * text,
                  const int32_t type = 0) {
  const uint32_t len = static_cast<uint32_t>(std::strlen(text));
  const uint32_t offset = vocab.token_bytes_used;
  std::memcpy(vocab.token_storage.data() + offset, text, len);
  const uint32_t id = vocab.n_tokens;
  vocab.entries[id].text_offset = offset;
  vocab.entries[id].text_length = len;
  vocab.entries[id].score = 0.0f;
  vocab.entries[id].type = type;
  vocab.token_bytes_used += len;
  vocab.n_tokens = id + 1;
  return static_cast<int32_t>(id);
}

emel::model::data::vocab & make_vocab() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::DEFAULT;
  return vocab;
}

constexpr int32_t detokenizer_error_code(const emel::text::detokenizer::error err) noexcept {
  return emel::text::detokenizer::error_code(err);
}

struct detokenizer_callback_recorder {
  int bind_done = 0;
  int bind_error = 0;
  int detokenize_done = 0;
  int detokenize_error = 0;
  int32_t last_error =
      detokenizer_error_code(emel::text::detokenizer::error::none);
  size_t last_output_length = 0;
  size_t last_pending_length = 0;
};

bool on_detok_bind_done(
    void * owner,
    const emel::text::detokenizer::events::binding_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<detokenizer_callback_recorder *>(owner)->bind_done += 1;
  return true;
}

bool on_detok_bind_error(
    void * owner,
    const emel::text::detokenizer::events::binding_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<detokenizer_callback_recorder *>(owner);
  recorder->bind_error += 1;
  recorder->last_error = ev.err;
  return true;
}

bool on_detok_detokenize_done(
    void * owner,
    const emel::text::detokenizer::events::detokenize_done & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<detokenizer_callback_recorder *>(owner);
  recorder->detokenize_done += 1;
  recorder->last_output_length = ev.output_length;
  recorder->last_pending_length = ev.pending_length;
  return true;
}

bool on_detok_detokenize_error(
    void * owner,
    const emel::text::detokenizer::events::detokenize_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<detokenizer_callback_recorder *>(owner);
  recorder->detokenize_error += 1;
  recorder->last_error = ev.err;
  return true;
}

}  // namespace

TEST_CASE("detokenizer_dispatches_error_callbacks_for_invalid_requests") {
  auto & vocab = make_vocab();
  const int32_t token_id = add_token(vocab, "A");
  const int32_t detok_none =
      detokenizer_error_code(emel::text::detokenizer::error::none);
  const int32_t detok_invalid_request =
      detokenizer_error_code(emel::text::detokenizer::error::invalid_request);

  emel::text::detokenizer::sm detokenizer{};
  detokenizer_callback_recorder callbacks = {};

  int32_t bind_ok = detok_none;
  emel::text::detokenizer::event::bind good_bind{
      vocab,
      bind_ok,
      &callbacks,
      on_detok_bind_done,
      on_detok_bind_error};
  CHECK(detokenizer.process_event(good_bind));
  CHECK(bind_ok == detok_none);
  CHECK(callbacks.bind_done == 1);
  CHECK(callbacks.bind_error == 0);

  std::array<uint8_t, 4> pending = {};
  std::array<char, 8> output = {};
  size_t output_length = 7;
  size_t pending_length = 7;
  int32_t detok_err = detok_none;

  emel::text::detokenizer::event::detokenize bad_detok{
      token_id,
      true,
      nullptr,
      0,
      pending.size(),
      output.data(),
      output.size(),
      output_length,
      pending_length,
      detok_err,
      &callbacks,
      on_detok_detokenize_done,
      on_detok_detokenize_error};

  CHECK_FALSE(detokenizer.process_event(bad_detok));
  CHECK(detok_err == detok_invalid_request);
  CHECK(output_length == 0);
  CHECK(pending_length == 0);
  CHECK(callbacks.detokenize_done == 0);
  CHECK(callbacks.detokenize_error == 1);
  CHECK(callbacks.last_error == detok_invalid_request);
}

TEST_CASE("detokenizer_internal_reentry_rejections_dispatch_error_callbacks") {
  namespace sml = boost::sml;

  auto & vocab = make_vocab();
  add_token(vocab, "B");

  const int32_t detok_none =
      detokenizer_error_code(emel::text::detokenizer::error::none);
  const int32_t detok_invalid_request =
      detokenizer_error_code(emel::text::detokenizer::error::invalid_request);

  emel::text::detokenizer::action::context ctx = {};
  boost::sml::sm<
      emel::text::detokenizer::model,
      boost::sml::testing> machine{ctx};
  detokenizer_callback_recorder callbacks = {};

  machine.set_current_states(sml::state<emel::text::detokenizer::binding_done_callback>);
  int32_t bind_err = detok_none;
  emel::text::detokenizer::event::bind bind_ev{
      vocab,
      bind_err,
      &callbacks,
      nullptr,
      on_detok_bind_error};

  CHECK_FALSE(machine.process_event(bind_ev));
  CHECK(bind_err == detok_invalid_request);
  CHECK(callbacks.bind_error == 1);
  CHECK(callbacks.last_error == detok_invalid_request);
  CHECK(machine.is(sml::state<emel::text::detokenizer::errored>));

  machine.set_current_states(sml::state<emel::text::detokenizer::detokenize_done_callback>);
  std::array<uint8_t, 4> pending = {};
  std::array<char, 8> output = {};
  size_t output_length = 9;
  size_t pending_length = 9;
  int32_t detok_err = detok_none;
  emel::text::detokenizer::event::detokenize detok_ev{
      0,
      true,
      pending.data(),
      0,
      pending.size(),
      output.data(),
      output.size(),
      output_length,
      pending_length,
      detok_err,
      &callbacks,
      nullptr,
      on_detok_detokenize_error};

  CHECK_FALSE(machine.process_event(detok_ev));
  CHECK(detok_err == detok_invalid_request);
  CHECK(callbacks.detokenize_error == 1);
  CHECK(callbacks.last_error == detok_invalid_request);
  CHECK(machine.is(sml::state<emel::text::detokenizer::errored>));
}

TEST_CASE("detokenizer_action_and_guard_paths") {
  using emel::text::detokenizer::action::context;
  using emel::text::detokenizer::action::detail::is_special_token_type;
  using emel::text::detokenizer::action::detail::is_utf8_continuation;
  using emel::text::detokenizer::action::detail::parse_hex_nibble;
  using emel::text::detokenizer::action::detail::parse_plamo2_byte_token;
  using emel::text::detokenizer::action::detail::utf8_sequence_length;

  uint8_t value = 0;
  CHECK(parse_hex_nibble('f', value));
  CHECK(value == 15);
  CHECK(parse_hex_nibble('A', value));
  CHECK(value == 10);
  CHECK_FALSE(parse_hex_nibble('z', value));

  CHECK(parse_plamo2_byte_token("<0x4A>", value));
  CHECK(value == 0x4A);
  CHECK_FALSE(parse_plamo2_byte_token("<bad>", value));

  CHECK(utf8_sequence_length(0x24u) == 1);
  CHECK(utf8_sequence_length(0xC2u) == 2);
  CHECK(utf8_sequence_length(0xE2u) == 3);
  CHECK(utf8_sequence_length(0xF0u) == 4);
  CHECK(utf8_sequence_length(0xFFu) == 0);

  CHECK(is_utf8_continuation(0x80u));
  CHECK_FALSE(is_utf8_continuation(0x41u));
  CHECK(is_special_token_type(3));
  CHECK_FALSE(is_special_token_type(0));

  auto & vocab = make_vocab();
  const int32_t plain_id = add_token(vocab, "A");
  const int32_t special_id = add_token(vocab, "<special>", 3);
  const int32_t byte_id = add_token(vocab, "<0x41>");

  context ctx = {};
  std::array<uint8_t, 4> pending = {};
  std::array<char, 8> output = {};
  size_t out_len = 0;
  size_t pending_len = 0;
  const int32_t detok_ok =
      detokenizer_error_code(emel::text::detokenizer::error::none);
  const int32_t detok_invalid_request =
      detokenizer_error_code(emel::text::detokenizer::error::invalid_request);
  const int32_t detok_model_invalid =
      detokenizer_error_code(emel::text::detokenizer::error::model_invalid);
  const int32_t detok_backend_error =
      detokenizer_error_code(emel::text::detokenizer::error::backend_error);
  int32_t err = detok_ok;

  emel::text::detokenizer::event::detokenize detok_ev{
      plain_id,
      true,
      pending.data(),
      0,
      pending.size(),
      output.data(),
      output.size(),
      out_len,
      pending_len,
      err};

  CHECK(emel::text::detokenizer::action::write_bytes(detok_ev, out_len, pending_len, "", 0));
  CHECK(out_len == 0);
  detok_ev.output = nullptr;
  CHECK_FALSE(emel::text::detokenizer::action::write_bytes(
      detok_ev, out_len, pending_len, "x", 1));
  CHECK(err == detok_invalid_request);
  detok_ev.output = output.data();
  err = detok_ok;

  pending[0] = 0xFFu;
  pending_len = 1;
  CHECK(emel::text::detokenizer::guard::detokenize_pending_head_invalid{}(detok_ev));

  err = detok_ok;
  out_len = 0;
  pending[0] = 0xE2u;
  pending_len = 1;
  CHECK(emel::text::detokenizer::guard::detokenize_pending_head_incomplete{}(detok_ev));

  err = detok_ok;
  out_len = 0;
  pending[0] = 0x41u;
  pending_len = 1;
  CHECK(emel::text::detokenizer::guard::detokenize_pending_head_complete{}(detok_ev));
  emel::text::detokenizer::action::write_pending_head_sequence(detok_ev);
  CHECK(err == detok_ok);
  CHECK(out_len == 1);
  CHECK(output[0] == 'A');
  CHECK(pending_len == 0);

  err = detok_ok;
  out_len = 0;
  pending[0] = 0xE2u;
  pending[1] = 0x80u;
  pending[2] = 0x20u;
  pending_len = 3;
  CHECK(emel::text::detokenizer::guard::detokenize_pending_head_invalid{}(detok_ev));

  emel::text::detokenizer::event::bind bind_ev{vocab, err};
  err = detok_ok;
  emel::text::detokenizer::action::begin_bind(bind_ev, ctx);
  CHECK(ctx.vocab == &vocab);
  CHECK_FALSE(ctx.is_bound);
  CHECK(err == detok_ok);

  ctx.vocab = &vocab;
  emel::text::detokenizer::action::commit_bind(bind_ev, ctx);
  CHECK(err == detok_ok);
  CHECK(ctx.is_bound);

  emel::text::detokenizer::action::reject_bind(bind_ev, ctx);
  CHECK(err == detok_invalid_request);
  CHECK_FALSE(ctx.is_bound);

  detok_ev.token_id = special_id;
  detok_ev.emit_special = false;
  detok_ev.pending_length = 0;
  out_len = 99;
  pending_len = 99;
  err = detok_ok;
  emel::text::detokenizer::action::begin_detokenize(detok_ev);
  CHECK(out_len == 0);
  CHECK(pending_len == 0);
  CHECK(err == detok_ok);

  emel::text::detokenizer::sm unbound_detokenizer{};
  detok_ev.token_id = plain_id;
  detok_ev.emit_special = true;
  detok_ev.pending_length = 0;
  out_len = 99;
  pending_len = 99;
  err = detok_ok;
  CHECK_FALSE(unbound_detokenizer.process_event(detok_ev));
  CHECK(err == detok_invalid_request);

  emel::text::detokenizer::sm detokenizer{};
  int32_t bind_sm_err = detok_ok;
  emel::text::detokenizer::event::bind bind_sm_ev{vocab, bind_sm_err};
  CHECK(detokenizer.process_event(bind_sm_ev));
  CHECK(bind_sm_err == detok_ok);

  detok_ev.token_id = 999;
  detok_ev.emit_special = true;
  detok_ev.pending_length = 0;
  out_len = 99;
  pending_len = 99;
  err = detok_ok;
  CHECK_FALSE(detokenizer.process_event(detok_ev));
  CHECK(err == detok_model_invalid);

  detok_ev.token_id = special_id;
  detok_ev.emit_special = false;
  detok_ev.pending_length = 0;
  out_len = 99;
  pending_len = 99;
  err = detok_ok;
  CHECK(detokenizer.process_event(detok_ev));
  CHECK(err == detok_ok);
  CHECK(out_len == 0);
  CHECK(pending_len == 0);

  detok_ev.token_id = byte_id;
  detok_ev.emit_special = true;
  detok_ev.pending_length = detok_ev.pending_capacity;
  out_len = 0;
  pending_len = detok_ev.pending_capacity;
  err = detok_ok;
  CHECK_FALSE(detokenizer.process_event(detok_ev));
  CHECK(err == detok_invalid_request);

  detok_ev.token_id = plain_id;
  detok_ev.pending_length = 1;
  pending[0] = 0xE2u;
  out_len = 0;
  pending_len = 1;
  err = detok_ok;
  CHECK_FALSE(detokenizer.process_event(detok_ev));
  CHECK(err == detok_invalid_request);

  detok_ev.pending_length = 0;
  detok_ev.output_capacity = 0;
  out_len = 0;
  pending_len = 0;
  err = detok_ok;
  CHECK_FALSE(detokenizer.process_event(detok_ev));
  CHECK(err == detok_invalid_request);

  detok_ev.output_capacity = output.size();
  out_len = 0;
  pending_len = 0;
  err = detok_ok;
  CHECK(detokenizer.process_event(detok_ev));
  CHECK(err == detok_ok);
  CHECK(out_len == 1);
  CHECK(pending_len == 0);
  CHECK(output[0] == 'A');

  emel::text::detokenizer::action::reject_detokenize(detok_ev);
  CHECK(err == detok_invalid_request);
  err = detok_ok;
  emel::text::detokenizer::action::mark_done(detok_ev);
  CHECK(err == detok_ok);
  out_len = 7;
  pending_len = 7;
  err = detok_ok;
  detok_ev.pending_length = 2;
  emel::text::detokenizer::action::on_unexpected(bind_ev, ctx);
  CHECK(err == detok_ok);
  emel::text::detokenizer::action::on_unexpected(detok_ev, ctx);
  CHECK(err == detok_ok);
  CHECK(out_len == 7);
  CHECK(pending_len == 7);

  const auto * saved_vocab = ctx.vocab;
  const bool saved_is_bound = ctx.is_bound;
  emel::text::detokenizer::action::clear_request(ctx);
  CHECK(ctx.vocab == saved_vocab);
  CHECK(ctx.is_bound == saved_is_bound);

  CHECK(emel::text::detokenizer::guard::valid_bind{}(bind_ev));
  CHECK_FALSE(emel::text::detokenizer::guard::invalid_bind{}(bind_ev));

  emel::text::detokenizer::event::detokenize bad_detok{
      plain_id,
      true,
      nullptr,
      0,
      pending.size(),
      output.data(),
      output.size(),
      out_len,
      pending_len,
      err};
  CHECK_FALSE(emel::text::detokenizer::guard::valid_detokenize{}(bad_detok, ctx));
  CHECK(emel::text::detokenizer::guard::invalid_detokenize{}(bad_detok, ctx));
  ctx.vocab = &vocab;
  ctx.is_bound = true;
  bad_detok.pending_bytes = pending.data();
  CHECK(emel::text::detokenizer::guard::valid_detokenize{}(bad_detok, ctx));
  CHECK(emel::text::detokenizer::guard::detokenize_token_in_vocab{}(bad_detok, ctx));
  CHECK_FALSE(emel::text::detokenizer::guard::detokenize_token_out_of_vocab{}(bad_detok, ctx));

  bad_detok.token_id = special_id;
  bad_detok.emit_special = false;
  CHECK(emel::text::detokenizer::guard::detokenize_skip_special_piece{}(bad_detok, ctx));
  CHECK_FALSE(emel::text::detokenizer::guard::detokenize_byte_piece{}(bad_detok, ctx));
  CHECK_FALSE(emel::text::detokenizer::guard::detokenize_text_piece{}(bad_detok, ctx));

  bad_detok.token_id = byte_id;
  bad_detok.emit_special = true;
  pending_len = 0;
  CHECK(emel::text::detokenizer::guard::detokenize_byte_piece{}(bad_detok, ctx));
  CHECK(emel::text::detokenizer::guard::detokenize_pending_has_capacity_for_byte{}(bad_detok, ctx));
  pending_len = bad_detok.pending_capacity;
  CHECK(emel::text::detokenizer::guard::detokenize_pending_no_capacity_for_byte{}(bad_detok, ctx));

  bad_detok.token_id = plain_id;
  pending_len = 0;
  CHECK(emel::text::detokenizer::guard::detokenize_text_piece{}(bad_detok, ctx));

  err = detok_ok;
  pending_len = 0;
  CHECK(emel::text::detokenizer::guard::bind_phase_ok{}(bind_ev));
  CHECK(emel::text::detokenizer::guard::detokenize_phase_ok{}(bad_detok));
  CHECK(emel::text::detokenizer::guard::detokenize_pending_empty{}(bad_detok));
  pending_len = 1;
  CHECK(emel::text::detokenizer::guard::detokenize_pending_not_empty{}(bad_detok));
  err = detok_backend_error;
  CHECK(emel::text::detokenizer::guard::bind_phase_failed{}(bind_ev));
  CHECK(emel::text::detokenizer::guard::detokenize_phase_failed{}(bad_detok));
}
