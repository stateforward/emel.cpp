#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/detokenizer/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/renderer/sm.hpp"

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

bool detokenizer_bind_dispatch(
    void * detokenizer_sm,
    const emel::text::detokenizer::event::bind & ev) {
  if (detokenizer_sm == nullptr) {
    return false;
  }
  return static_cast<emel::text::detokenizer::sm *>(detokenizer_sm)->process_event(ev);
}

bool detokenizer_detokenize_dispatch(
    void * detokenizer_sm,
    const emel::text::detokenizer::event::detokenize & ev) {
  if (detokenizer_sm == nullptr) {
    return false;
  }
  return static_cast<emel::text::detokenizer::sm *>(detokenizer_sm)->process_event(ev);
}

bool detokenizer_bind_fail_no_error(
    void *,
    const emel::text::detokenizer::event::bind &) {
  return false;
}

bool detokenizer_bind_fail_with_error(
    void *,
    const emel::text::detokenizer::event::bind & ev) {
  ev.error_out = detokenizer_error_code(emel::text::detokenizer::error::model_invalid);
  return true;
}

bool detokenizer_detokenize_fail_no_error(
    void *,
    const emel::text::detokenizer::event::detokenize &) {
  return false;
}

bool detokenizer_detokenize_fail_with_error(
    void *,
    const emel::text::detokenizer::event::detokenize & ev) {
  ev.error_out = detokenizer_error_code(emel::text::detokenizer::error::model_invalid);
  return true;
}

bool detokenizer_detokenize_bad_output_length(
    void *,
    const emel::text::detokenizer::event::detokenize & ev) {
  ev.error_out = detokenizer_error_code(emel::text::detokenizer::error::none);
  ev.output_length_out = ev.output_capacity + 1;
  ev.pending_length_out = ev.pending_length;
  return true;
}

bool detokenizer_detokenize_bad_pending_length(
    void *,
    const emel::text::detokenizer::event::detokenize & ev) {
  ev.error_out = detokenizer_error_code(emel::text::detokenizer::error::none);
  ev.output_length_out = 0;
  ev.pending_length_out = ev.pending_capacity + 1;
  return true;
}

struct callback_recorder {
  int bind_done = 0;
  int bind_error = 0;
  int render_done = 0;
  int render_error = 0;
  int flush_done = 0;
  int flush_error = 0;
  size_t last_output_length = 0;
  emel::text::renderer::sequence_status last_status =
      emel::text::renderer::sequence_status::running;
  int32_t last_error = EMEL_OK;
};

bool on_bind_done(void * owner,
                  const emel::text::renderer::events::binding_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<callback_recorder *>(owner)->bind_done += 1;
  return true;
}

bool on_bind_error(void * owner,
                   const emel::text::renderer::events::binding_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->bind_error += 1;
  recorder->last_error = ev.err;
  return true;
}

bool on_render_done(void * owner,
                    const emel::text::renderer::events::rendering_done & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->render_done += 1;
  recorder->last_output_length = ev.output_length;
  recorder->last_status = ev.status;
  return true;
}

bool on_render_error(void * owner,
                     const emel::text::renderer::events::rendering_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->render_error += 1;
  recorder->last_error = ev.err;
  return true;
}

bool on_flush_done(void * owner,
                   const emel::text::renderer::events::flush_done & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->flush_done += 1;
  recorder->last_output_length = ev.output_length;
  recorder->last_status = ev.status;
  return true;
}

bool on_flush_error(void * owner,
                    const emel::text::renderer::events::flush_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->flush_error += 1;
  recorder->last_error = ev.err;
  return true;
}

bool bind_renderer(emel::text::renderer::sm & renderer,
                   emel::text::detokenizer::sm & detokenizer,
                   const emel::model::data::vocab & vocab,
                   const bool strip_leading_space,
                   const std::string_view * stop_sequences,
                   const size_t stop_count,
                   int32_t & err_out) {
  emel::text::renderer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.detokenizer_sm = &detokenizer;
  bind_ev.dispatch_detokenizer_bind = detokenizer_bind_dispatch;
  bind_ev.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  bind_ev.strip_leading_space = strip_leading_space;
  bind_ev.stop_sequences = stop_sequences;
  bind_ev.stop_sequence_count = stop_count;
  bind_ev.error_out = &err_out;
  return renderer.process_event(bind_ev);
}

}  // namespace

TEST_CASE("renderer_bind_render_and_flush_without_stop_sequences") {
  auto & vocab = make_vocab();
  const int32_t hi_id = add_token(vocab, "hi");
  CHECK(hi_id == 0);

  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};

  int32_t bind_err = EMEL_OK;
  CHECK(bind_renderer(renderer, detokenizer, vocab, false, nullptr, 0, bind_err));
  CHECK(bind_err == EMEL_OK);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::stop_sequence_matched;
  int32_t render_err = EMEL_OK;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = hi_id;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &render_err;

  CHECK(renderer.process_event(render_ev));
  CHECK(render_err == EMEL_OK);
  CHECK(output_length == 2);
  CHECK(status == emel::text::renderer::sequence_status::running);
  CHECK(std::string_view(output.data(), output_length) == "hi");

  output.fill('\0');
  output_length = 99;
  status = emel::text::renderer::sequence_status::stop_sequence_matched;
  int32_t flush_err = EMEL_OK;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == EMEL_OK);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::running);
}

TEST_CASE("renderer_bind_rejects_invalid_stop_sequences") {
  auto & vocab = make_vocab();
  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};

  const std::array<std::string_view, 1> invalid_stops = {
      std::string_view("0123456789012345678901234567890123456789")};

  int32_t bind_err = EMEL_OK;
  emel::text::renderer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.detokenizer_sm = &detokenizer;
  bind_ev.dispatch_detokenizer_bind = detokenizer_bind_dispatch;
  bind_ev.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  bind_ev.stop_sequences = invalid_stops.data();
  bind_ev.stop_sequence_count = invalid_stops.size();
  bind_ev.error_out = &bind_err;

  CHECK_FALSE(renderer.process_event(bind_ev));
  CHECK(bind_err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("renderer_handles_plamo2_byte_fallback_utf8") {
  auto & vocab = make_vocab();
  const int32_t b0 = add_token(vocab, "<0xE2>");
  const int32_t b1 = add_token(vocab, "<0x82>");
  const int32_t b2 = add_token(vocab, "<0xAC>");

  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};

  int32_t bind_err = EMEL_OK;
  CHECK(bind_renderer(renderer, detokenizer, vocab, false, nullptr, 0, bind_err));
  CHECK(bind_err == EMEL_OK);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = EMEL_OK;

  emel::text::renderer::event::render render_ev = {};
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  render_ev.token_id = b0;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 0);

  render_ev.token_id = b1;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 0);

  render_ev.token_id = b2;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 3);
  CHECK(static_cast<unsigned char>(output[0]) == 0xE2u);
  CHECK(static_cast<unsigned char>(output[1]) == 0x82u);
  CHECK(static_cast<unsigned char>(output[2]) == 0xACu);
  CHECK(status == emel::text::renderer::sequence_status::running);
}

TEST_CASE("renderer_stop_sequence_matches_across_token_boundary") {
  auto & vocab = make_vocab();
  const int32_t ab_id = add_token(vocab, "ab");
  const int32_t cd_id = add_token(vocab, "cd");
  const std::array<std::string_view, 1> stops = {"bc"};

  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};

  int32_t bind_err = EMEL_OK;
  CHECK(bind_renderer(renderer,
                      detokenizer,
                      vocab,
                      false,
                      stops.data(),
                      stops.size(),
                      bind_err));
  CHECK(bind_err == EMEL_OK);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = EMEL_OK;

  emel::text::renderer::event::render render_ev = {};
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  render_ev.token_id = ab_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 1);
  CHECK(std::string_view(output.data(), output_length) == "a");
  CHECK(status == emel::text::renderer::sequence_status::running);

  output.fill('\0');
  render_ev.token_id = cd_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);

  output.fill('\0');
  output_length = 0;
  int32_t flush_err = EMEL_OK;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == EMEL_OK);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);
}

TEST_CASE("renderer_flush_emits_holdback_when_no_stop_match") {
  auto & vocab = make_vocab();
  const int32_t ab_id = add_token(vocab, "ab");
  const std::array<std::string_view, 1> stops = {"xyz"};

  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};

  int32_t bind_err = EMEL_OK;
  CHECK(bind_renderer(renderer,
                      detokenizer,
                      vocab,
                      false,
                      stops.data(),
                      stops.size(),
                      bind_err));
  CHECK(bind_err == EMEL_OK);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = EMEL_OK;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = ab_id;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 0);

  output.fill('\0');
  output_length = 0;
  int32_t flush_err = EMEL_OK;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == EMEL_OK);
  CHECK(output_length == 2);
  CHECK(std::string_view(output.data(), output_length) == "ab");
  CHECK(status == emel::text::renderer::sequence_status::running);
}

TEST_CASE("renderer_strips_leading_whitespace_when_enabled") {
  auto & vocab = make_vocab();
  const int32_t spaced = add_token(vocab, "  hi");

  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};

  int32_t bind_err = EMEL_OK;
  CHECK(bind_renderer(renderer, detokenizer, vocab, true, nullptr, 0, bind_err));
  CHECK(bind_err == EMEL_OK);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = EMEL_OK;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = spaced;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  CHECK(renderer.process_event(render_ev));
  CHECK(err == EMEL_OK);
  CHECK(output_length == 2);
  CHECK(std::string_view(output.data(), output_length) == "hi");
}

TEST_CASE("renderer_dispatches_done_and_error_callbacks") {
  auto & vocab = make_vocab();
  const int32_t hi_id = add_token(vocab, "hi");

  emel::text::detokenizer::sm detokenizer{};
  emel::text::renderer::sm renderer{};
  callback_recorder recorder{};

  int32_t bind_err = EMEL_OK;
  emel::text::renderer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.detokenizer_sm = &detokenizer;
  bind_ev.dispatch_detokenizer_bind = detokenizer_bind_dispatch;
  bind_ev.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  bind_ev.error_out = &bind_err;
  bind_ev.owner_sm = &recorder;
  bind_ev.dispatch_done = on_bind_done;
  bind_ev.dispatch_error = on_bind_error;

  CHECK(renderer.process_event(bind_ev));
  CHECK(bind_err == EMEL_OK);
  CHECK(recorder.bind_done == 1);
  CHECK(recorder.bind_error == 0);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t render_err = EMEL_OK;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = hi_id;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &render_err;
  render_ev.owner_sm = &recorder;
  render_ev.dispatch_done = on_render_done;
  render_ev.dispatch_error = on_render_error;

  CHECK(renderer.process_event(render_ev));
  CHECK(render_err == EMEL_OK);
  CHECK(recorder.render_done == 1);
  CHECK(recorder.render_error == 0);
  CHECK(recorder.last_output_length == 2);
  CHECK(recorder.last_status == emel::text::renderer::sequence_status::running);

  emel::text::renderer::event::render bad_render_ev = {};
  bad_render_ev.token_id = hi_id;
  bad_render_ev.sequence_id = static_cast<int32_t>(emel::text::renderer::action::k_max_sequences);
  bad_render_ev.output = output.data();
  bad_render_ev.output_capacity = output.size();
  bad_render_ev.output_length_out = &output_length;
  bad_render_ev.status_out = &status;
  bad_render_ev.error_out = &render_err;
  bad_render_ev.owner_sm = &recorder;
  bad_render_ev.dispatch_done = on_render_done;
  bad_render_ev.dispatch_error = on_render_error;

  CHECK_FALSE(renderer.process_event(bad_render_ev));
  CHECK(render_err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(recorder.render_error == 1);

  size_t flush_length = 0;
  int32_t flush_err = EMEL_OK;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &flush_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;
  flush_ev.owner_sm = &recorder;
  flush_ev.dispatch_done = on_flush_done;
  flush_ev.dispatch_error = on_flush_error;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == EMEL_OK);
  CHECK(recorder.flush_done == 1);
  CHECK(recorder.flush_error == 0);
}

TEST_CASE("renderer_action_and_guard_paths") {
  auto & vocab = make_vocab();
  const int32_t token_id = add_token(vocab, "ab");
  const int32_t special_id = add_token(vocab, "<special>", 3);
  (void)token_id;
  int dummy = 0;
  void * dummy_ptr = &dummy;

  emel::text::renderer::action::context ctx = {};
  std::array<char, 8> output = {};

  emel::text::renderer::event::bind bind_ev = {};
  CHECK_FALSE(emel::text::renderer::guard::valid_bind{}(bind_ev));
  CHECK(emel::text::renderer::guard::invalid_bind{}(bind_ev));

  bind_ev.vocab = &vocab;
  bind_ev.detokenizer_sm = dummy_ptr;
  bind_ev.dispatch_detokenizer_bind = detokenizer_bind_dispatch;
  bind_ev.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  CHECK(emel::text::renderer::guard::valid_bind{}(bind_ev));

  std::array<std::string_view, 1> long_stop = {
      std::string_view("0123456789012345678901234567890123456789")};
  bind_ev.stop_sequences = long_stop.data();
  bind_ev.stop_sequence_count = long_stop.size();
  emel::text::renderer::action::begin_bind(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  bind_ev.stop_sequences = nullptr;
  bind_ev.stop_sequence_count = 0;
  emel::text::renderer::action::begin_bind(bind_ev, ctx);
  CHECK(ctx.vocab == &vocab);

  ctx.output = output.data();
  ctx.output_capacity = 0;
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'x';
  CHECK_FALSE(emel::text::renderer::action::compose_output(
      ctx.sequences[0], ctx, 1, 0, 0));
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx = emel::text::renderer::action::context{};
  ctx.output = output.data();
  ctx.output_capacity = output.size();
  ctx.stop_sequence_count = 1;
  ctx.stop_sequences[0].offset = 0;
  ctx.stop_sequences[0].length = 2;
  ctx.stop_storage[0] = 'b';
  ctx.stop_storage[1] = 'c';
  ctx.stop_max_length = 2;
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'b';
  output[0] = 'c';
  CHECK(emel::text::renderer::action::apply_stop_matching(
      ctx.sequences[0], ctx, 1));
  CHECK(ctx.status == emel::text::renderer::sequence_status::stop_sequence_matched);

  ctx = emel::text::renderer::action::context{};
  ctx.output = output.data();
  ctx.output_capacity = output.size();
  ctx.stop_sequence_count = 1;
  ctx.stop_sequences[0].offset = 0;
  ctx.stop_sequences[0].length = 2;
  ctx.stop_storage[0] = 'z';
  ctx.stop_storage[1] = 'z';
  ctx.stop_max_length = 2;
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'a';
  output[0] = 'b';
  CHECK(emel::text::renderer::action::apply_stop_matching(
      ctx.sequences[0], ctx, 1));
  CHECK(ctx.output_length == 1);

  ctx.vocab = nullptr;
  emel::text::renderer::action::bind_detokenizer(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = &vocab;
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.detokenizer_sm = dummy_ptr;
  ctx.dispatch_detokenizer_bind = detokenizer_bind_fail_no_error;
  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  emel::text::renderer::action::bind_detokenizer(ctx);
  CHECK(ctx.last_error ==
        detokenizer_error_code(emel::text::detokenizer::error::backend_error));

  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.dispatch_detokenizer_bind = detokenizer_bind_fail_with_error;
  emel::text::renderer::action::bind_detokenizer(ctx);
  CHECK(ctx.last_error ==
        detokenizer_error_code(emel::text::detokenizer::error::model_invalid));

  emel::text::detokenizer::sm detokenizer{};
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.detokenizer_sm = &detokenizer;
  ctx.dispatch_detokenizer_bind = detokenizer_bind_dispatch;
  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  emel::text::renderer::action::bind_detokenizer(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.is_bound);

  emel::text::renderer::event::render render_ev = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &out_len;
  render_ev.status_out = &status;
  render_ev.error_out = &err;
  render_ev.sequence_id = 0;
  render_ev.token_id = 0;
  emel::text::renderer::action::begin_render(render_ev, ctx);
  CHECK(ctx.output == output.data());

  ctx.is_bound = false;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.is_bound = true;
  ctx.vocab = &vocab;
  ctx.output = output.data();
  ctx.output_capacity = output.size();
  ctx.sequence_id = 0;
  ctx.token_id = 0;
  ctx.detokenizer_sm = dummy_ptr;
  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_fail_no_error;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.last_error ==
        detokenizer_error_code(emel::text::detokenizer::error::backend_error));

  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_fail_with_error;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.last_error ==
        detokenizer_error_code(emel::text::detokenizer::error::model_invalid));

  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  ctx.detokenizer_sm = &detokenizer;
  ctx.output = nullptr;
  ctx.output_capacity = 0;
  ctx.token_id = special_id;
  ctx.emit_special = false;
  ctx.sequences[0].pending_length = 0;
  ctx.sequences[0].holdback_length = 0;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.output_length == 0);

  ctx.output = output.data();
  ctx.output_capacity = output.size();
  ctx.token_id = token_id;
  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_bad_output_length;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_bad_pending_length;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.dispatch_detokenizer_detokenize = detokenizer_detokenize_dispatch;
  ctx.sequences[0].stop_matched = true;
  emel::text::renderer::action::run_render(ctx);
  CHECK(ctx.status == emel::text::renderer::sequence_status::stop_sequence_matched);
  CHECK(ctx.output_length == 0);
  ctx.sequences[0].stop_matched = false;

  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &out_len;
  flush_ev.status_out = &status;
  flush_ev.error_out = &err;
  emel::text::renderer::action::begin_flush(flush_ev, ctx);
  CHECK(ctx.sequence_id == 0);

  ctx.output = output.data();
  ctx.output_capacity = 0;
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'h';
  emel::text::renderer::action::run_flush(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.output_capacity = output.size();
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'h';
  emel::text::renderer::action::run_flush(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.output_length == 1);

  emel::text::renderer::action::reject_render(render_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  emel::text::renderer::action::reject_flush(flush_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  emel::text::renderer::action::reject_bind(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  emel::text::renderer::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  emel::text::renderer::action::mark_done(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  emel::text::renderer::action::on_unexpected(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  emel::text::renderer::action::clear_request(ctx);
  CHECK(ctx.output == nullptr);

  emel::text::renderer::event::render bad_render = {};
  CHECK_FALSE(emel::text::renderer::guard::valid_render{}(bad_render, ctx));
  CHECK(emel::text::renderer::guard::invalid_render{}(bad_render, ctx));
  ctx.is_bound = true;
  ctx.vocab = &vocab;
  bad_render.output = output.data();
  bad_render.output_capacity = output.size();
  bad_render.output_length_out = &out_len;
  bad_render.status_out = &status;
  bad_render.error_out = &err;
  bad_render.sequence_id = 0;
  CHECK(emel::text::renderer::guard::valid_render{}(bad_render, ctx));

  bad_render.output = nullptr;
  bad_render.output_capacity = 0;
  CHECK(emel::text::renderer::guard::valid_render{}(bad_render, ctx));

  emel::text::renderer::event::flush bad_flush = {};
  CHECK_FALSE(emel::text::renderer::guard::valid_flush{}(bad_flush, ctx));
  CHECK(emel::text::renderer::guard::invalid_flush{}(bad_flush, ctx));
  bad_flush.sequence_id = 0;
  bad_flush.output = output.data();
  bad_flush.output_capacity = output.size();
  bad_flush.output_length_out = &out_len;
  bad_flush.status_out = &status;
  bad_flush.error_out = &err;
  CHECK(emel::text::renderer::guard::valid_flush{}(bad_flush, ctx));

  ctx.phase_error = EMEL_OK;
  CHECK(emel::text::renderer::guard::phase_ok{}(ctx));
  CHECK_FALSE(emel::text::renderer::guard::phase_failed{}(ctx));
}
