#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/detokenizer/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/renderer/errors.hpp"
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

constexpr emel::error::type renderer_error_type(
    const emel::text::renderer::error err) noexcept {
  return emel::error::cast(err);
}

constexpr int32_t renderer_error_code(
    const emel::text::renderer::error err) noexcept {
  return static_cast<int32_t>(renderer_error_type(err));
}

constexpr int32_t k_renderer_ok =
    renderer_error_code(emel::text::renderer::error::none);
constexpr int32_t k_renderer_invalid_request =
    renderer_error_code(emel::text::renderer::error::invalid_request);
constexpr int32_t k_renderer_model_invalid =
    renderer_error_code(emel::text::renderer::error::model_invalid);
constexpr int32_t k_detok_ok = static_cast<int32_t>(
    emel::error::cast(emel::text::detokenizer::error::none));
constexpr int32_t k_detok_backend_error = static_cast<int32_t>(
    emel::error::cast(emel::text::detokenizer::error::backend_error));
constexpr int32_t k_detok_model_invalid = static_cast<int32_t>(
    emel::error::cast(emel::text::detokenizer::error::model_invalid));

static_assert(
    std::is_reference_v<decltype(std::declval<const emel::text::renderer::event::initialize &>().vocab)>);
static_assert(std::is_same_v<
              std::remove_reference_t<
                  decltype(std::declval<emel::text::renderer::action::context &>().detokenizer)>,
              emel::text::detokenizer::sm>);

struct callback_recorder {
  int initialize_done_count = 0;
  int initialize_error_count = 0;
  int render_done = 0;
  int render_error = 0;
  int flush_done = 0;
  int flush_error = 0;
  size_t last_output_length = 0;
  emel::text::renderer::sequence_status last_status =
      emel::text::renderer::sequence_status::running;
  int32_t last_error = k_renderer_ok;
};

bool on_initialize_done(void * owner,
                  const emel::text::renderer::events::initialize_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<callback_recorder *>(owner)->initialize_done_count += 1;
  return true;
}

bool on_initialize_error(void * owner,
                   const emel::text::renderer::events::initialize_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->initialize_error_count += 1;
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

bool initialize_renderer(emel::text::renderer::sm & renderer,
                   const emel::model::data::vocab & vocab,
                   const bool strip_leading_space,
                   const std::string_view * stop_sequences,
                   const size_t stop_count,
                   int32_t & err_out) {
  emel::text::renderer::event::initialize initialize_ev{vocab};
  initialize_ev.strip_leading_space = strip_leading_space;
  initialize_ev.stop_sequences = stop_sequences;
  initialize_ev.stop_sequence_count = stop_count;
  initialize_ev.error_out = &err_out;
  return renderer.process_event(initialize_ev);
}

}  // namespace

TEST_CASE("renderer_initialize_render_and_flush_without_stop_sequences") {
  auto & vocab = make_vocab();
  const int32_t hi_id = add_token(vocab, "hi");
  CHECK(hi_id == 0);

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer, vocab, false, nullptr, 0, initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::stop_sequence_matched;
  int32_t render_err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = hi_id;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &render_err;

  CHECK(renderer.process_event(render_ev));
  CHECK(render_err == k_renderer_ok);
  CHECK(output_length == 2);
  CHECK(status == emel::text::renderer::sequence_status::running);
  CHECK(std::string_view(output.data(), output_length) == "hi");

  output.fill('\0');
  output_length = 99;
  status = emel::text::renderer::sequence_status::stop_sequence_matched;
  int32_t flush_err = k_renderer_ok;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == k_renderer_ok);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::running);
}

TEST_CASE("renderer_initialize_succeeds_without_external_detokenizer_injection") {
  auto & vocab = make_vocab();
  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  emel::text::renderer::event::initialize initialize_ev{vocab};
  initialize_ev.error_out = &initialize_err;

  CHECK(renderer.process_event(initialize_ev));
  CHECK(initialize_err == k_renderer_ok);
}

TEST_CASE("renderer_initialize_rejects_invalid_stop_sequences") {
  auto & vocab = make_vocab();
  emel::text::renderer::sm renderer{};

  const std::array<std::string_view, 1> invalid_stops = {
      std::string_view("0123456789012345678901234567890123456789")};

  int32_t initialize_err = k_renderer_ok;
  emel::text::renderer::event::initialize initialize_ev{vocab};
  initialize_ev.stop_sequences = invalid_stops.data();
  initialize_ev.stop_sequence_count = invalid_stops.size();
  initialize_ev.error_out = &initialize_err;

  CHECK_FALSE(renderer.process_event(initialize_ev));
  CHECK(initialize_err == k_renderer_invalid_request);
}

TEST_CASE("renderer_handles_plamo2_byte_fallback_utf8") {
  auto & vocab = make_vocab();
  const int32_t b0 = add_token(vocab, "<0xE2>");
  const int32_t b1 = add_token(vocab, "<0x82>");
  const int32_t b2 = add_token(vocab, "<0xAC>");

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer, vocab, false, nullptr, 0, initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  render_ev.token_id = b0;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 0);

  render_ev.token_id = b1;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 0);

  render_ev.token_id = b2;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
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

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer,
                      vocab,
                      false,
                      stops.data(),
                      stops.size(),
                      initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  render_ev.token_id = ab_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 1);
  CHECK(std::string_view(output.data(), output_length) == "a");
  CHECK(status == emel::text::renderer::sequence_status::running);

  output.fill('\0');
  render_ev.token_id = cd_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);

  output.fill('\0');
  output_length = 0;
  int32_t flush_err = k_renderer_ok;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == k_renderer_ok);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);
}

TEST_CASE("renderer_stop_sequence_state_latches_across_calls") {
  auto & vocab = make_vocab();
  const int32_t ab_id = add_token(vocab, "ab");
  const int32_t cd_id = add_token(vocab, "cd");
  const std::array<std::string_view, 1> stops = {"bc"};

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer,
                           vocab,
                           false,
                           stops.data(),
                           stops.size(),
                           initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  render_ev.token_id = ab_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 1);
  CHECK(std::string_view(output.data(), output_length) == "a");
  CHECK(status == emel::text::renderer::sequence_status::running);

  output.fill('\0');
  render_ev.token_id = cd_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);

  output.fill('\0');
  output_length = 99;
  status = emel::text::renderer::sequence_status::running;
  err = k_renderer_ok;
  render_ev.token_id = ab_id;
  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);
}

TEST_CASE("renderer_flush_emits_holdback_when_no_stop_match") {
  auto & vocab = make_vocab();
  const int32_t ab_id = add_token(vocab, "ab");
  const std::array<std::string_view, 1> stops = {"xyz"};

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer,
                      vocab,
                      false,
                      stops.data(),
                      stops.size(),
                      initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = ab_id;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 0);

  output.fill('\0');
  output_length = 0;
  int32_t flush_err = k_renderer_ok;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK(renderer.process_event(flush_ev));
  CHECK(flush_err == k_renderer_ok);
  CHECK(output_length == 2);
  CHECK(std::string_view(output.data(), output_length) == "ab");
  CHECK(status == emel::text::renderer::sequence_status::running);
}

TEST_CASE("renderer_strips_leading_whitespace_when_enabled") {
  auto & vocab = make_vocab();
  const int32_t spaced = add_token(vocab, "  hi");

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer, vocab, true, nullptr, 0, initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = spaced;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &err;

  CHECK(renderer.process_event(render_ev));
  CHECK(err == k_renderer_ok);
  CHECK(output_length == 2);
  CHECK(std::string_view(output.data(), output_length) == "hi");
}

TEST_CASE("renderer_dispatches_done_and_error_callbacks") {
  auto & vocab = make_vocab();
  const int32_t hi_id = add_token(vocab, "hi");

  emel::text::renderer::sm renderer{};
  callback_recorder recorder{};

  int32_t initialize_err = k_renderer_ok;
  emel::text::renderer::event::initialize initialize_ev{vocab};
  initialize_ev.error_out = &initialize_err;
  initialize_ev.owner_sm = &recorder;
  initialize_ev.dispatch_done = on_initialize_done;
  initialize_ev.dispatch_error = on_initialize_error;

  CHECK(renderer.process_event(initialize_ev));
  CHECK(initialize_err == k_renderer_ok);
  CHECK(recorder.initialize_done_count == 1);
  CHECK(recorder.initialize_error_count == 0);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t render_err = k_renderer_ok;

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
  CHECK(render_err == k_renderer_ok);
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
  CHECK(render_err == k_renderer_invalid_request);
  CHECK(recorder.render_error == 1);

  size_t flush_length = 0;
  int32_t flush_err = k_renderer_ok;
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
  CHECK(flush_err == k_renderer_ok);
  CHECK(recorder.flush_done == 1);
  CHECK(recorder.flush_error == 0);
}

TEST_CASE("renderer_invalid_render_and_flush_set_output_status_defaults") {
  auto & vocab = make_vocab();
  add_token(vocab, "hi");

  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer, vocab, false, nullptr, 0, initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 123;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::stop_sequence_matched;
  int32_t render_err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = 0;
  render_ev.sequence_id =
      static_cast<int32_t>(emel::text::renderer::action::k_max_sequences);
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &render_err;

  CHECK_FALSE(renderer.process_event(render_ev));
  CHECK(render_err == k_renderer_invalid_request);
  CHECK(output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::running);

  size_t flush_output_length = 123;
  status = emel::text::renderer::sequence_status::stop_sequence_matched;
  int32_t flush_err = k_renderer_ok;
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id =
      static_cast<int32_t>(emel::text::renderer::action::k_max_sequences);
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &flush_output_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;

  CHECK_FALSE(renderer.process_event(flush_ev));
  CHECK(flush_err == k_renderer_invalid_request);
  CHECK(flush_output_length == 0);
  CHECK(status == emel::text::renderer::sequence_status::running);
}

TEST_CASE("renderer_surfaces_local_model_invalid_error_on_render_failure") {
  auto & vocab = make_vocab();
  add_token(vocab, "ok");
  emel::text::renderer::sm renderer{};

  int32_t initialize_err = k_renderer_ok;
  CHECK(initialize_renderer(renderer, vocab, false, nullptr, 0, initialize_err));
  CHECK(initialize_err == k_renderer_ok);

  std::array<char, 16> output = {};
  size_t output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t render_err = k_renderer_ok;

  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = 99999;
  render_ev.sequence_id = 0;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &output_length;
  render_ev.status_out = &status;
  render_ev.error_out = &render_err;

  CHECK_FALSE(renderer.process_event(render_ev));
  CHECK(render_err == k_renderer_model_invalid);
}

TEST_CASE("renderer_action_and_guard_paths") {
  auto & vocab = make_vocab();
  const int32_t token_id = add_token(vocab, "ab");
  const int32_t special_id = add_token(vocab, "<special>", 3);
  (void)token_id;

  emel::text::renderer::action::context ctx = {};
  std::array<char, 8> output = {};

  emel::text::renderer::event::initialize initialize_ev{vocab};
  emel::text::renderer::event::initialize_ctx initialize_runtime_ctx = {};
  emel::text::renderer::event::initialize_runtime initialize_runtime_ev{initialize_ev, initialize_runtime_ctx};
  CHECK(emel::text::renderer::guard::valid_initialize{}(initialize_runtime_ev));
  CHECK_FALSE(emel::text::renderer::guard::invalid_initialize{}(initialize_runtime_ev));

  std::array<std::string_view, 1> long_stop = {
      std::string_view("0123456789012345678901234567890123456789")};
  initialize_ev.stop_sequences = long_stop.data();
  initialize_ev.stop_sequence_count = long_stop.size();
  CHECK_FALSE(emel::text::renderer::guard::valid_initialize{}(initialize_runtime_ev));
  emel::text::renderer::action::set_error(
      initialize_runtime_ctx,
      emel::text::renderer::error::invalid_request);
  CHECK(initialize_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));

  initialize_ev.stop_sequences = nullptr;
  initialize_ev.stop_sequence_count = 0;
  initialize_runtime_ctx = {};
  emel::text::renderer::action::begin_initialize(initialize_runtime_ev, ctx);
  CHECK(ctx.vocab == nullptr);

  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'x';
  size_t output_length = 0;
  emel::text::renderer::event::render_ctx compose_ctx = {};
  CHECK_FALSE(emel::text::renderer::action::compose_output(
      ctx.sequences[0], output.data(), 0, 1, 0, 0, output_length, compose_ctx));
  CHECK(compose_ctx.err == renderer_error_type(emel::text::renderer::error::invalid_request));

  emel::text::renderer::action::context match_context = {};
  match_context.stop_sequence_count = 1;
  match_context.stop_sequences[0].offset = 0;
  match_context.stop_sequences[0].length = 2;
  match_context.stop_storage[0] = 'b';
  match_context.stop_storage[1] = 'c';
  match_context.stop_max_length = 2;
  match_context.sequences[0].holdback_length = 1;
  match_context.sequences[0].holdback[0] = 'b';
  output[0] = 'c';
  emel::text::renderer::event::render_ctx match_ctx = {};
  output_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  CHECK(emel::text::renderer::action::apply_stop_matching(
      match_context.sequences[0],
      match_context,
      output.data(),
      output.size(),
      1,
      output_length,
      status,
      match_ctx));
  CHECK(status == emel::text::renderer::sequence_status::stop_sequence_matched);

  emel::text::renderer::action::context no_match_context = {};
  no_match_context.stop_sequence_count = 1;
  no_match_context.stop_sequences[0].offset = 0;
  no_match_context.stop_sequences[0].length = 2;
  no_match_context.stop_storage[0] = 'z';
  no_match_context.stop_storage[1] = 'z';
  no_match_context.stop_max_length = 2;
  no_match_context.sequences[0].holdback_length = 1;
  no_match_context.sequences[0].holdback[0] = 'a';
  output[0] = 'b';
  emel::text::renderer::event::render_ctx no_match_ctx = {};
  output_length = 0;
  status = emel::text::renderer::sequence_status::running;
  CHECK(emel::text::renderer::action::apply_stop_matching(
      no_match_context.sequences[0],
      no_match_context,
      output.data(),
      output.size(),
      1,
      output_length,
      status,
      no_match_ctx));
  CHECK(output_length == 1);

  initialize_runtime_ctx.detokenizer_err = k_detok_backend_error;
  CHECK(emel::text::renderer::guard::initialize_dispatch_backend_failure{}(initialize_runtime_ev));
  emel::text::renderer::action::set_backend_error(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::backend_error));

  initialize_runtime_ctx.err = renderer_error_type(emel::text::renderer::error::none);
  initialize_runtime_ctx.detokenizer_err = k_detok_model_invalid;
  CHECK(emel::text::renderer::guard::initialize_dispatch_reported_error{}(initialize_runtime_ev));
  emel::text::renderer::action::set_error_from_detokenizer(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::model_invalid));

  initialize_runtime_ctx = {};
  emel::text::renderer::action::begin_initialize(initialize_runtime_ev, ctx);
  emel::text::renderer::action::dispatch_initialize_detokenizer(initialize_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::initialize_dispatch_ok{}(initialize_runtime_ev));
  emel::text::renderer::action::commit_initialize_success(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err == renderer_error_type(emel::text::renderer::error::none));
  CHECK(ctx.vocab == &vocab);

  emel::text::renderer::event::render render_ev = {};
  size_t out_len = 0;
  int32_t err = k_renderer_ok;
  status = emel::text::renderer::sequence_status::running;
  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.output_length_out = &out_len;
  render_ev.status_out = &status;
  render_ev.error_out = &err;
  render_ev.sequence_id = 0;
  render_ev.token_id = 0;
  emel::text::renderer::event::render_ctx render_runtime_ctx = {};
  emel::text::renderer::event::render_runtime render_runtime_ev{render_ev, render_runtime_ctx};
  emel::text::renderer::action::begin_render(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.output_length == 0);

  render_runtime_ctx.detokenizer_err = k_detok_backend_error;
  CHECK(emel::text::renderer::guard::render_dispatch_backend_failure{}(render_runtime_ev));
  emel::text::renderer::action::set_backend_error(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::backend_error));

  render_runtime_ctx.err = renderer_error_type(emel::text::renderer::error::none);
  render_runtime_ctx.detokenizer_err = k_detok_model_invalid;
  CHECK(emel::text::renderer::guard::render_dispatch_reported_error{}(render_runtime_ev));
  emel::text::renderer::action::set_error_from_detokenizer(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::model_invalid));

  render_runtime_ctx = {};
  render_ev.output = nullptr;
  render_ev.output_capacity = 0;
  render_ev.token_id = special_id;
  render_ev.emit_special = false;
  ctx.sequences[0].pending_length = 0;
  ctx.sequences[0].holdback_length = 0;
  emel::text::renderer::action::begin_render(render_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::sequence_running{}(render_runtime_ev, ctx));
  emel::text::renderer::action::dispatch_render_detokenizer(render_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::render_dispatch_ok{}(render_runtime_ev, ctx));
  emel::text::renderer::action::commit_render_detokenizer_output(render_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::strip_not_needed{}(render_runtime_ev, ctx));
  emel::text::renderer::action::update_render_strip_state(render_runtime_ev, ctx);
  emel::text::renderer::action::apply_render_stop_matching(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::none));
  CHECK(render_runtime_ctx.output_length == 0);

  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  output[0] = ' ';
  output[1] = '\t';
  output[2] = 'x';
  render_runtime_ctx.detokenizer_err = k_detok_ok;
  render_runtime_ctx.detokenizer_output_length = 3;
  render_runtime_ctx.detokenizer_pending_length = 0;
  ctx.sequences[0].strip_leading_space = true;
  emel::text::renderer::action::commit_render_detokenizer_output(render_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::strip_needed{}(render_runtime_ev, ctx));
  emel::text::renderer::action::compute_render_leading_space_prefix(render_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::strip_prefix_nonzero{}(render_runtime_ev, ctx));
  CHECK_FALSE(emel::text::renderer::guard::strip_prefix_zero{}(render_runtime_ev, ctx));
  emel::text::renderer::action::apply_render_leading_space_strip(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.produced_length == 1);
  CHECK(output[0] == 'x');

  render_ev.output = output.data();
  render_ev.output_capacity = output.size();
  render_ev.token_id = token_id;
  render_runtime_ctx = {};
  emel::text::renderer::action::begin_render(render_runtime_ev, ctx);
  render_runtime_ctx.detokenizer_err = k_detok_ok;
  render_runtime_ctx.detokenizer_output_length = render_ev.output_capacity + 1;
  render_runtime_ctx.detokenizer_pending_length = 0;
  CHECK(emel::text::renderer::guard::render_dispatch_lengths_invalid{}(render_runtime_ev, ctx));
  emel::text::renderer::action::set_invalid_request(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));

  render_runtime_ctx = {};
  emel::text::renderer::action::begin_render(render_runtime_ev, ctx);
  render_runtime_ctx.detokenizer_err = k_detok_ok;
  render_runtime_ctx.detokenizer_output_length = 0;
  render_runtime_ctx.detokenizer_pending_length =
      ctx.sequences[0].pending_bytes.size() + 1;
  CHECK(emel::text::renderer::guard::render_dispatch_lengths_invalid{}(render_runtime_ev, ctx));
  emel::text::renderer::action::set_invalid_request(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));

  ctx.sequences[0].stop_matched = true;
  render_runtime_ctx = {};
  emel::text::renderer::action::begin_render(render_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::sequence_stop_matched{}(render_runtime_ev, ctx));
  emel::text::renderer::action::render_sequence_already_stopped(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.status ==
        emel::text::renderer::sequence_status::stop_sequence_matched);
  CHECK(render_runtime_ctx.output_length == 0);
  ctx.sequences[0].stop_matched = false;

  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = output.data();
  flush_ev.output_capacity = output.size();
  flush_ev.output_length_out = &out_len;
  flush_ev.status_out = &status;
  flush_ev.error_out = &err;
  emel::text::renderer::event::flush_ctx flush_runtime_ctx = {};
  emel::text::renderer::event::flush_runtime flush_runtime_ev{flush_ev, flush_runtime_ctx};
  emel::text::renderer::action::begin_flush(flush_runtime_ev, ctx);
  CHECK(flush_runtime_ctx.output_length == 0);

  flush_ev.output = output.data();
  flush_ev.output_capacity = 0;
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'h';
  flush_runtime_ctx = {};
  emel::text::renderer::action::begin_flush(flush_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::flush_output_too_large{}(flush_runtime_ev, ctx));
  emel::text::renderer::action::set_invalid_request(flush_runtime_ev, ctx);
  CHECK(flush_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));

  flush_ev.output_capacity = output.size();
  ctx.sequences[0].holdback_length = 1;
  ctx.sequences[0].holdback[0] = 'h';
  flush_runtime_ctx = {};
  emel::text::renderer::action::begin_flush(flush_runtime_ev, ctx);
  CHECK(emel::text::renderer::guard::flush_output_fits{}(flush_runtime_ev, ctx));
  emel::text::renderer::action::flush_copy_sequence_buffers(flush_runtime_ev, ctx);
  CHECK(flush_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::none));
  CHECK(flush_runtime_ctx.output_length == 1);

  render_runtime_ctx = {};
  emel::text::renderer::action::reject_render(render_runtime_ev, ctx);
  CHECK(render_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));
  flush_runtime_ctx = {};
  emel::text::renderer::action::reject_flush(flush_runtime_ev, ctx);
  CHECK(flush_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));
  initialize_runtime_ctx = {};
  emel::text::renderer::action::reject_initialize(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));

  initialize_runtime_ctx.err = renderer_error_type(emel::text::renderer::error::none);
  emel::text::renderer::action::ensure_last_error(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::backend_error));
  emel::text::renderer::action::mark_done(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err == renderer_error_type(emel::text::renderer::error::none));
  emel::text::renderer::action::on_unexpected(initialize_runtime_ev, ctx);
  CHECK(initialize_runtime_ctx.err ==
        renderer_error_type(emel::text::renderer::error::invalid_request));

  emel::text::renderer::event::render bad_render = {};
  emel::text::renderer::event::render_ctx bad_render_ctx = {};
  bad_render.token_id = -1;
  CHECK_FALSE(emel::text::renderer::guard::valid_render{}(
      emel::text::renderer::event::render_runtime{bad_render, bad_render_ctx}, ctx));
  CHECK(emel::text::renderer::guard::invalid_render{}(
      emel::text::renderer::event::render_runtime{bad_render, bad_render_ctx}, ctx));
  ctx.vocab = &vocab;
  bad_render_ctx = {};
  bad_render.token_id = 0;
  bad_render.output = output.data();
  bad_render.output_capacity = output.size();
  bad_render.output_length_out = &out_len;
  bad_render.status_out = &status;
  bad_render.error_out = &err;
  bad_render.sequence_id = 0;
  CHECK(emel::text::renderer::guard::valid_render{}(
      emel::text::renderer::event::render_runtime{bad_render, bad_render_ctx}, ctx));

  bad_render.output = nullptr;
  bad_render.output_capacity = 0;
  CHECK(emel::text::renderer::guard::valid_render{}(
      emel::text::renderer::event::render_runtime{bad_render, bad_render_ctx}, ctx));

  emel::text::renderer::event::flush bad_flush = {};
  emel::text::renderer::event::flush_ctx bad_flush_ctx = {};
  emel::text::renderer::event::flush_runtime bad_flush_runtime{bad_flush, bad_flush_ctx};
  CHECK(emel::text::renderer::guard::valid_flush{}(bad_flush_runtime, ctx));
  CHECK_FALSE(emel::text::renderer::guard::invalid_flush{}(bad_flush_runtime, ctx));
  bad_flush.sequence_id = 0;
  bad_flush.output = output.data();
  bad_flush.output_capacity = output.size();
  bad_flush.output_length_out = &out_len;
  bad_flush.status_out = &status;
  bad_flush.error_out = &err;
  CHECK(emel::text::renderer::guard::valid_flush{}(bad_flush_runtime, ctx));
}
