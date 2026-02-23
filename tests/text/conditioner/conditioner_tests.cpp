#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

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

emel::model::data::vocab & make_bpe_vocab() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  vocab.ignore_merges = true;
  vocab.add_bos = true;
  vocab.add_eos = true;

  const int32_t hello_id = add_token(vocab, "hello");
  const int32_t world_id = add_token(vocab, "\xC4\xA0" "world");
  const int32_t bos_id = add_token(vocab, "<bos>");
  const int32_t eos_id = add_token(vocab, "<eos>");
  CHECK(hello_id == 0);
  CHECK(world_id == 1);
  vocab.bos_id = bos_id;
  vocab.eos_id = eos_id;
  return vocab;
}

bool tokenizer_bind_dispatch(
    void * tokenizer_sm,
    const emel::text::tokenizer::event::bind & ev) {
  if (tokenizer_sm == nullptr) {
    return false;
  }
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(
    void * tokenizer_sm,
    const emel::text::tokenizer::event::tokenize & ev) {
  if (tokenizer_sm == nullptr) {
    return false;
  }
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

struct callback_recorder {
  int bind_done = 0;
  int bind_error = 0;
  int prepare_done = 0;
  int prepare_error = 0;
  int32_t last_token_count = 0;
  int32_t last_error = EMEL_OK;
};

bool on_bind_done(void * owner,
                  const emel::text::conditioner::events::binding_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<callback_recorder *>(owner)->bind_done += 1;
  return true;
}

bool on_bind_error(void * owner,
                   const emel::text::conditioner::events::binding_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->bind_error += 1;
  recorder->last_error = ev.err;
  return true;
}

bool on_prepare_done(void * owner,
                     const emel::text::conditioner::events::conditioning_done & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->prepare_done += 1;
  recorder->last_token_count = ev.token_count;
  return true;
}

bool on_prepare_error(
    void * owner,
    const emel::text::conditioner::events::conditioning_error & ev) {
  if (owner == nullptr) {
    return false;
  }
  auto * recorder = static_cast<callback_recorder *>(owner);
  recorder->prepare_error += 1;
  recorder->last_error = ev.err;
  return true;
}

struct fixed_formatter_ctx {
  std::string_view text = {};
};

bool format_fixed(void * formatter_ctx,
                  const emel::text::formatter::format_request & request,
                  int32_t * error_out) {
  if (error_out != nullptr) {
    *error_out = EMEL_OK;
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  if (formatter_ctx == nullptr) {
    if (error_out != nullptr) {
      *error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }

  const auto * ctx = static_cast<const fixed_formatter_ctx *>(formatter_ctx);
  if ((request.output == nullptr && request.output_capacity > 0) ||
      request.output_capacity < ctx->text.size()) {
    if (error_out != nullptr) {
      *error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }

  if (!ctx->text.empty()) {
    std::memcpy(request.output, ctx->text.data(), ctx->text.size());
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = ctx->text.size();
  }
  return true;
}

bool tokenizer_bind_fail_no_error(
    void *,
    const emel::text::tokenizer::event::bind &) {
  return false;
}

bool tokenizer_bind_fail_with_error(
    void *,
    const emel::text::tokenizer::event::bind & ev) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_MODEL_INVALID;
  }
  return true;
}

bool tokenizer_tokenize_fail_no_error(
    void *,
    const emel::text::tokenizer::event::tokenize &) {
  return false;
}

bool tokenizer_tokenize_fail_with_error(
    void *,
    const emel::text::tokenizer::event::tokenize & ev) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_MODEL_INVALID;
  }
  return true;
}

bool formatter_fail_no_error(
    void *,
    const emel::text::formatter::format_request & request,
    int32_t *) {
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  return false;
}

bool formatter_fail_with_error(
    void *,
    const emel::text::formatter::format_request & request,
    int32_t * error_out) {
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  if (error_out != nullptr) {
    *error_out = EMEL_ERR_MODEL_INVALID;
  }
  return false;
}

}  // namespace

TEST_CASE("conditioner_bind_and_prepare_with_default_formatter") {
  auto & vocab = make_bpe_vocab();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  callback_recorder recorder{};

  int32_t bind_err = EMEL_OK;
  emel::text::conditioner::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.tokenizer_sm = &tokenizer;
  bind_ev.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  bind_ev.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  bind_ev.error_out = &bind_err;
  bind_ev.owner_sm = &recorder;
  bind_ev.dispatch_done = on_bind_done;
  bind_ev.dispatch_error = on_bind_error;

  CHECK(conditioner.process_event(bind_ev));
  CHECK(bind_err == EMEL_OK);
  CHECK(recorder.bind_done == 1);
  CHECK(recorder.bind_error == 0);

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t prepare_err = EMEL_OK;
  emel::text::conditioner::event::prepare prepare_ev = {};
  prepare_ev.input = "hello world";
  prepare_ev.use_bind_defaults = true;
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());
  prepare_ev.token_count_out = &token_count;
  prepare_ev.error_out = &prepare_err;
  prepare_ev.owner_sm = &recorder;
  prepare_ev.dispatch_done = on_prepare_done;
  prepare_ev.dispatch_error = on_prepare_error;

  CHECK(conditioner.process_event(prepare_ev));
  CHECK(prepare_err == EMEL_OK);
  CHECK(token_count == 4);
  CHECK(tokens[0] == vocab.bos_id);
  CHECK(tokens[1] == 0);
  CHECK(tokens[2] == 1);
  CHECK(tokens[3] == vocab.eos_id);
  CHECK(recorder.prepare_done == 1);
  CHECK(recorder.prepare_error == 0);
  CHECK(recorder.last_token_count == 4);
}

TEST_CASE("conditioner_prepare_requires_bind") {
  auto & vocab = make_bpe_vocab();
  emel::text::conditioner::sm conditioner{};
  callback_recorder recorder{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t prepare_err = EMEL_OK;
  emel::text::conditioner::event::prepare prepare_ev = {};
  prepare_ev.input = "hello";
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());
  prepare_ev.token_count_out = &token_count;
  prepare_ev.error_out = &prepare_err;
  prepare_ev.owner_sm = &recorder;
  prepare_ev.dispatch_done = on_prepare_done;
  prepare_ev.dispatch_error = on_prepare_error;

  CHECK_FALSE(conditioner.process_event(prepare_ev));
  CHECK(prepare_err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(token_count == 0);
  CHECK(recorder.prepare_done == 0);
  CHECK(recorder.prepare_error == 1);
  CHECK(recorder.last_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(vocab.n_tokens >= 2);
}

TEST_CASE("conditioner_uses_injected_formatter_function") {
  auto & vocab = make_bpe_vocab();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};

  fixed_formatter_ctx formatter_ctx{};
  formatter_ctx.text = "hello world";

  int32_t bind_err = EMEL_OK;
  emel::text::conditioner::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.tokenizer_sm = &tokenizer;
  bind_ev.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  bind_ev.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  bind_ev.formatter_ctx = &formatter_ctx;
  bind_ev.format_prompt = format_fixed;
  bind_ev.add_special = false;
  bind_ev.parse_special = false;
  bind_ev.error_out = &bind_err;

  CHECK(conditioner.process_event(bind_ev));
  CHECK(bind_err == EMEL_OK);

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t prepare_err = EMEL_OK;
  emel::text::conditioner::event::prepare prepare_ev = {};
  prepare_ev.input = "ignored";
  prepare_ev.use_bind_defaults = true;
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());
  prepare_ev.token_count_out = &token_count;
  prepare_ev.error_out = &prepare_err;

  CHECK(conditioner.process_event(prepare_ev));
  CHECK(prepare_err == EMEL_OK);
  CHECK(token_count == 2);
  CHECK(tokens[0] == 0);
  CHECK(tokens[1] == 1);
}

TEST_CASE("conditioner_action_and_guard_error_paths") {
  auto & vocab = make_bpe_vocab();
  int dummy = 0;
  void * dummy_ptr = &dummy;

  emel::text::conditioner::action::context ctx = {};
  emel::text::conditioner::event::bind bind_ev = {};
  emel::text::conditioner::event::prepare prepare_ev = {};

  CHECK_FALSE(emel::text::conditioner::guard::valid_bind{}(bind_ev));
  CHECK(emel::text::conditioner::guard::invalid_bind{}(bind_ev));

  bind_ev.vocab = &vocab;
  bind_ev.tokenizer_sm = dummy_ptr;
  bind_ev.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  bind_ev.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  bind_ev.format_prompt = emel::text::formatter::format_raw;
  CHECK(emel::text::conditioner::guard::valid_bind{}(bind_ev));

  emel::text::conditioner::action::reject_bind(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  prepare_ev.token_ids_out = nullptr;
  prepare_ev.token_count_out = nullptr;
  prepare_ev.error_out = nullptr;
  CHECK(emel::text::conditioner::guard::invalid_prepare{}(prepare_ev, ctx));

  emel::text::conditioner::action::begin_bind(bind_ev, ctx);
  CHECK(ctx.vocab == &vocab);
  CHECK_FALSE(ctx.is_bound);

  ctx.vocab = nullptr;
  emel::text::conditioner::action::bind_tokenizer(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = &vocab;
  ctx.tokenizer_sm = dummy_ptr;
  ctx.dispatch_tokenizer_bind = tokenizer_bind_fail_no_error;
  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  ctx.format_prompt = emel::text::formatter::format_raw;
  emel::text::conditioner::action::bind_tokenizer(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.dispatch_tokenizer_bind = tokenizer_bind_fail_with_error;
  emel::text::conditioner::action::bind_tokenizer(ctx);
  CHECK(ctx.last_error == EMEL_ERR_MODEL_INVALID);

  ctx.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  emel::text::tokenizer::sm tokenizer{};
  ctx.tokenizer_sm = &tokenizer;
  emel::text::conditioner::action::bind_tokenizer(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.is_bound);

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  prepare_ev.input = "hello world";
  prepare_ev.use_bind_defaults = false;
  prepare_ev.add_special = false;
  prepare_ev.parse_special = true;
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());
  prepare_ev.token_count_out = &token_count;
  prepare_ev.error_out = &err;

  emel::text::conditioner::action::begin_prepare(prepare_ev, ctx);
  CHECK_FALSE(ctx.add_special);
  CHECK(ctx.parse_special);
  CHECK(ctx.token_count == 0);
  CHECK(emel::text::conditioner::guard::valid_prepare{}(prepare_ev, ctx));

  ctx.format_prompt = formatter_fail_no_error;
  emel::text::conditioner::action::run_format(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.format_prompt = formatter_fail_with_error;
  emel::text::conditioner::action::run_format(ctx);
  CHECK(ctx.last_error == EMEL_ERR_MODEL_INVALID);

  ctx.format_prompt = emel::text::formatter::format_raw;
  ctx.last_error = EMEL_OK;
  emel::text::conditioner::action::run_format(ctx);
  CHECK(ctx.last_error == EMEL_OK);

  ctx.dispatch_tokenizer_tokenize = nullptr;
  emel::text::conditioner::action::run_tokenize(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_fail_no_error;
  emel::text::conditioner::action::run_tokenize(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_fail_with_error;
  emel::text::conditioner::action::run_tokenize(ctx);
  CHECK(ctx.last_error == EMEL_ERR_MODEL_INVALID);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  ctx.tokenizer_sm = &tokenizer;
  ctx.parse_special = false;
  emel::text::conditioner::action::run_tokenize(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.token_count > 0);

  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  emel::text::conditioner::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.last_error = EMEL_ERR_MODEL_INVALID;
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  emel::text::conditioner::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_MODEL_INVALID);

  emel::text::conditioner::action::mark_done(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::text::conditioner::action::on_unexpected(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.token_count == 0);

  emel::text::conditioner::action::clear_prepare_request(ctx);
  CHECK(ctx.token_ids_out == nullptr);

  ctx.phase_error = EMEL_OK;
  CHECK(emel::text::conditioner::guard::phase_ok{}(ctx));
  CHECK_FALSE(emel::text::conditioner::guard::phase_failed{}(ctx));
}

TEST_CASE("formatter_format_raw_handles_invalid_and_empty_inputs") {
  int32_t err = EMEL_OK;
  size_t out_len = 7;

  emel::text::formatter::format_request bad_req = {};
  bad_req.input = "x";
  bad_req.output = nullptr;
  bad_req.output_capacity = 1;
  bad_req.output_length_out = &out_len;
  CHECK_FALSE(emel::text::formatter::format_raw(nullptr, bad_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(out_len == 0);

  err = EMEL_OK;
  out_len = 9;
  emel::text::formatter::format_request empty_req = {};
  empty_req.input = "";
  empty_req.output = nullptr;
  empty_req.output_capacity = 0;
  empty_req.output_length_out = &out_len;
  CHECK(emel::text::formatter::format_raw(nullptr, empty_req, &err));
  CHECK(err == EMEL_OK);
  CHECK(out_len == 0);
}
