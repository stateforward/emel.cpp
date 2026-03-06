#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

constexpr int32_t
conditioner_code(const emel::text::conditioner::error err) noexcept {
  return static_cast<int32_t>(err);
}

constexpr int32_t k_external_model_invalid_code =
    emel::text::tokenizer::error_code(emel::text::tokenizer::error::model_invalid);

int32_t add_token(emel::model::data::vocab &vocab, const char *text,
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

emel::model::data::vocab &make_bpe_vocab() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  vocab.ignore_merges = true;
  vocab.add_bos = true;
  vocab.add_eos = true;

  const int32_t hello_id = add_token(vocab, "hello");
  const int32_t world_id = add_token(vocab, "\xC4\xA0"
                                            "world");
  const int32_t bos_id = add_token(vocab, "<bos>");
  const int32_t eos_id = add_token(vocab, "<eos>");
  CHECK(hello_id == 0);
  CHECK(world_id == 1);
  vocab.bos_id = bos_id;
  vocab.eos_id = eos_id;
  return vocab;
}

bool tokenizer_bind_dispatch(void *tokenizer_sm,
                             const emel::text::tokenizer::event::bind &ev) {
  if (tokenizer_sm == nullptr) {
    return false;
  }
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)
      ->process_event(ev);
}

bool tokenizer_tokenize_dispatch(
    void *tokenizer_sm, const emel::text::tokenizer::event::tokenize &ev) {
  if (tokenizer_sm == nullptr) {
    return false;
  }
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)
      ->process_event(ev);
}

struct callback_recorder {
  int bind_done = 0;
  int bind_error = 0;
  int prepare_done = 0;
  int prepare_error = 0;
  int32_t last_token_count = 0;
  int32_t last_error = conditioner_code(emel::text::conditioner::error::none);
};

bool on_bind_done(void *owner,
                  const emel::text::conditioner::events::binding_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<callback_recorder *>(owner)->bind_done += 1;
  return true;
}

bool on_bind_error(void *owner,
                   const emel::text::conditioner::events::binding_error &ev) {
  if (owner == nullptr) {
    return false;
  }
  auto *recorder = static_cast<callback_recorder *>(owner);
  recorder->bind_error += 1;
  recorder->last_error = ev.err;
  return true;
}

bool on_prepare_done(
    void *owner, const emel::text::conditioner::events::conditioning_done &ev) {
  if (owner == nullptr) {
    return false;
  }
  auto *recorder = static_cast<callback_recorder *>(owner);
  recorder->prepare_done += 1;
  recorder->last_token_count = ev.token_count;
  return true;
}

bool on_prepare_error(
    void *owner,
    const emel::text::conditioner::events::conditioning_error &ev) {
  if (owner == nullptr) {
    return false;
  }
  auto *recorder = static_cast<callback_recorder *>(owner);
  recorder->prepare_error += 1;
  recorder->last_error = ev.err;
  return true;
}

struct fixed_formatter_ctx {
  std::string_view text = {};
};

bool format_fixed(void *formatter_ctx,
                  const emel::text::formatter::format_request &request,
                  int32_t *error_out) {
  if (error_out != nullptr) {
    *error_out = conditioner_code(emel::text::conditioner::error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  if (formatter_ctx == nullptr) {
    if (error_out != nullptr) {
      *error_out =
          conditioner_code(emel::text::conditioner::error::invalid_argument);
    }
    return false;
  }

  const auto *ctx = static_cast<const fixed_formatter_ctx *>(formatter_ctx);
  if ((request.output == nullptr && request.output_capacity > 0) ||
      request.output_capacity < ctx->text.size()) {
    if (error_out != nullptr) {
      *error_out =
          conditioner_code(emel::text::conditioner::error::invalid_argument);
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

bool tokenizer_bind_fail_no_error(void *,
                                  const emel::text::tokenizer::event::bind &) {
  return false;
}

bool tokenizer_bind_fail_with_error(
    void *, const emel::text::tokenizer::event::bind &ev) {
  if (ev.error_out != nullptr) {
    *ev.error_out = k_external_model_invalid_code;
  }
  return true;
}

bool tokenizer_tokenize_fail_no_error(
    void *, const emel::text::tokenizer::event::tokenize &) {
  return false;
}

bool tokenizer_tokenize_fail_with_error(
    void *, const emel::text::tokenizer::event::tokenize &ev) {
  if (ev.error_out != nullptr) {
    *ev.error_out = k_external_model_invalid_code;
  }
  return true;
}

bool tokenizer_tokenize_negative_count(
    void *, const emel::text::tokenizer::event::tokenize &ev) {
  if (ev.error_out != nullptr) {
    *ev.error_out = conditioner_code(emel::text::conditioner::error::none);
  }
  if (ev.token_count_out != nullptr) {
    *ev.token_count_out = -1;
  }
  return true;
}

bool tokenizer_tokenize_over_capacity(
    void *, const emel::text::tokenizer::event::tokenize &ev) {
  if (ev.error_out != nullptr) {
    *ev.error_out = conditioner_code(emel::text::conditioner::error::none);
  }
  if (ev.token_count_out != nullptr) {
    *ev.token_count_out = ev.token_capacity + 1;
  }
  return true;
}

bool formatter_fail_no_error(
    void *, const emel::text::formatter::format_request &request, int32_t *) {
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  return false;
}

bool formatter_fail_with_error(
    void *, const emel::text::formatter::format_request &request,
    int32_t *error_out) {
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  if (error_out != nullptr) {
    *error_out = k_external_model_invalid_code;
  }
  return false;
}

bool formatter_oversized_length(
    void *, const emel::text::formatter::format_request &request,
    int32_t *error_out) {
  if (error_out != nullptr) {
    *error_out = conditioner_code(emel::text::conditioner::error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = request.output_capacity + 1;
  }
  return true;
}

} // namespace

TEST_CASE("conditioner_bind_and_prepare_with_default_formatter") {
  auto &vocab = make_bpe_vocab();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  callback_recorder recorder{};

  int32_t bind_err = conditioner_code(emel::text::conditioner::error::none);
  emel::text::conditioner::event::bind bind_ev{vocab};
  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  bind_ev.tokenizer_sm = &tokenizer;
  bind_ev.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  bind_ev.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  bind_ev.error_out = &bind_err;
  bind_ev.owner_sm = &recorder;
  bind_ev.dispatch_done = on_bind_done;
  bind_ev.dispatch_error = on_bind_error;

  CHECK(conditioner.process_event(bind_ev));
  CHECK(bind_err == conditioner_code(emel::text::conditioner::error::none));
  CHECK(recorder.bind_done == 1);
  CHECK(recorder.bind_error == 0);

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t prepare_err = conditioner_code(emel::text::conditioner::error::none);
  emel::text::conditioner::event::prepare prepare_ev{token_count, prepare_err};
  prepare_ev.input = "hello world";
  prepare_ev.use_bind_defaults = true;
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());
  prepare_ev.owner_sm = &recorder;
  prepare_ev.dispatch_done = on_prepare_done;
  prepare_ev.dispatch_error = on_prepare_error;

  CHECK(conditioner.process_event(prepare_ev));
  CHECK(prepare_err == conditioner_code(emel::text::conditioner::error::none));
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
  auto &vocab = make_bpe_vocab();
  emel::text::conditioner::sm conditioner{};
  callback_recorder recorder{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t prepare_err = conditioner_code(emel::text::conditioner::error::none);
  emel::text::conditioner::event::prepare prepare_ev{token_count, prepare_err};
  prepare_ev.input = "hello";
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());
  prepare_ev.owner_sm = &recorder;
  prepare_ev.dispatch_done = on_prepare_done;
  prepare_ev.dispatch_error = on_prepare_error;

  CHECK_FALSE(conditioner.process_event(prepare_ev));
  CHECK(prepare_err ==
        conditioner_code(emel::text::conditioner::error::invalid_argument));
  CHECK(token_count == 0);
  CHECK(recorder.prepare_done == 0);
  CHECK(recorder.prepare_error == 1);
  CHECK(recorder.last_error ==
        conditioner_code(emel::text::conditioner::error::invalid_argument));
  CHECK(vocab.n_tokens >= 2);
}

TEST_CASE("conditioner_uses_injected_formatter_function") {
  auto &vocab = make_bpe_vocab();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};

  fixed_formatter_ctx formatter_ctx{};
  formatter_ctx.text = "hello world";

  int32_t bind_err = conditioner_code(emel::text::conditioner::error::none);
  emel::text::conditioner::event::bind bind_ev{vocab};
  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  bind_ev.tokenizer_sm = &tokenizer;
  bind_ev.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  bind_ev.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  bind_ev.formatter_ctx = &formatter_ctx;
  bind_ev.format_prompt = format_fixed;
  bind_ev.add_special = false;
  bind_ev.parse_special = false;
  bind_ev.error_out = &bind_err;

  CHECK(conditioner.process_event(bind_ev));
  CHECK(bind_err == conditioner_code(emel::text::conditioner::error::none));

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t prepare_err = conditioner_code(emel::text::conditioner::error::none);
  emel::text::conditioner::event::prepare prepare_ev{token_count, prepare_err};
  prepare_ev.input = "ignored";
  prepare_ev.use_bind_defaults = true;
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());

  CHECK(conditioner.process_event(prepare_ev));
  CHECK(prepare_err == conditioner_code(emel::text::conditioner::error::none));
  CHECK(token_count == 2);
  CHECK(tokens[0] == 0);
  CHECK(tokens[1] == 1);
}

TEST_CASE("conditioner_action_and_guard_error_paths") {
  using conditioner_error = emel::text::conditioner::error;

  auto &vocab = make_bpe_vocab();
  int dummy = 0;
  void *dummy_ptr = &dummy;
  int32_t token_count = 0;
  int32_t err = conditioner_code(emel::text::conditioner::error::none);

  emel::text::conditioner::action::context ctx = {};
  emel::text::conditioner::event::bind bind_ev{vocab};
  emel::text::conditioner::event::bind_ctx bind_ctx = {};
  emel::text::conditioner::event::bind_runtime bind_runtime{bind_ev, bind_ctx};
  emel::text::conditioner::event::prepare prepare_ev{token_count, err};
  emel::text::conditioner::event::prepare_ctx prepare_ctx = {};
  std::array<char, emel::text::conditioner::action::k_max_formatted_bytes>
      formatted = {};
  prepare_ctx.formatted = formatted.data();
  prepare_ctx.formatted_capacity = formatted.size();
  emel::text::conditioner::event::prepare_runtime prepare_runtime{prepare_ev,
                                                                  prepare_ctx};

  CHECK_FALSE(emel::text::conditioner::guard::valid_bind{}(bind_runtime));
  CHECK(emel::text::conditioner::guard::invalid_bind{}(bind_runtime));

  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  bind_ev.tokenizer_sm = dummy_ptr;
  bind_ev.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  bind_ev.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  bind_ev.format_prompt = emel::text::formatter::format_raw;
  CHECK(emel::text::conditioner::guard::valid_bind{}(bind_runtime));

  emel::text::conditioner::action::reject_bind(bind_runtime, ctx);
  CHECK(bind_ctx.err == conditioner_error::invalid_argument);
  CHECK_FALSE(bind_ctx.result);

  CHECK(
      emel::text::conditioner::guard::invalid_prepare{}(prepare_runtime, ctx));

  emel::text::conditioner::action::begin_bind(bind_runtime, ctx);
  CHECK(ctx.vocab == &vocab);
  CHECK_FALSE(ctx.is_bound);
  CHECK(bind_ctx.err == conditioner_error::none);
  CHECK_FALSE(bind_ctx.result);

  ctx.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  ctx.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  ctx.tokenizer_sm = dummy_ptr;
  ctx.dispatch_tokenizer_bind = tokenizer_bind_fail_no_error;
  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  ctx.format_prompt = emel::text::formatter::format_raw;
  emel::text::conditioner::action::dispatch_bind_tokenizer(bind_runtime, ctx);
  CHECK(emel::text::conditioner::guard::bind_rejected_no_error{}(bind_runtime));
  emel::text::conditioner::action::bind_error_backend(bind_runtime, ctx);
  CHECK(bind_ctx.err == conditioner_error::backend);
  CHECK_FALSE(bind_ctx.result);

  ctx.dispatch_tokenizer_bind = tokenizer_bind_fail_with_error;
  emel::text::conditioner::action::dispatch_bind_tokenizer(bind_runtime, ctx);
  CHECK(emel::text::conditioner::guard::bind_error_model_invalid_code{}(
      bind_runtime));
  emel::text::conditioner::action::set_error_model_invalid(bind_runtime, ctx);
  CHECK(bind_ctx.err == conditioner_error::model_invalid);
  CHECK_FALSE(bind_ctx.result);

  ctx.dispatch_tokenizer_bind = tokenizer_bind_dispatch;
  emel::text::tokenizer::sm tokenizer{};
  ctx.tokenizer_sm = &tokenizer;
  emel::text::conditioner::action::dispatch_bind_tokenizer(bind_runtime, ctx);
  CHECK(emel::text::conditioner::guard::bind_successful{}(bind_runtime));
  emel::text::conditioner::action::bind_success(bind_runtime, ctx);
  CHECK(bind_ctx.err == conditioner_error::none);
  CHECK(bind_ctx.result);
  CHECK(ctx.is_bound);

  std::array<int32_t, 8> tokens = {};
  prepare_ev.input = "hello world";
  prepare_ev.use_bind_defaults = false;
  prepare_ev.add_special = false;
  prepare_ev.parse_special = true;
  prepare_ev.token_ids_out = tokens.data();
  prepare_ev.token_capacity = static_cast<int32_t>(tokens.size());

  CHECK(
      emel::text::conditioner::guard::use_request_overrides{}(prepare_runtime));
  emel::text::conditioner::action::begin_prepare_from_request(prepare_runtime,
                                                              ctx);
  CHECK_FALSE(prepare_ctx.add_special);
  CHECK(prepare_ctx.parse_special);
  CHECK(prepare_ctx.token_count == 0);
  CHECK(emel::text::conditioner::guard::valid_prepare{}(prepare_runtime, ctx));

  ctx.format_prompt = formatter_fail_no_error;
  emel::text::conditioner::action::dispatch_format(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::format_rejected_no_error{}(
      prepare_runtime));
  emel::text::conditioner::action::format_error_backend(prepare_runtime, ctx);
  CHECK(prepare_ctx.err == conditioner_error::backend);
  CHECK_FALSE(prepare_ctx.result);

  ctx.format_prompt = formatter_fail_with_error;
  emel::text::conditioner::action::dispatch_format(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::format_error_model_invalid_code{}(
      prepare_runtime));
  emel::text::conditioner::action::set_error_model_invalid(prepare_runtime,
                                                           ctx);
  CHECK(prepare_ctx.err == conditioner_error::model_invalid);
  CHECK_FALSE(prepare_ctx.result);

  ctx.format_prompt = formatter_oversized_length;
  emel::text::conditioner::action::dispatch_format(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::format_length_overflow{}(
      prepare_runtime));
  emel::text::conditioner::action::format_error_invalid_argument(
      prepare_runtime, ctx);
  CHECK(prepare_ctx.err == conditioner_error::invalid_argument);
  CHECK_FALSE(prepare_ctx.result);

  ctx.format_prompt = emel::text::formatter::format_raw;
  emel::text::conditioner::action::dispatch_format(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::format_successful{}(prepare_runtime));
  CHECK(prepare_ctx.formatted_length > 0);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_fail_no_error;
  emel::text::conditioner::action::dispatch_tokenize(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::tokenize_rejected_no_error{}(
      prepare_runtime));
  emel::text::conditioner::action::tokenize_error_backend(prepare_runtime, ctx);
  CHECK(prepare_ctx.err == conditioner_error::backend);
  CHECK_FALSE(prepare_ctx.result);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_fail_with_error;
  emel::text::conditioner::action::dispatch_tokenize(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::tokenize_error_model_invalid_code{}(
      prepare_runtime));
  emel::text::conditioner::action::set_error_model_invalid(prepare_runtime,
                                                           ctx);
  CHECK(prepare_ctx.err == conditioner_error::model_invalid);
  CHECK_FALSE(prepare_ctx.result);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_negative_count;
  emel::text::conditioner::action::dispatch_tokenize(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::tokenize_count_invalid{}(
      prepare_runtime));
  emel::text::conditioner::action::tokenize_error_backend(prepare_runtime, ctx);
  CHECK(prepare_ctx.err == conditioner_error::backend);
  CHECK_FALSE(prepare_ctx.result);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_over_capacity;
  emel::text::conditioner::action::dispatch_tokenize(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::tokenize_count_invalid{}(
      prepare_runtime));
  emel::text::conditioner::action::tokenize_error_backend(prepare_runtime, ctx);
  CHECK(prepare_ctx.err == conditioner_error::backend);
  CHECK_FALSE(prepare_ctx.result);

  ctx.dispatch_tokenizer_tokenize = tokenizer_tokenize_dispatch;
  ctx.tokenizer_sm = &tokenizer;
  prepare_ctx.parse_special = false;
  emel::text::conditioner::action::dispatch_tokenize(prepare_runtime, ctx);
  CHECK(emel::text::conditioner::guard::tokenize_successful{}(prepare_runtime));
  emel::text::conditioner::action::prepare_success(prepare_runtime, ctx);
  CHECK(prepare_ctx.err == conditioner_error::none);
  CHECK(prepare_ctx.result);
  CHECK(prepare_ctx.token_count > 0);

  emel::text::conditioner::action::on_unexpected(bind_runtime, ctx);
  CHECK(bind_ctx.err == conditioner_error::invalid_argument);
  CHECK_FALSE(bind_ctx.result);
  emel::text::conditioner::action::on_unexpected(prepare_runtime, ctx);
  CHECK(prepare_ctx.token_count == 0);
  CHECK(prepare_ctx.err == conditioner_error::invalid_argument);
  CHECK_FALSE(prepare_ctx.result);
}
