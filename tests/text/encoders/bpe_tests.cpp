#include "test_support.hpp"

TEST_CASE("encoder_bpe_ignore_merges_prefers_full_token") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t full_id = builder.add_token("hello", 0.5f, 1);
  builder.vocab->ignore_merges = true;

  emel::text::encoders::bpe::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hello",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 1);
  CHECK(tokens[0] == full_id);
}

TEST_CASE("encoder_bpe_merges_ranked_pair") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t he_id = builder.add_token("he", 0.9f, 1);
  builder.add_token("h", 0.1f, 1);
  builder.add_token("e", 0.1f, 1);
  builder.add_merge("h e");

  emel::text::encoders::bpe::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "he",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 1);
  CHECK(tokens[0] == he_id);
}

TEST_CASE("encoder_bpe_byte_fallback") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t byte_id = builder.add_byte_token(static_cast<uint8_t>('!'));

  emel::text::encoders::bpe::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "!",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 1);
  CHECK(tokens[0] == byte_id);
}

TEST_CASE("encoder_bpe_byte_fallback_multibyte_symbols") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t byte0_id = builder.add_byte_token(static_cast<uint8_t>(0));
  const int32_t byte1_id = builder.add_byte_token(static_cast<uint8_t>(1));

  const std::string byte0 = emel::text::unicode_byte_to_utf8(0);
  const std::string byte1 = emel::text::unicode_byte_to_utf8(1);
  const std::string merge = byte0 + " " + byte1;
  builder.add_merge(merge.c_str());

  const std::string word = byte0 + byte1;

  emel::text::encoders::bpe::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = word,
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 2);
  CHECK(tokens[0] == byte0_id);
  CHECK(tokens[1] == byte1_id);
}

TEST_CASE("encoder_detail_bpe_merge_and_errors") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("h", 0.1f, 1);
  builder.add_token("e", 0.1f, 1);
  const int32_t he_id = builder.add_token("he", 0.5f, 1);
  builder.add_merge("h e");

  emel::text::encoders::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "he",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto merged = emel::text::encoders::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
  CHECK(merged.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(merged.token_count >= 1);
  CHECK(tokens[0] == he_id);

  builder.vocab->ignore_merges = true;
  emel::text::encoders::bpe::action::context ctx_fail{};
  ctx_fail.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx_fail));

  emel::text::encoders::event::encode ev_fail{
    .vocab = *builder.vocab,
    .text = "he",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result_fail = emel::text::encoders::bpe::detail::encode_bpe(
    ev_fail, ctx_fail, *builder.vocab);
  CHECK(result_fail.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
}

TEST_CASE("encoder_detail_bpe_buffer_overflow") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.1f, 1);

  emel::text::encoders::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  std::string text(70000, 'a');
  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = text,
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
}

TEST_CASE("encoder_bpe_merge_path_rejects_symbol_capacity_overflow") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.1f, 1);

  emel::text::encoders::bpe::sm machine{};
  std::array<int32_t, 1> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  std::string text(emel::text::encoders::detail::k_max_encode_symbols + 1, 'a');

  CHECK_FALSE(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = text,
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(tokens.size())),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  CHECK(token_count == 0);
}

TEST_CASE("encoder_detail_bpe_byte_push_overflow") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.0f, 1);
  builder.add_token("b", 0.0f, 1);
  builder.add_merge("a b");

  emel::text::encoders::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "ab",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(0)),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
}
