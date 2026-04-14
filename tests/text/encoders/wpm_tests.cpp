#include "test_support.hpp"

TEST_CASE("encoder_wpm_emits_whole_word_token_without_prefix") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t token_id = builder.add_token("hello", 0.2f, 1);
  builder.vocab->unk_id = 0;

  emel::text::encoders::wpm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hello",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 1);
  CHECK(tokens[0] == token_id);
}

TEST_CASE("encoder_wpm_emits_bert_continuation_pieces") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t her_id = builder.add_token("her", 0.2f, 1);
  const int32_t tz_id = builder.add_token("##tz", 0.2f, 1);
  builder.vocab->unk_id = builder.add_token("<unk>", 0.0f, 2);

  emel::text::encoders::wpm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hertz",
    .token_ids = std::span<int32_t>(
        tokens.data(),
        static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(
                   emel::text::encoders::error::code::ok));
  CHECK(token_count == 2);
  CHECK(tokens[0] == her_id);
  CHECK(tokens[1] == tz_id);
}

TEST_CASE("encoder_wpm_prefers_word_start_marker_when_vocab_provides_it") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t hello_id = builder.add_token("\xE2\x96\x81hello", 0.2f, 1);
  const int32_t world_word_start_id = builder.add_token("\xE2\x96\x81world", 0.2f, 1);
  (void) builder.add_token("world", 0.2f, 1);
  builder.vocab->unk_id = builder.add_token("<unk>", 0.0f, 2);

  emel::text::encoders::wpm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hello world",
    .token_ids = std::span<int32_t>(
        tokens.data(),
        static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(
                   emel::text::encoders::error::code::ok));
  CHECK(token_count == 2);
  CHECK(tokens[0] == hello_id);
  CHECK(tokens[1] == world_word_start_id);
}

TEST_CASE("encoder_wpm_falls_back_to_raw_continuation_piece_when_hash_prefix_absent") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t hello_id = builder.add_token("\xE2\x96\x81hello", 0.2f, 1);
  const int32_t world_id = builder.add_token("world", 0.2f, 1);
  builder.vocab->unk_id = builder.add_token("<unk>", 0.0f, 2);

  emel::text::encoders::wpm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "helloworld",
    .token_ids = std::span<int32_t>(
        tokens.data(),
        static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(
                   emel::text::encoders::error::code::ok));
  CHECK(token_count == 2);
  CHECK(tokens[0] == hello_id);
  CHECK(tokens[1] == world_id);
}

TEST_CASE("encoder_wpm_falls_back_to_unk") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
  builder.vocab->unk_id = unk_id;

  emel::text::encoders::wpm::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hello",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 1);
  CHECK(tokens[0] == unk_id);
}

TEST_CASE("encoder_detail_wpm_preprocess_whitespace") {
  const auto parts = emel::text::encoders::wpm::detail::wpm_preprocess("a b");
  CHECK(parts.size() >= 2);
  CHECK(parts[0] == "a");
  CHECK(parts[1] == "b");
}

TEST_CASE("encoder_detail_wpm_empty_text") {
  vocab_builder builder{};
  builder.set_model("bert");
  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::wpm::detail::encode_wpm_empty(ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_wpm_encode_requires_prepared_tables") {
  vocab_builder builder{};
  builder.set_model("bert");
  builder.add_token("hello", 0.2f, 1);

  emel::text::encoders::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "hello",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::wpm::detail::encode_wpm_missing_tables(
    ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_wpm_rejects_continuation_capacity_overflow") {
  vocab_builder builder{};
  builder.set_model("bert");
  builder.add_token("<unk>", 0.0f, 1);

  emel::text::encoders::wpm::sm machine{};
  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  std::string text(emel::text::encoders::detail::k_max_encode_bytes, 'a');

  CHECK_FALSE(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = text,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(tokens.size())),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  CHECK(token_count == 0);
}

TEST_CASE("encoder_detail_wpm_preprocess_punctuation_and_control") {
  const std::string input = std::string("hi,") + "\xEF\xBF\xBD" + "\xE4\xB8\xAD";
  const auto parts = emel::text::encoders::wpm::detail::wpm_preprocess(input);
  CHECK(parts.size() == 3);
  CHECK(parts[0] == "hi");
  CHECK(parts[1] == ",");
  CHECK(parts[2] == std::string("\xE4\xB8\xAD"));
}

TEST_CASE("encoder_detail_wpm_skips_unknown_without_unk") {
  vocab_builder builder{};
  builder.set_model("bert");
  builder.add_token("hello", 0.0f, 1);
  builder.vocab->unk_id = emel::text::encoders::detail::k_token_null;

  emel::text::encoders::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::wpm::detail::ensure_wpm_tables(ctx, *builder.vocab));
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "unknown",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::wpm::detail::encode_wpm_ready_tables(
    ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_wpm_matches_bert_wordpiece_candidates") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t pure_id = builder.add_token("pure", 0.0f, 1);
  const int32_t h440_id = builder.add_token("440", 0.0f, 1);
  const int32_t her_id = builder.add_token("her", 0.0f, 1);
  const int32_t tz_id = builder.add_token("##tz", 0.0f, 1);
  const int32_t sin_id = builder.add_token("sin", 0.0f, 1);
  const int32_t e_id = builder.add_token("##e", 0.0f, 1);
  const int32_t tone_id = builder.add_token("tone", 0.0f, 1);
  emel::text::encoders::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::wpm::detail::ensure_wpm_tables(ctx, *builder.vocab));
  std::array<int32_t, 8> out_tokens = {};
  int32_t token_count = 0;
  int32_t err =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "pure 440 hertz sine tone",
    .token_ids = std::span<int32_t>(
        out_tokens.data(),
        static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::wpm::detail::encode_wpm_ready_tables(
    ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(
                           emel::text::encoders::error::code::ok));
  REQUIRE(result.token_count == 7);
  CHECK(out_tokens[0] == pure_id);
  CHECK(out_tokens[1] == h440_id);
  CHECK(out_tokens[2] == her_id);
  CHECK(out_tokens[3] == tz_id);
  CHECK(out_tokens[4] == sin_id);
  CHECK(out_tokens[5] == e_id);
  CHECK(out_tokens[6] == tone_id);
}

TEST_CASE("encoder_detail_wpm_push_overflow") {
  vocab_builder builder{};
  builder.set_model("bert");
  builder.add_token("a", 0.0f, 1);
  emel::text::encoders::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::wpm::detail::ensure_wpm_tables(ctx, *builder.vocab));
  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "a",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(0)),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::wpm::detail::encode_wpm_ready_tables(
    ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
}
