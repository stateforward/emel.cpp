#include "test_support.hpp"

TEST_CASE("unicode_helpers_cover_common_paths") {
  CHECK(emel::text::unicode_len_utf8('a') == 1);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xC2)) == 2);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xE2)) == 3);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xF0)) == 4);

  const std::string euro = emel::text::unicode_cpt_to_utf8(0x20AC);
  const std::vector<uint32_t> cpts = emel::text::unicode_cpts_from_utf8("A" + euro);
  CHECK(cpts.size() == 2);
  CHECK(cpts[0] == static_cast<uint32_t>('A'));
  CHECK(cpts[1] == 0x20AC);

  const emel::text::unicode_cpt_flags flags_a =
    emel::text::unicode_cpt_flags_from_cpt(static_cast<uint32_t>('A'));
  CHECK(flags_a.is_letter);
  CHECK(!flags_a.is_number);
  CHECK(!flags_a.is_whitespace);

  CHECK(emel::text::unicode_cpt_is_han(0x4E00));
}

TEST_CASE("unicode_regex_split_custom_paths") {
  const std::string text = "hello 123!";
  const auto cpts = emel::text::unicode_cpts_from_utf8(text);
  const std::vector<size_t> offsets{cpts.size()};

  const auto gpt2_offsets = emel::text::unicode_regex_split_custom_gpt2(text, offsets);
  CHECK(!gpt2_offsets.empty());
  CHECK(sum_offsets(gpt2_offsets) == cpts.size());

  const auto llama3_offsets = emel::text::unicode_regex_split_custom_llama3(text, offsets);
  CHECK(!llama3_offsets.empty());
  CHECK(sum_offsets(llama3_offsets) == cpts.size());

  const auto kimi_offsets = emel::text::unicode_regex_split_custom_kimi_k2(text, offsets);
  CHECK(!kimi_offsets.empty());
  CHECK(sum_offsets(kimi_offsets) == cpts.size());

  const auto afmoe_offsets = emel::text::unicode_regex_split_custom_afmoe(text, offsets);
  CHECK(!afmoe_offsets.empty());
  CHECK(sum_offsets(afmoe_offsets) == cpts.size());
}

TEST_CASE("unicode_regex_split_collapsed_categories") {
  std::vector<std::string> exprs = {
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  };
  const std::vector<std::string> parts = emel::text::unicode_regex_split("hello 42", exprs);
  CHECK(!parts.empty());
  CHECK(parts.front().find("hello") != std::string::npos);
}

TEST_CASE("encoder_rejects_invalid_input") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("hello", 0.5f, 1);

  emel::text::encoders::bpe::sm machine{};

  int32_t token_count = 7;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(!machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hello",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
}

TEST_CASE("encoder_dispatch_callbacks") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.vocab->ignore_merges = true;
  builder.add_token("hello", 0.5f, 1);

  emel::text::encoders::bpe::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  dispatch_recorder recorder{};

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hello",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
    .owner_sm = &recorder,
    .dispatch_done = record_done,
    .dispatch_error = record_error,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(recorder.done_count == 1);
  CHECK(recorder.error_count == 0);
}

TEST_CASE("encoder_dispatch_error_on_missing_bytes") {
  vocab_builder builder{};
  builder.set_model("unknown");

  emel::text::encoders::fallback::sm machine{};

  std::array<int32_t, 1> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  dispatch_recorder recorder{};

  CHECK(!machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "x",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
    .owner_sm = &recorder,
    .dispatch_done = record_done,
    .dispatch_error = record_error,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend));
  CHECK(recorder.done_count == 0);
  CHECK(recorder.error_count == 1);
}

TEST_CASE("encoder_unexpected_event_is_handled") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("hello", 0.5f, 1);

  emel::text::encoders::bpe::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  dispatch_recorder recorder{};

  emel::text::encoders::event::encode request{
    .vocab = *builder.vocab,
    .text = "hello",
    .preprocessed = true,
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
    .owner_sm = &recorder,
    .dispatch_done = record_done,
    .dispatch_error = record_error,
  };

  CHECK(machine.process_event(emel::text::encoders::events::encoding_done{request, 0}));
  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  CHECK(recorder.error_count == 1);
}

TEST_CASE("encoder_detail_trie_basic") {
  emel::text::encoders::detail::naive_trie trie{};
  trie.insert("ab", 2, 42);

  const auto * node = trie.traverse('a');
  CHECK(node != nullptr);
  const auto * node_b = node->traverse('b');
  CHECK(node_b != nullptr);
  CHECK(node_b->has_value);
  CHECK(node_b->value == 42);
  CHECK(node->traverse('z') == nullptr);
}

TEST_CASE("encoder_ensure_tables_populates_state") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.2f, 1);
  builder.add_token("b", 0.1f, 4);
  builder.add_token("c", 0.3f, 5);
  builder.add_merge("a b");
  builder.set_charsmap_a_to_b();

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;

  CHECK(emel::text::encoders::detail::ensure_tables(ctx));
  CHECK(ctx.tables_ready);
  CHECK(!ctx.token_to_id.empty());
  CHECK(!ctx.bpe_ranks.empty());
  CHECK(ctx.ugm_ready);
}

TEST_CASE("encoder_guard_validates_inputs") {
  vocab_builder builder{};
  std::array<int32_t, 2> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  emel::text::encoders::event::encode valid{
    .vocab = *builder.vocab,
    .text = "ok",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  emel::text::encoders::event::encode invalid{
    .vocab = *builder.vocab,
    .text = "bad",
    .token_ids = std::span<int32_t>(),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  emel::text::encoders::event::encode missing_count{
    .vocab = *builder.vocab,
    .text = "bad",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = nullptr,
    .error_out = &err,
  };

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;

  emel::text::encoders::event::encode_ctx valid_ctx{};
  emel::text::encoders::event::encode_ctx invalid_ctx{};
  emel::text::encoders::event::encode_ctx missing_count_ctx{};
  emel::text::encoders::event::encode_runtime valid_runtime{valid, valid_ctx};
  emel::text::encoders::event::encode_runtime invalid_runtime{invalid, invalid_ctx};
  emel::text::encoders::event::encode_runtime missing_count_runtime{missing_count,
                                                                    missing_count_ctx};

  CHECK(emel::text::encoders::guard::valid_encode{}(valid_runtime, ctx));
  CHECK(!emel::text::encoders::guard::valid_encode{}(invalid_runtime, ctx));
  CHECK(emel::text::encoders::guard::valid_encode{}(missing_count_runtime, ctx));

  emel::text::encoders::action::context empty_ctx{};
  CHECK(emel::text::encoders::guard::valid_encode{}(valid_runtime, empty_ctx));
}

TEST_CASE("encoder_detail_misc_branches") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t raw_x = builder.add_token("x", 0.0f, 1);
  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  CHECK(emel::text::encoders::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('x'),
    emel::model::data::tokenizer_model::SPM) == raw_x);

  CHECK(emel::text::encoders::detail::token_text(*builder.vocab, -1).empty());
  CHECK(!emel::text::encoders::detail::is_token_type(*builder.vocab, -1, 1));

  const std::array<char, 1> empty_arr{{'\0'}};
  const auto empty_view =
    emel::text::encoders::detail::string_view_from_array(empty_arr);
  CHECK(empty_view.empty());
}

TEST_CASE("unicode_helpers_extra_branches") {
  const std::string invalid_utf8 = "\xC3\x28";
  const auto invalid_cpts = emel::text::unicode_cpts_from_utf8(invalid_utf8);
  CHECK(!invalid_cpts.empty());

  const std::string b = emel::text::unicode_byte_to_utf8(0x62);
  CHECK(emel::text::unicode_utf8_to_byte(b) == 0x62);

  const auto flags_empty = emel::text::unicode_cpt_flags_from_utf8(std::string{});
  CHECK(!flags_empty.is_letter);
  CHECK(!flags_empty.is_number);
  CHECK(!flags_empty.is_whitespace);
}

TEST_CASE("encoder_detail_helpers") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t hello_id = builder.add_token("hello", 0.5f, 1);
  const int32_t world_id = builder.add_token("world", 0.4f, 1);
  builder.add_byte_token(static_cast<uint8_t>('!'));

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  const std::array<char, 6> hello_arr{{'h', 'e', 'l', 'l', 'o', '\0'}};
  const auto view = emel::text::encoders::detail::string_view_from_array(hello_arr);
  CHECK(view == "hello");
  const std::array<char, 3> xyz_arr{{'x', 'y', 'z'}};
  const auto view_full = emel::text::encoders::detail::string_view_from_array(xyz_arr);
  CHECK(view_full.size() == 3);
  CHECK(emel::text::encoders::detail::utf8_len(static_cast<char>(0x7F)) == 1);
  CHECK(emel::text::encoders::detail::utf8_len(static_cast<char>(0xC2)) == 2);
  CHECK(emel::text::encoders::detail::utf8_len(static_cast<char>(0xE2)) == 3);
  CHECK(emel::text::encoders::detail::utf8_len(static_cast<char>(0xF0)) == 4);

  CHECK(emel::text::encoders::detail::token_text(*builder.vocab, hello_id) == "hello");
  CHECK(emel::text::encoders::detail::is_token_type(*builder.vocab, world_id, 1));

  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::BPE);
  CHECK(emel::text::encoders::detail::lookup_token(ctx, "hello") == hello_id);

  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .text = "hello",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  int32_t count = 0;
  CHECK(emel::text::encoders::detail::push_token(ev, hello_id, count));
  CHECK(count == 1);
  CHECK(!emel::text::encoders::detail::push_token(ev, world_id, count));

  std::vector<std::string_view> parts;
  emel::text::encoders::detail::split_whitespace("a  b\tc", parts);
  std::vector<std::string_view> non_empty;
  for (const auto part : parts) {
    if (!part.empty()) {
      non_empty.push_back(part);
    }
  }
  REQUIRE(non_empty.size() >= 3);
  auto has_token = [] (const std::vector<std::string_view> & values,
                        const std::string_view value) {
    for (const auto item : values) {
      if (item == value) {
        return true;
      }
    }
    return false;
  };
  CHECK(has_token(non_empty, "a"));
  CHECK(has_token(non_empty, "b"));
  CHECK(has_token(non_empty, "c"));

  const auto wpm_tokens = emel::text::encoders::wpm::detail::wpm_preprocess("hello!");
  CHECK(!wpm_tokens.empty());

  CHECK(emel::text::encoders::detail::is_chinese_char(0x4E00));
  CHECK(!emel::text::encoders::detail::is_chinese_char(0x0041));
  CHECK(!emel::text::encoders::detail::cpt_to_utf8(0x24).empty());
  CHECK(emel::text::encoders::detail::cpt_to_utf8(0x00A2).size() == 2);
  CHECK(emel::text::encoders::detail::cpt_to_utf8(0x20AC).size() == 3);
  CHECK(emel::text::encoders::detail::cpt_to_utf8(0x1F4A9).size() == 4);
  CHECK(emel::text::encoders::detail::byte_to_utf8_table()[static_cast<size_t>('A')] == "A");
  CHECK(!emel::text::encoders::detail::byte_to_utf8_table()[0x01].empty());

  emel::text::encoders::ugm::action::context ugm_ctx{};
  ugm_ctx.vocab = builder.vocab;
  std::string_view normalized{};
  CHECK(emel::text::encoders::ugm::detail::normalize_ugm_into(
    *builder.vocab, ugm_ctx, "hello", normalized));
  CHECK(!normalized.empty());
}

TEST_CASE("encoder_encode_impl_variants") {
  auto run_variant = [](
    const char * model,
    const char * pre,
    std::string_view text,
    const auto & setup_vocab) {
      vocab_builder builder{};
      builder.set_model(model);
      if (pre != nullptr) {
        builder.set_pre(pre);
      }
      setup_vocab(builder);

      std::array<int32_t, 32> out_tokens = {};
      int32_t token_count = 0;
      int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
      emel::text::encoders::event::encode ev{
        .text = text,
        .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
        .token_count_out = &token_count,
        .error_out = &err,
      };

      const auto model_id = builder.vocab->tokenizer_model_id;
      emel::text::encoders::detail::encode_result result{};
      switch (model_id) {
        case emel::model::data::tokenizer_model::SPM: {
          emel::text::encoders::spm::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));
          result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::BPE: {
          emel::text::encoders::bpe::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::text::encoders::detail::ensure_tables(ctx));
          ev.preprocessed = true;
          result = emel::text::encoders::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::WPM: {
          emel::text::encoders::wpm::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::text::encoders::wpm::detail::ensure_wpm_tables(ctx, *builder.vocab));
          result = emel::text::encoders::wpm::detail::encode_wpm_ready_tables(
            ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::UGM: {
          emel::text::encoders::ugm::sm machine{};
          emel::text::encoders::event::encode ev_ugm{
            .vocab = *builder.vocab,
            .text = ev.text,
            .preprocessed = ev.preprocessed,
            .token_ids = ev.token_ids,
            .token_count_out = ev.token_count_out,
            .error_out = ev.error_out,
            .owner_sm = ev.owner_sm,
            .dispatch_done = ev.dispatch_done,
            .dispatch_error = ev.dispatch_error,
          };
          (void)machine.process_event(ev_ugm);
          result.token_count = token_count;
          result.error = err;
          break;
        }
        case emel::model::data::tokenizer_model::RWKV: {
          emel::text::encoders::rwkv::sm machine{};
          emel::text::encoders::event::encode ev_rwkv{
            .vocab = *builder.vocab,
            .text = ev.text,
            .preprocessed = ev.preprocessed,
            .token_ids = ev.token_ids,
            .token_count_out = ev.token_count_out,
            .error_out = ev.error_out,
            .owner_sm = ev.owner_sm,
            .dispatch_done = ev.dispatch_done,
            .dispatch_error = ev.dispatch_error,
          };
          (void)machine.process_event(ev_rwkv);
          result.token_count = token_count;
          result.error = err;
          break;
        }
        case emel::model::data::tokenizer_model::PLAMO2: {
          emel::text::encoders::plamo2::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::text::encoders::detail::ensure_tables(ctx));
          result = emel::text::encoders::plamo2::detail::encode_plamo2(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::UNKNOWN: {
          emel::text::encoders::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::text::encoders::fallback::detail::ensure_fallback_tables(ctx, *builder.vocab));
          result = emel::text::encoders::fallback::detail::encode_fallback_exec(
            ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::NONE:
          result.error = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
          break;
      }
      if (ev.token_count_out != nullptr) {
        *ev.token_count_out = result.token_count;
      }
      if (ev.error_out != nullptr) {
        *ev.error_out = result.error;
      }
      (void)result;
      CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
    };

  run_variant("gpt2", "gpt2", "hello world", [] (vocab_builder & builder) {
    builder.add_token("hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    builder.add_token("h", 0.1f, 1);
    builder.add_token("e", 0.1f, 1);
    builder.add_merge("h e");
    builder.add_byte_token(static_cast<uint8_t>(' '));
  });

  run_variant("bert", nullptr, "hello", [] (vocab_builder & builder) {
    builder.add_token("he", 0.2f, 1);
    builder.add_token("##llo", 0.2f, 1);
    builder.add_token("<unk>", 0.0f, 2);
  });

  run_variant("t5", "gpt2", "hello world", [] (vocab_builder & builder) {
    builder.add_token("\xE2\x96\x81hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    builder.set_charsmap_a_to_b();
  });

  run_variant("rwkv", nullptr, "r", [] (vocab_builder & builder) {
    builder.add_byte_token(static_cast<uint8_t>('r'));
  });

  run_variant("plamo2", nullptr, "p", [] (vocab_builder & builder) {
    builder.add_token("<unk>", 0.0f, 2);
    builder.add_all_plamo2_byte_tokens();
    builder.add_plamo2_byte_token(static_cast<uint8_t>('p'));
  });

  run_variant("unknown", nullptr, "x", [] (vocab_builder & builder) {
    builder.add_byte_token(static_cast<uint8_t>('x'));
  });
}

TEST_CASE("encoder_detail_encode_direct_calls") {
  std::array<int32_t, 32> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .text = "hello world",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  {
    vocab_builder builder{};
    builder.set_model("gpt2");
    builder.set_pre("gpt2");
    builder.add_token("h", 0.1f, 1);
    builder.add_token("e", 0.1f, 1);
    builder.add_merge("h e");
    builder.add_byte_token(static_cast<uint8_t>(' '));
    emel::text::encoders::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::detail::ensure_tables(ctx));
    emel::text::encoders::event::encode ev_plain = ev;
    ev_plain.preprocessed = true;
    auto result = emel::text::encoders::bpe::detail::encode_bpe(ev_plain, ctx, *builder.vocab);
    (void)result;
    emel::text::encoders::event::encode ev_punct = ev;
    ev_punct.text = "hello, world!";
    ev_punct.preprocessed = true;
    auto result_punct = emel::text::encoders::bpe::detail::encode_bpe(ev_punct, ctx, *builder.vocab);
    (void)result_punct;
    emel::text::encoders::event::encode ev_empty = ev;
    ev_empty.text = "";
    ev_empty.preprocessed = true;
    auto result_empty = emel::text::encoders::bpe::detail::encode_bpe(ev_empty, ctx, *builder.vocab);
    (void)result_empty;
  }

  {
    vocab_builder builder{};
    builder.set_model("bert");
    builder.add_token("he", 0.2f, 1);
    builder.add_token("##llo", 0.2f, 1);
    builder.add_token("<unk>", 0.0f, 2);
    emel::text::encoders::wpm::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::wpm::detail::ensure_wpm_tables(ctx, *builder.vocab));
    emel::text::encoders::event::encode ev_wpm = ev;
    ev_wpm.text = "unaffable";
    auto result = emel::text::encoders::wpm::detail::encode_wpm_ready_tables(
      ev_wpm, ctx, *builder.vocab);
    (void)result;
    emel::text::encoders::event::encode ev_unknown = ev;
    ev_unknown.text = "xyzxyz";
    auto result_unknown = emel::text::encoders::wpm::detail::encode_wpm_ready_tables(
      ev_unknown, ctx, *builder.vocab);
    (void)result_unknown;
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    builder.set_pre("gpt2");
    builder.add_token("\xE2\x96\x81hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    builder.set_charsmap_a_to_b();
    emel::text::encoders::spm::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));
    emel::text::encoders::event::encode ev_spm = ev;
    ev_spm.text = "hello world";
    auto result = emel::text::encoders::spm::detail::encode_spm(ev_spm, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    builder.add_token("\xE2\x96\x81hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    emel::text::encoders::ugm::sm machine{};
    emel::text::encoders::event::encode ev_ugm{
      .vocab = *builder.vocab,
      .text = "hello",
      .token_ids = ev.token_ids,
      .token_count_out = ev.token_count_out,
      .error_out = ev.error_out,
    };
    CHECK(machine.process_event(ev_ugm));
    CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  }

  {
    vocab_builder builder{};
    builder.set_model("rwkv");
    builder.add_byte_token(static_cast<uint8_t>('r'));
    emel::text::encoders::rwkv::sm machine{};
    emel::text::encoders::event::encode ev_rwkv{
      .vocab = *builder.vocab,
      .text = ev.text,
      .token_ids = ev.token_ids,
      .token_count_out = ev.token_count_out,
      .error_out = ev.error_out,
    };
    CHECK(machine.process_event(ev_rwkv));
    CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  }

  {
    vocab_builder builder{};
    builder.set_model("plamo2");
    builder.add_token("<unk>", 0.0f, 2);
    builder.add_all_plamo2_byte_tokens();
    builder.add_plamo2_byte_token(static_cast<uint8_t>('p'));
    emel::text::encoders::plamo2::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::detail::ensure_tables(ctx));
    auto result = emel::text::encoders::plamo2::detail::encode_plamo2(ev, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("unknown");
    builder.add_byte_token(static_cast<uint8_t>('x'));
    emel::text::encoders::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::fallback::detail::ensure_fallback_tables(ctx, *builder.vocab));
    auto result = emel::text::encoders::fallback::detail::encode_fallback_exec(
      ev, ctx, *builder.vocab);
    (void)result;
  }
}

TEST_CASE("encoder_detail_branch_coverage") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t hex_id = builder.add_token("<0x41>", 0.0f, 1);
  const int32_t raw_id = builder.add_token("A", 0.0f, 1);
  const int32_t byte_id = builder.add_byte_token(static_cast<uint8_t>('!'));

  emel::text::encoders::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  CHECK(emel::text::encoders::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('A'),
    emel::model::data::tokenizer_model::SPM) == hex_id);
  CHECK(emel::text::encoders::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('!'),
    emel::model::data::tokenizer_model::BPE) == byte_id);
  CHECK(emel::text::encoders::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('A'),
    emel::model::data::tokenizer_model::UNKNOWN) == raw_id);

  builder.set_model("t5");
  builder.vocab->escape_whitespaces = true;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->add_space_prefix = true;
  builder.vocab->remove_extra_whitespaces = true;
  emel::text::encoders::ugm::action::context ugm_ctx{};
  ugm_ctx.vocab = builder.vocab;
  std::string_view normalized{};
  CHECK(emel::text::encoders::ugm::detail::normalize_ugm_into(
    *builder.vocab, ugm_ctx, "  a", normalized));
  CHECK(!normalized.empty());

  builder.set_model("none");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::NONE);
  builder.set_model("no_vocab");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::NONE);
  builder.set_model("llama");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::SPM);
  builder.set_model("bert");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::WPM);
  builder.set_model("t5");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::UGM);
  builder.set_model("rwkv");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::RWKV);
  builder.set_model("plamo2");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::PLAMO2);
  builder.set_model("unknown");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::UNKNOWN);

  builder.set_pre("");
}

TEST_CASE("encoder_detail_merge_and_token_helpers") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t token_id = builder.add_token("token", 0.2f, 1);
  const int32_t empty_id = builder.add_token("x", 0.1f, 1);
  builder.add_merge("a b");

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  CHECK(emel::text::encoders::detail::token_text(*builder.vocab, -1).empty());
  builder.vocab->entries[static_cast<uint32_t>(empty_id)].text_length = 0;
  CHECK(emel::text::encoders::detail::token_text(*builder.vocab, empty_id).empty());

  CHECK(emel::text::encoders::detail::merge_text(*builder.vocab, -1).empty());
  const uint32_t original_len = builder.vocab->merge_lengths[0];
  builder.vocab->merge_lengths[0] =
      static_cast<uint32_t>(builder.vocab->merge_storage.size() + 1);
  CHECK(emel::text::encoders::detail::merge_text(*builder.vocab, 0).empty());
  builder.vocab->merge_lengths[0] = original_len;
  CHECK(!emel::text::encoders::detail::merge_text(*builder.vocab, 0).empty());

  CHECK(!emel::text::encoders::detail::merge_match("", "a", "b"));
  CHECK(!emel::text::encoders::detail::merge_match("ab", "a", "b"));
  CHECK(!emel::text::encoders::detail::merge_match("b a", "a", "b"));
  CHECK(!emel::text::encoders::detail::merge_match("a b c", "a", "b"));
  CHECK(!emel::text::encoders::detail::merge_match("a x", "a", "b"));
  CHECK(emel::text::encoders::detail::merge_match("a b", "a", "b"));

  CHECK(emel::text::encoders::detail::hash_sv("token") != 0u);
  CHECK(emel::text::encoders::detail::hash_pair("a", "b") != 0u);

  emel::text::encoders::detail::token_map token_map{};
  CHECK(emel::text::encoders::detail::insert_token_map(
    token_map, *builder.vocab, "", token_id));
  CHECK(emel::text::encoders::detail::insert_token_map(
    token_map, *builder.vocab, "token", token_id));
  CHECK(emel::text::encoders::detail::insert_token_map(
    token_map, *builder.vocab, "token", token_id + 1));
  CHECK(token_map.count >= 1);

  emel::text::encoders::detail::merge_map merge_map{};
  CHECK(!emel::text::encoders::detail::insert_merge_map(
    merge_map, "", "b", 0, *builder.vocab));
  CHECK(emel::text::encoders::detail::insert_merge_map(
    merge_map, "a", "b", 0, *builder.vocab));
  CHECK(emel::text::encoders::detail::insert_merge_map(
    merge_map, "a", "b", 1, *builder.vocab));
  CHECK(merge_map.count >= 1);
}

TEST_CASE("encoder_detail_lookup_helpers") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t token_a = builder.add_token("a", 0.1f, 1);
  const int32_t token_ab = builder.add_token("ab", 0.1f, 1);
  const int32_t token_ac = builder.add_token("ac", 0.1f, 1);
  const int32_t token_cb = builder.add_token("cb", 0.1f, 1);
  builder.add_merge("a b");

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));

  CHECK(emel::text::encoders::detail::lookup_token(ctx, "") ==
        emel::text::encoders::detail::k_token_null);
  CHECK(emel::text::encoders::detail::lookup_token(ctx, "missing") ==
        emel::text::encoders::detail::k_token_null);

  const uint32_t hash = emel::text::encoders::detail::hash_concat("a", "b");
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1;
  const uint32_t slot = hash & mask;

  ctx.token_to_id.hashes[slot] = hash;
  ctx.token_to_id.values[slot] = token_cb;
  int32_t concat = emel::text::encoders::detail::lookup_token_concat(ctx, "a", "b");
  CHECK((concat == token_ab || concat == emel::text::encoders::detail::k_token_null));

  ctx.token_to_id.hashes[slot] = hash;
  ctx.token_to_id.values[slot] = token_ac;
  concat = emel::text::encoders::detail::lookup_token_concat(ctx, "a", "b");
  CHECK((concat == token_ab || concat == emel::text::encoders::detail::k_token_null));

  ctx.token_to_id.hashes[slot] = hash;
  ctx.token_to_id.values[slot] = token_a;
  concat = emel::text::encoders::detail::lookup_token_concat(ctx, "a", "b");
  CHECK((concat == token_ab || concat == emel::text::encoders::detail::k_token_null));

  CHECK(emel::text::encoders::detail::lookup_merge_rank(ctx, *builder.vocab, "", "b") ==
        emel::text::encoders::detail::k_token_null);
  CHECK(emel::text::encoders::detail::lookup_merge_rank(ctx, *builder.vocab, "a", "") ==
        emel::text::encoders::detail::k_token_null);
}

TEST_CASE("encoder_detail_ensure_tables_null_vocab") {
  emel::text::encoders::action::context ctx{};
  CHECK(!emel::text::encoders::detail::ensure_tables(ctx));
}

TEST_CASE("encoder_detail_empty_encode_variants") {
  vocab_builder builder{};
  builder.set_model("unknown");

  emel::text::encoders::action::context fallback_ctx{};
  fallback_ctx.vocab = builder.vocab;
  emel::text::encoders::plamo2::action::context plamo2_ctx{};
  plamo2_ctx.vocab = builder.vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .text = "",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto fallback = emel::text::encoders::fallback::detail::encode_fallback_empty_text(
    ev, fallback_ctx, *builder.vocab);
  CHECK(fallback.token_count == 0);
  CHECK(fallback.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));

  emel::text::encoders::rwkv::sm rwkv_machine{};
  CHECK(rwkv_machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = ev.text,
    .token_ids = ev.token_ids,
    .token_count_out = ev.token_count_out,
    .error_out = ev.error_out,
  }));
  CHECK(token_count == 0);
  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));

  const auto plamo2 =
    emel::text::encoders::plamo2::detail::encode_plamo2(ev, plamo2_ctx, *builder.vocab);
  CHECK(plamo2.token_count == 0);
  CHECK(plamo2.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
}

TEST_CASE("encoder_detail_encode_cpt_utf8_branches") {
  char out[4] = {};
  CHECK(emel::text::encoders::detail::encode_cpt_utf8(0x20AC, out) == 3);
  CHECK(emel::text::encoders::detail::encode_cpt_utf8(0x1F4A9, out) == 4);
}

TEST_CASE("encoder_detail_byte_to_token_none") {
  vocab_builder builder{};
  builder.set_model("none");
  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('A'),
    emel::model::data::tokenizer_model::NONE) ==
    emel::text::encoders::detail::k_token_null);
}

TEST_CASE("encoder_detail_ensure_tables_merge_variants") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.1f, 1);
  builder.add_merge("");
  builder.add_merge("nospace");

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::detail::ensure_tables(ctx));
}

TEST_CASE("encoder_detail_xcda_error_paths") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_charsmap_a_to_b();

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;

  const uint32_t original_size = builder.vocab->precompiled_charsmap_size;
  builder.vocab->precompiled_charsmap_size = 2;
  CHECK(!emel::text::encoders::ugm::detail::init_xcda_tables(ctx));
  builder.vocab->precompiled_charsmap_size = original_size;
  CHECK(emel::text::encoders::ugm::detail::init_xcda_tables(ctx));
}

TEST_CASE("encoder_detail_charsmap_into_paths") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_charsmap_a_to_b();

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));

  const std::string a_input = "a";
  const auto mapped_a =
    emel::text::encoders::ugm::detail::normalize_prefix(*builder.vocab, ctx, a_input, 0);
  REQUIRE(mapped_a.normalized != nullptr);
  CHECK(std::string_view(mapped_a.normalized, mapped_a.normalized_len) == "b");
  CHECK(mapped_a.consumed_input == 1);

  const std::string b_input = "b";
  const auto mapped_b =
    emel::text::encoders::ugm::detail::normalize_prefix(*builder.vocab, ctx, b_input, 0);
  REQUIRE(mapped_b.normalized != nullptr);
  CHECK(std::string_view(mapped_b.normalized, mapped_b.normalized_len) == "b");
  CHECK(mapped_b.consumed_input == 1);

  builder.vocab->precompiled_charsmap_size = 0;
  emel::text::encoders::ugm::action::context ctx_no_table{};
  ctx_no_table.vocab = builder.vocab;
  CHECK(!emel::text::encoders::ugm::detail::init_xcda_tables(ctx_no_table));
}

TEST_CASE("encoder_detail_insert_token_map_full") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  const int32_t token_x = builder.add_token("x", 0.0f, 1);
  const int32_t token_y = builder.add_token("y", 0.0f, 1);

  emel::text::encoders::detail::token_map map{};
  const std::string_view target = "y";
  const uint32_t hash = emel::text::encoders::detail::hash_sv(target);
  for (uint32_t i = 0; i < emel::text::encoders::detail::k_token_hash_size; ++i) {
    map.hashes.get()[i] = hash;
    map.values.get()[i] = token_x;
  }

  const bool ok =
      emel::text::encoders::detail::insert_token_map(map, *builder.vocab, target, token_y);
  CHECK_FALSE(ok);
}

TEST_CASE("encoder_detail_insert_merge_map_full") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.add_merge("a b");

  emel::text::encoders::detail::merge_map map{};
  const std::string_view left = "x";
  const std::string_view right = "y";
  const uint32_t hash = emel::text::encoders::detail::hash_pair(left, right);
  for (uint32_t i = 0; i < emel::text::encoders::detail::k_merge_hash_size; ++i) {
    map.hashes.get()[i] = hash;
    map.values.get()[i] = 0;
  }

  const bool ok =
      emel::text::encoders::detail::insert_merge_map(map, left, right, 1, *builder.vocab);
  CHECK_FALSE(ok);
}

TEST_CASE("encoder_detail_lookup_token_full_probe") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  const int32_t token_x = builder.add_token("x", 0.0f, 1);
  builder.add_token("y", 0.0f, 1);

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;
  const std::string_view target = "y";
  const uint32_t hash = emel::text::encoders::detail::hash_sv(target);
  for (uint32_t i = 0; i < emel::text::encoders::detail::k_token_hash_size; ++i) {
    ctx.token_to_id.hashes.get()[i] = hash;
    ctx.token_to_id.values.get()[i] = token_x;
  }

  const int32_t id = emel::text::encoders::detail::lookup_token(ctx, target);
  CHECK(id == emel::text::encoders::detail::k_token_null);
}

TEST_CASE("encoder_encode_branch_cases") {
  std::array<int32_t, 8> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .text = "hello",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  {
    vocab_builder builder{};
    builder.set_model("gpt2");
    builder.set_pre("gpt2");
    builder.add_token("hello", 0.5f, 1);
    builder.vocab->ignore_merges = true;
    emel::text::encoders::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::detail::ensure_tables(ctx));
    ev.preprocessed = true;
    auto result = emel::text::encoders::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    emel::text::encoders::ugm::sm machine{};
    emel::text::encoders::event::encode ev_ugm{
      .vocab = *builder.vocab,
      .text = "xyz",
      .token_ids = ev.token_ids,
      .token_count_out = ev.token_count_out,
      .error_out = ev.error_out,
    };
    (void)machine.process_event(ev_ugm);
  }

  {
    vocab_builder builder{};
    builder.set_model("rwkv");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    emel::text::encoders::rwkv::sm machine{};
    emel::text::encoders::event::encode ev_rwkv{
      .vocab = *builder.vocab,
      .text = "x",
      .token_ids = ev.token_ids,
      .token_count_out = ev.token_count_out,
      .error_out = ev.error_out,
    };
    (void)machine.process_event(ev_rwkv);
  }

  {
    vocab_builder builder{};
    builder.set_model("unknown");
    emel::text::encoders::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::text::encoders::detail::ensure_tables(ctx));
    CHECK(emel::text::encoders::detail::byte_to_token(
      ctx, *builder.vocab, static_cast<uint8_t>('x'),
      emel::model::data::tokenizer_model::NONE) == emel::text::encoders::detail::k_token_null);
  }
}

TEST_CASE("encoder_naive_trie_branches") {
  emel::text::encoders::detail::naive_trie trie{};
  trie.insert("", 0, 7);
  trie.insert("ab", 2, 11);
  trie.insert("ab", 2, 12);

  const auto * node_a = trie.traverse('a');
  REQUIRE(node_a != nullptr);
  const auto * node_b = node_a->traverse('b');
  REQUIRE(node_b != nullptr);
  CHECK(node_b->has_value);
  CHECK(node_b->value == 12);

  CHECK(trie.traverse('z') == nullptr);
}

TEST_CASE("encoder_bigram_comparators") {
  using spm_bigram = emel::text::encoders::detail::spm_bigram;
  std::priority_queue<spm_bigram, spm_bigram::queue_storage, spm_bigram::comparator> spm_queue;
  spm_bigram spm_a;
  spm_a.score = 0.5f;
  spm_a.left = 1;
  spm_bigram spm_b;
  spm_b.score = 1.0f;
  spm_b.left = 0;
  spm_queue.push(spm_a);
  spm_queue.push(spm_b);
  CHECK(spm_queue.top().score == 1.0f);

  using bpe_bigram = emel::text::encoders::detail::bpe_bigram;
  std::priority_queue<bpe_bigram, std::vector<bpe_bigram>, bpe_bigram::comparator> bpe_queue;
  bpe_bigram bpe_a;
  bpe_a.rank = 5;
  bpe_a.left = 2;
  bpe_bigram bpe_b;
  bpe_b.rank = 1;
  bpe_b.left = 3;
  bpe_queue.push(bpe_a);
  bpe_queue.push(bpe_b);
  CHECK(bpe_queue.top().rank == 1);
}

TEST_CASE("encoder_action_guard_wrapper_coverage") {
  std::array<int32_t, 2> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  auto make_event = [&](const char * text, const int32_t capacity,
                        const emel::model::data::vocab * vocab) {
    return emel::text::encoders::event::encode{
      .vocab = *vocab,
      .text = text,
      .preprocessed = true,
      .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(capacity)),
      .token_count_out = &token_count,
      .error_out = &err,
    };
  };

  auto make_invalid_event = [&](const emel::model::data::vocab * vocab) {
    return emel::text::encoders::event::encode{
      .vocab = *vocab,
      .text = "x",
      .preprocessed = true,
      .token_ids = std::span<int32_t>(),
      .token_count_out = &token_count,
      .error_out = &err,
    };
  };

  vocab_builder base_builder{};
  base_builder.set_model("gpt2");
  base_builder.set_pre("gpt2");
  base_builder.add_token("x", 0.1f, 1);

  emel::text::encoders::action::context base_ctx{};
  base_ctx.vocab = base_builder.vocab;
  auto base_ev = make_event("x", 1, base_builder.vocab);
  dispatch_recorder base_recorder{};
  base_ev.owner_sm = &base_recorder;
  base_ev.dispatch_done = record_done;
  base_ev.dispatch_error = record_error;
  auto invalid_ev = make_invalid_event(base_builder.vocab);
  emel::text::encoders::event::encode_ctx base_runtime{};
  emel::text::encoders::event::encode_ctx base_invalid_runtime{};
  emel::text::encoders::event::encode_runtime base_runtime_ev{base_ev, base_runtime};
  emel::text::encoders::event::encode_runtime base_invalid_runtime_ev{invalid_ev,
                                                                      base_invalid_runtime};

  emel::text::encoders::action::reject_invalid_encode(base_runtime_ev, base_ctx);
  CHECK(base_runtime.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  CHECK(base_runtime.token_count == 0);

  base_runtime.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::action::ensure_last_error(base_runtime_ev, base_ctx);
  CHECK(base_runtime.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend));

  emel::text::encoders::action::on_unexpected(base_runtime_ev, base_ctx);
  CHECK(base_runtime.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  token_count = 1;
  err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::action::on_unexpected(
      emel::text::encoders::events::encoding_done{base_ev, 0}, base_ctx);
  CHECK(token_count == 0);
  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
  CHECK(base_recorder.error_count == 1);

  CHECK(emel::text::encoders::guard::valid_encode{}(base_runtime_ev, base_ctx));
  CHECK(emel::text::encoders::guard::invalid_encode{}(base_invalid_runtime_ev, base_ctx));
  CHECK(emel::text::encoders::guard::vocab_unchanged{}(base_runtime_ev, base_ctx));
  base_ctx.vocab = nullptr;
  CHECK(emel::text::encoders::guard::vocab_changed{}(base_runtime_ev, base_ctx));

  {
    vocab_builder builder{};
    builder.set_model("gpt2");
    builder.set_pre("gpt2");
    builder.add_token("x", 0.1f, 1);
    builder.add_byte_token(static_cast<uint8_t>('x'));

    emel::text::encoders::bpe::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};

    emel::text::encoders::bpe::action::begin_encode_sync_vocab(runtime_ok_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    emel::text::encoders::bpe::action::prepare_tables(runtime_ok_ev, ctx);
    CHECK(emel::text::encoders::bpe::guard::direct_word_token_available{}(runtime_ok_ev, ctx));
    CHECK_FALSE(emel::text::encoders::bpe::guard::ignore_merges_enabled{}(runtime_ok_ev, ctx));

    builder.vocab->ignore_merges = true;
    CHECK(emel::text::encoders::bpe::guard::ignore_merges_enabled{}(runtime_ok_ev, ctx));
    CHECK(emel::text::encoders::bpe::guard::direct_word_token_available{}(runtime_ok_ev, ctx));

    emel::text::encoders::bpe::action::run_encode_ignore_merges(runtime_ok_ev, ctx);
    emel::text::encoders::bpe::action::run_encode_merge_path(runtime_error_ev, ctx);
    emel::text::encoders::bpe::action::mark_done(runtime_ok_ev, ctx);
    emel::text::encoders::bpe::action::ensure_last_error(runtime_error_ev, ctx);
    emel::text::encoders::bpe::action::on_unexpected(runtime_ok_ev, ctx);
    emel::text::encoders::bpe::action::begin_encode(runtime_error_ev, ctx);

    CHECK(emel::text::encoders::bpe::guard::valid_encode{}(runtime_ok_ev, ctx));
    CHECK(emel::text::encoders::bpe::guard::invalid_encode{}(runtime_invalid_ev, ctx));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::bpe::guard::encode_result_ok{}(runtime_ok_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(emel::text::encoders::bpe::guard::encode_result_backend_error{}(runtime_ok_ev));
    CHECK_FALSE(emel::text::encoders::bpe::guard::table_prepare_unclassified_error_code{}(runtime_ok_ev));
    CHECK_FALSE(emel::text::encoders::bpe::guard::encode_result_unclassified_error_code{}(runtime_ok_ev));
    runtime_ok.err = static_cast<int32_t>(0x7FFF);
    CHECK(emel::text::encoders::bpe::guard::table_prepare_unclassified_error_code{}(runtime_ok_ev));
    CHECK(emel::text::encoders::bpe::guard::encode_result_unclassified_error_code{}(runtime_ok_ev));
  }

  {
    vocab_builder builder{};
    builder.set_model("bert");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    builder.add_token("x", 0.1f, 1);

    emel::text::encoders::wpm::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};

    emel::text::encoders::wpm::action::begin_encode_sync_vocab(runtime_ok_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    CHECK(emel::text::encoders::wpm::guard::tables_missing{}(runtime_ok_ev, ctx));
    CHECK(emel::text::encoders::wpm::guard::text_non_empty{}(runtime_ok_ev));
    CHECK(emel::text::encoders::wpm::guard::tables_missing{}(runtime_ok_ev, ctx));
    emel::text::encoders::wpm::action::sync_tables(runtime_ok_ev, ctx);
    CHECK(emel::text::encoders::wpm::guard::tables_ready{}(runtime_ok_ev, ctx));
    CHECK(emel::text::encoders::wpm::guard::text_non_empty{}(runtime_ok_ev));
    CHECK(emel::text::encoders::wpm::guard::tables_ready{}(runtime_ok_ev, ctx));
    emel::text::encoders::wpm::action::run_encode(runtime_error_ev, ctx);
    emel::text::encoders::wpm::action::mark_done(runtime_ok_ev, ctx);
    emel::text::encoders::wpm::action::ensure_last_error(runtime_error_ev, ctx);
    emel::text::encoders::wpm::action::on_unexpected(runtime_ok_ev, ctx);
    emel::text::encoders::wpm::action::begin_encode(runtime_error_ev, ctx);

    CHECK(emel::text::encoders::wpm::guard::valid_encode{}(runtime_ok_ev, ctx));
    CHECK(emel::text::encoders::wpm::guard::invalid_encode{}(runtime_invalid_ev, ctx));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::wpm::guard::encode_result_ok{}(runtime_ok_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(emel::text::encoders::wpm::guard::encode_result_backend_error{}(runtime_ok_ev));
    CHECK_FALSE(emel::text::encoders::wpm::guard::table_sync_unclassified_error_code{}(runtime_ok_ev));
    CHECK_FALSE(emel::text::encoders::wpm::guard::encode_result_unclassified_error_code{}(runtime_ok_ev));
    runtime_ok.err = static_cast<int32_t>(0x7FFF);
    CHECK(emel::text::encoders::wpm::guard::table_sync_unclassified_error_code{}(runtime_ok_ev));
    CHECK(emel::text::encoders::wpm::guard::encode_result_unclassified_error_code{}(runtime_ok_ev));
  }

  {
    vocab_builder builder{};
    builder.set_model("llama");
    builder.set_pre("gpt2");
    builder.add_token("x", 0.1f, 1);

    emel::text::encoders::spm::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};
    emel::text::encoders::spm::runtime::encode_runtime runtime_ok_spm_ev{runtime_ok_ev};
    emel::text::encoders::spm::runtime::encode_runtime runtime_error_spm_ev{runtime_error_ev};
    emel::text::encoders::spm::runtime::encode_runtime runtime_invalid_spm_ev{runtime_invalid_ev};

    emel::text::encoders::spm::action::begin_encode_sync_vocab(runtime_ok_spm_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    CHECK(emel::text::encoders::spm::guard::tables_missing{}(runtime_ok_spm_ev, ctx));
    CHECK(emel::text::encoders::spm::guard::text_non_empty{}(runtime_ok_spm_ev));
    CHECK(emel::text::encoders::spm::guard::tables_missing{}(runtime_ok_spm_ev, ctx));
    emel::text::encoders::spm::action::sync_tables(runtime_ok_spm_ev, ctx);
    CHECK(emel::text::encoders::spm::guard::tables_ready{}(runtime_ok_spm_ev, ctx));
    CHECK(emel::text::encoders::spm::guard::text_non_empty{}(runtime_ok_spm_ev));
    CHECK(emel::text::encoders::spm::guard::tables_ready{}(runtime_ok_spm_ev, ctx));
    emel::text::encoders::spm::action::run_prepare(runtime_ok_spm_ev, ctx);
    emel::text::encoders::spm::action::run_merge(runtime_ok_spm_ev, ctx);
    emel::text::encoders::spm::action::run_encode(runtime_error_spm_ev, ctx);
    emel::text::encoders::spm::action::apply_emit_result_failed(runtime_error_spm_ev, ctx);
    emel::text::encoders::spm::action::mark_done(runtime_ok_spm_ev, ctx);
    emel::text::encoders::spm::action::ensure_last_error(runtime_error_spm_ev, ctx);
    emel::text::encoders::spm::action::on_unexpected(runtime_ok_spm_ev, ctx);

    CHECK(emel::text::encoders::spm::guard::valid_encode{}(runtime_ok_spm_ev, ctx));
    CHECK(emel::text::encoders::spm::guard::invalid_encode{}(runtime_invalid_spm_ev, ctx));
    CHECK(emel::text::encoders::spm::guard::emit_result_failed{}(runtime_error_spm_ev));
    runtime_ok_spm_ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::spm::guard::emit_result_ok{}(runtime_ok_spm_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::spm::guard::encode_result_ok{}(runtime_ok_spm_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(emel::text::encoders::spm::guard::encode_result_backend_error{}(runtime_ok_spm_ev));
    emel::text::encoders::spm::action::begin_encode(runtime_error_spm_ev, ctx);
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    builder.add_token("x", 0.1f, 1);

    emel::text::encoders::ugm::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};
    emel::text::encoders::ugm::runtime::encode_runtime runtime_ok_ugm_ev{runtime_ok_ev};
    emel::text::encoders::ugm::runtime::encode_runtime runtime_error_ugm_ev{runtime_error_ev};
    emel::text::encoders::ugm::runtime::encode_runtime runtime_invalid_ugm_ev{runtime_invalid_ev};

    emel::text::encoders::ugm::action::begin_encode_sync_vocab(runtime_ok_ugm_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    CHECK(emel::text::encoders::ugm::guard::tables_missing{}(runtime_ok_ugm_ev, ctx));
    CHECK(emel::text::encoders::ugm::guard::text_non_empty{}(runtime_ok_ugm_ev));
    CHECK(emel::text::encoders::ugm::guard::tables_missing{}(runtime_ok_ugm_ev, ctx));
    emel::text::encoders::ugm::action::sync_tables(runtime_ok_ugm_ev, ctx);
    CHECK(emel::text::encoders::ugm::guard::tables_ready{}(runtime_ok_ugm_ev, ctx));
    CHECK(emel::text::encoders::ugm::guard::text_non_empty{}(runtime_ok_ugm_ev));
    CHECK(emel::text::encoders::ugm::guard::tables_ready{}(runtime_ok_ugm_ev, ctx));
    emel::text::encoders::ugm::action::begin_encode(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::resolve_vocab_unk(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::normalize_input(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::prepare_dp_input(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::run_dp_trace(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::emit_tokens(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::mark_done(runtime_ok_ugm_ev, ctx);
    emel::text::encoders::ugm::action::ensure_last_error(runtime_error_ugm_ev, ctx);
    emel::text::encoders::ugm::action::on_unexpected(runtime_ok_ugm_ev, ctx);

    CHECK(emel::text::encoders::ugm::guard::valid_encode{}(runtime_ok_ugm_ev, ctx));
    CHECK(emel::text::encoders::ugm::guard::invalid_encode{}(runtime_invalid_ugm_ev, ctx));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::ugm::guard::table_sync_ok{}(runtime_ok_ugm_ev));
    CHECK(emel::text::encoders::ugm::guard::normalize_result_ok{}(runtime_ok_ugm_ev));
    runtime_ok_ugm_ev.normalized = std::string_view{"x"};
    CHECK(emel::text::encoders::ugm::guard::input_prepare_result_non_empty_ok{}(runtime_ok_ugm_ev));
    runtime_ok_ugm_ev.normalized = std::string_view{};
    CHECK(emel::text::encoders::ugm::guard::input_prepare_result_empty_ok{}(runtime_ok_ugm_ev));
    CHECK(emel::text::encoders::ugm::guard::dp_forward_result_ok{}(runtime_ok_ugm_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(emel::text::encoders::ugm::guard::table_sync_backend_error{}(runtime_ok_ugm_ev));
    CHECK(emel::text::encoders::ugm::guard::normalize_result_backend_error{}(runtime_ok_ugm_ev));
    CHECK(
      emel::text::encoders::ugm::guard::input_prepare_result_backend_error{}(runtime_ok_ugm_ev));
    CHECK(emel::text::encoders::ugm::guard::dp_forward_result_backend_error{}(runtime_ok_ugm_ev));
  }

  {
    vocab_builder builder{};
    builder.set_model("rwkv");
    builder.add_byte_token(static_cast<uint8_t>('x'));

    emel::text::encoders::rwkv::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};
    emel::text::encoders::rwkv::runtime::encode_runtime runtime_ok_rwkv_ev{runtime_ok_ev};
    emel::text::encoders::rwkv::runtime::encode_runtime runtime_error_rwkv_ev{runtime_error_ev};
    emel::text::encoders::rwkv::runtime::encode_runtime runtime_invalid_rwkv_ev{runtime_invalid_ev};

    emel::text::encoders::rwkv::action::begin_encode_sync_vocab(runtime_ok_rwkv_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    CHECK(emel::text::encoders::rwkv::guard::tables_missing{}(runtime_ok_rwkv_ev, ctx));
    emel::text::encoders::rwkv::action::sync_tables(runtime_ok_rwkv_ev, ctx);
    CHECK(emel::text::encoders::rwkv::guard::tables_ready{}(runtime_ok_rwkv_ev, ctx));
    emel::text::encoders::rwkv::action::begin_encode(runtime_error_rwkv_ev, ctx);
    emel::text::encoders::rwkv::action::resolve_vocab_unk(runtime_error_rwkv_ev, ctx);
    emel::text::encoders::rwkv::action::run_encode(runtime_error_rwkv_ev, ctx);
    emel::text::encoders::rwkv::action::mark_done(runtime_ok_rwkv_ev, ctx);
    emel::text::encoders::rwkv::action::ensure_last_error(runtime_error_rwkv_ev, ctx);
    emel::text::encoders::rwkv::action::on_unexpected(runtime_ok_rwkv_ev, ctx);

    CHECK(emel::text::encoders::rwkv::guard::valid_encode{}(runtime_ok_rwkv_ev, ctx));
    CHECK(emel::text::encoders::rwkv::guard::invalid_encode{}(runtime_invalid_rwkv_ev, ctx));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::rwkv::guard::encode_result_ok{}(runtime_ok_rwkv_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(emel::text::encoders::rwkv::guard::encode_result_backend_error{}(runtime_ok_rwkv_ev));
  }

  {
    vocab_builder builder{};
    builder.set_model("plamo2");
    builder.add_token("<unk>", 0.0f, 2);
    builder.add_all_plamo2_byte_tokens();
    builder.add_plamo2_byte_token(static_cast<uint8_t>('x'));

    emel::text::encoders::plamo2::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};
    emel::text::encoders::plamo2::runtime::encode_runtime runtime_ok_plamo2_ev{runtime_ok_ev};
    emel::text::encoders::plamo2::runtime::encode_runtime runtime_error_plamo2_ev{runtime_error_ev};
    emel::text::encoders::plamo2::runtime::encode_runtime runtime_invalid_plamo2_ev{
      runtime_invalid_ev};

    emel::text::encoders::plamo2::action::begin_encode_sync_vocab(runtime_ok_plamo2_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    emel::text::encoders::plamo2::action::sync_tables(runtime_ok_plamo2_ev, ctx);
    CHECK(emel::text::encoders::plamo2::guard::tables_ready{}(runtime_ok_plamo2_ev, ctx));
    emel::text::encoders::plamo2::action::decode_input(runtime_ok_plamo2_ev, ctx);
    CHECK(emel::text::encoders::plamo2::guard::decode_result_non_empty_ok{}(runtime_ok_plamo2_ev));
    emel::text::encoders::plamo2::action::prepare_dp(runtime_ok_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::run_dp(runtime_ok_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::emit_tokens(runtime_ok_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::sync_tables(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::decode_input(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::prepare_dp(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::run_dp(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::emit_tokens(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::apply_emit_result_failed(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::mark_done(runtime_ok_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::ensure_last_error(runtime_error_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::on_unexpected(runtime_ok_plamo2_ev, ctx);
    emel::text::encoders::plamo2::action::begin_encode(runtime_error_plamo2_ev, ctx);

    CHECK(emel::text::encoders::plamo2::guard::valid_encode{}(runtime_ok_plamo2_ev, ctx));
    CHECK(emel::text::encoders::plamo2::guard::invalid_encode{}(runtime_invalid_plamo2_ev, ctx));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::plamo2::guard::table_sync_ok{}(runtime_ok_plamo2_ev));
    CHECK(emel::text::encoders::plamo2::guard::encode_result_ok{}(runtime_ok_plamo2_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(emel::text::encoders::plamo2::guard::table_sync_backend_error{}(runtime_ok_plamo2_ev));
    CHECK(emel::text::encoders::plamo2::guard::decode_result_backend_error{}(runtime_ok_plamo2_ev));
    CHECK(emel::text::encoders::plamo2::guard::encode_result_backend_error{}(runtime_ok_plamo2_ev));
  }

  {
    vocab_builder builder{};
    builder.set_model("unknown");
    builder.add_byte_token(static_cast<uint8_t>('x'));

    emel::text::encoders::fallback::action::context ctx{};
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);
    emel::text::encoders::event::encode_ctx runtime_ok{};
    emel::text::encoders::event::encode_ctx runtime_error{};
    emel::text::encoders::event::encode_ctx runtime_invalid{};
    emel::text::encoders::event::encode_runtime runtime_ok_ev{ev_ok, runtime_ok};
    emel::text::encoders::event::encode_runtime runtime_error_ev{ev_error, runtime_error};
    emel::text::encoders::event::encode_runtime runtime_invalid_ev{ev_invalid, runtime_invalid};
    emel::text::encoders::fallback::runtime::encode_runtime runtime_ok_fallback_ev{runtime_ok_ev};
    emel::text::encoders::fallback::runtime::encode_runtime runtime_error_fallback_ev{runtime_error_ev};
    emel::text::encoders::fallback::runtime::encode_runtime runtime_invalid_fallback_ev{runtime_invalid_ev};

    emel::text::encoders::fallback::action::begin_encode_sync_vocab(runtime_ok_fallback_ev, ctx);
    CHECK(ctx.vocab == builder.vocab);
    emel::text::encoders::fallback::action::prepare_tables(runtime_ok_fallback_ev, ctx);
    emel::text::encoders::fallback::action::run_encode_exec(runtime_error_fallback_ev, ctx);
    emel::text::encoders::fallback::action::apply_emit_result_failed(runtime_error_fallback_ev, ctx);
    emel::text::encoders::fallback::action::mark_done(runtime_ok_fallback_ev, ctx);
    emel::text::encoders::fallback::action::ensure_last_error(runtime_error_fallback_ev, ctx);
    emel::text::encoders::fallback::action::on_unexpected(runtime_ok_fallback_ev, ctx);

    CHECK(emel::text::encoders::fallback::guard::valid_encode{}(runtime_ok_fallback_ev, ctx));
    CHECK(emel::text::encoders::fallback::guard::invalid_encode{}(runtime_invalid_fallback_ev, ctx));
    CHECK(emel::text::encoders::fallback::guard::emit_result_failed{}(runtime_error_fallback_ev));
    runtime_ok_fallback_ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::fallback::guard::emit_result_ok{}(runtime_ok_fallback_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    CHECK(emel::text::encoders::fallback::guard::table_prepare_ok{}(runtime_ok_fallback_ev));
    CHECK(emel::text::encoders::fallback::guard::encode_result_ok{}(runtime_ok_fallback_ev));
    runtime_ok.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
    CHECK(
      emel::text::encoders::fallback::guard::table_prepare_backend_error{}(runtime_ok_fallback_ev));
    CHECK(
      emel::text::encoders::fallback::guard::encode_result_backend_error{}(runtime_ok_fallback_ev));
    emel::text::encoders::fallback::action::begin_encode(runtime_error_fallback_ev, ctx);
  }
}
