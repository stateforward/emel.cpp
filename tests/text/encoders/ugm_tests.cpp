#include "test_support.hpp"

TEST_CASE("encoder_ugm_applies_precompiled_charsmap") {
  vocab_builder builder{};
  builder.set_model("t5");
  const int32_t token_id = builder.add_token("b", 0.1f, 1);
  builder.vocab->unk_id = 2;
  builder.vocab->escape_whitespaces = false;
  builder.vocab->add_space_prefix = false;
  builder.set_charsmap_a_to_b();

  emel::text::encoders::ugm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "a",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == token_id);
}

TEST_CASE("encoder_ugm_normalization_flags") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.vocab->escape_whitespaces = true;
  builder.vocab->add_space_prefix = true;
  builder.vocab->remove_extra_whitespaces = true;
  builder.vocab->treat_whitespace_as_suffix = false;

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::string_view normalized{};
  CHECK(emel::text::encoders::ugm::detail::normalize_ugm_into(
    *builder.vocab, ctx, "  hello   world ", normalized));

  CHECK(!normalized.empty());
  CHECK(normalized.find("hello") != std::string_view::npos);
  CHECK(normalized.find("world") != std::string_view::npos);
}

TEST_CASE("encoder_detail_normalize_ugm_into_paths") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_charsmap_a_to_b();
  builder.vocab->escape_whitespaces = true;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->add_space_prefix = true;
  builder.vocab->remove_extra_whitespaces = true;

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::string_view out;
  CHECK(emel::text::encoders::ugm::detail::normalize_ugm_into(
    *builder.vocab, ctx, "a   ", out));
  CHECK(!out.empty());
  CHECK(out.front() == '\xE2');

  std::string_view trimmed;
  CHECK(emel::text::encoders::ugm::detail::normalize_ugm_into(
    *builder.vocab, ctx, "  a  ", trimmed));
  CHECK(!trimmed.empty());
}

TEST_CASE("encoder_detail_ugm_helper_branches") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("", 0.0f, 1);
  builder.add_token("user", 0.0f, 4);
  builder.add_token("a", 0.1f, 1);

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));
  CHECK(emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));

  emel::text::encoders::detail::naive_trie trie{};
  trie.insert("a", 1, 1);
  CHECK(emel::text::encoders::ugm::detail::trie_longest_prefix(trie, "a", 1) == 1);
  CHECK(emel::text::encoders::ugm::detail::trie_longest_prefix(trie, "b", 1) == 0);
  CHECK(emel::text::encoders::ugm::detail::trie_longest_prefix(trie, "a", 0) == 0);

  emel::text::encoders::ugm::detail::xcda_view view{};
  CHECK(view.node(1) == 0);

  const std::string input = "user";
  const auto norm_end =
      emel::text::encoders::ugm::detail::normalize_prefix(*builder.vocab, ctx, input,
                                                   input.size());
  CHECK(norm_end.normalized_len == 0);

  const auto norm_user =
      emel::text::encoders::ugm::detail::normalize_prefix(*builder.vocab, ctx, input, 0);
  CHECK(norm_user.consumed_input == input.size());

  const std::string bad(1, static_cast<char>(0x80));
  const auto norm_bad =
      emel::text::encoders::ugm::detail::normalize_prefix(*builder.vocab, ctx, bad, 0);
  CHECK(norm_bad.consumed_input == 1);
}

TEST_CASE("encoder_detail_ugm_normalize_overflow") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->add_space_prefix = true;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->remove_extra_whitespaces = false;
  builder.vocab->escape_whitespaces = false;

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes, 'a');
  std::string_view normalized;
  CHECK_FALSE(emel::text::encoders::ugm::detail::normalize_ugm_into(*builder.vocab, ctx,
                                                             text, normalized));

  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::ugm::sm machine{};
  CHECK_FALSE(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = text,
    .token_ids = std::span<int32_t>(
      out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_ugm_normalize_empty") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->add_space_prefix = false;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->remove_extra_whitespaces = true;

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::ugm::sm machine{};
  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "   ",
    .token_ids = std::span<int32_t>(
      out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(token_count == 0);
}

TEST_CASE("encoder_ugm_encode_builds_tables_when_missing") {
  vocab_builder builder{};
  builder.set_model("t5");
  const int32_t token_id = builder.add_token("a", 0.0f, 1);
  builder.vocab->unk_id = token_id;

  std::array<int32_t, 2> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::ugm::sm machine{};
  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "a",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(out_tokens.size())),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(out_tokens[0] == token_id);
}

TEST_CASE("encoder_detail_ugm_append_space_and_overflow") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->add_space_prefix = true;
  builder.vocab->treat_whitespace_as_suffix = true;
  builder.vocab->remove_extra_whitespaces = false;
  builder.vocab->escape_whitespaces = false;

  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::string_view normalized;
  CHECK(emel::text::encoders::ugm::detail::normalize_ugm_into(*builder.vocab, ctx, "a",
                                                       normalized));
  CHECK(!normalized.empty());
  CHECK(normalized.back() == ' ');

  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string spaces(max_bytes + 1, ' ');
  CHECK_FALSE(emel::text::encoders::ugm::detail::normalize_ugm_into(*builder.vocab, ctx,
                                                             spaces, normalized));
}

TEST_CASE("encoder_detail_ugm_xcda_break_and_trie_paths") {
  emel::text::encoders::detail::naive_trie trie{};
  trie.insert("a", 1, 1);
  trie.insert("ab", 2, 2);
  CHECK(emel::text::encoders::ugm::detail::trie_longest_prefix(trie, "ac", 2) == 1);

  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  emel::text::encoders::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<uint32_t, 1> table = {0};
  ctx.xcda_table = table.data();
  ctx.xcda_table_size = table.size();
  ctx.prefix_replacements = "";
  ctx.prefix_replacements_size = 0;
  const std::string input(1, '\0');
  const auto norm = emel::text::encoders::ugm::detail::normalize_prefix(*builder.vocab, ctx,
                                                                 input, 0);
  CHECK(norm.consumed_input == 1);
}
