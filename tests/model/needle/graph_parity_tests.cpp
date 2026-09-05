#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "doctest/doctest.h"

#include "emel/cact/loader/sm.hpp"
#include "emel/model/needle/graph/sm.hpp"
#include "emel/model/needle/sm.hpp"

#include "../../allocation_tracker.hpp"
#include "emel/model/needle/request/sm.hpp"

namespace {

// Committed CQ4-weight/f32-activation fixture (route-w4-qat.logits.json): 3
// cases, each 1 prefill + 2 greedy decode steps. The explicit A8 training
// parity route is covered by the separate route-w4-qat-a8 fixture test below.
constexpr uint32_t k_vocab = 8192u;
constexpr uint32_t k_steps = 3u;

struct parity_case {
  std::vector<int32_t> prompt_ids;
  std::array<int32_t, k_steps> greedy;
  const char *file;
};

const parity_case k_cases[3] = {
    {{2, 1544, 1663, 2328}, {8097, 341, 359}, "route-w4-qat.logits.case0.bin"},
    {{2, 5722, 625, 5019}, {8063, 24, 7}, "route-w4-qat.logits.case1.bin"},
    {{2, 7551}, {8097, 2730, 8097}, "route-w4-qat.logits.case2.bin"},
};

const parity_case k_a8_cases[3] = {
    {{2, 1544, 1663, 2328},
     {8097, 341, 359},
     "route-w4-qat-a8.logits.case0.bin"},
    {{2, 5722, 625, 5019}, {8063, 24, 7}, "route-w4-qat-a8.logits.case1.bin"},
    {{2, 7551}, {8097, 2730, 8097}, "route-w4-qat-a8.logits.case2.bin"},
};

void on_loader_probe_done(
    const emel::cact::loader::events::probe_done &) noexcept {}
void on_loader_probe_error(
    const emel::cact::loader::events::probe_error &) noexcept {}
void on_loader_bind_done(
    const emel::cact::loader::events::bind_done &) noexcept {}
void on_loader_bind_error(
    const emel::cact::loader::events::bind_error &) noexcept {}
void on_loader_parse_done(
    const emel::cact::loader::events::parse_done &) noexcept {}
void on_loader_parse_error(
    const emel::cact::loader::events::parse_error &) noexcept {}
void on_needle_bind_done(
    const emel::model::needle::events::bind_done &) noexcept {}
void on_needle_bind_error(
    const emel::model::needle::events::bind_error &) noexcept {}

const emel::cact::loader::event::probe_done_fn k_probe_done =
    emel::cact::loader::event::probe_done_fn::from<&on_loader_probe_done>();
const emel::cact::loader::event::probe_error_fn k_probe_error =
    emel::cact::loader::event::probe_error_fn::from<&on_loader_probe_error>();
const emel::cact::loader::event::bind_done_fn k_bind_done =
    emel::cact::loader::event::bind_done_fn::from<&on_loader_bind_done>();
const emel::cact::loader::event::bind_error_fn k_bind_error =
    emel::cact::loader::event::bind_error_fn::from<&on_loader_bind_error>();
const emel::cact::loader::event::parse_done_fn k_parse_done =
    emel::cact::loader::event::parse_done_fn::from<&on_loader_parse_done>();
const emel::cact::loader::event::parse_error_fn k_parse_error =
    emel::cact::loader::event::parse_error_fn::from<&on_loader_parse_error>();
const emel::model::needle::event::bind_done_fn k_needle_done =
    emel::model::needle::event::bind_done_fn::from<&on_needle_bind_done>();
const emel::model::needle::event::bind_error_fn k_needle_error =
    emel::model::needle::event::bind_error_fn::from<&on_needle_bind_error>();

std::vector<uint8_t> read_file_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  REQUIRE(size > 0);
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(input.good());
  return bytes;
}

std::vector<float> read_reference_logits(const char *name) {
  const auto path =
      std::filesystem::path{EMEL_TEST_REPO_ROOT} / "tests/fixtures/cact" / name;
  const auto bytes = read_file_bytes(path);
  REQUIRE(bytes.size() == static_cast<size_t>(k_steps) * k_vocab * 4u);
  std::vector<float> values(static_cast<size_t>(k_steps) * k_vocab);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

uint32_t argmax(const std::span<const float> logits) {
  uint32_t best = 0u;
  for (uint32_t i = 1u; i < logits.size(); ++i)
    best = logits[i] > logits[best] ? i : best;
  return best;
}

struct step_error {
  double max_abs = 0.0;
  double rel = 0.0;
};

step_error compare_step(const std::span<const float> native,
                        const std::span<const float> reference) {
  step_error result{};
  double max_ref = 0.0;
  for (uint32_t i = 0u; i < k_vocab; ++i) {
    const double diff = std::abs(static_cast<double>(native[i]) -
                                 static_cast<double>(reference[i]));
    result.max_abs = diff > result.max_abs ? diff : result.max_abs;
    const double magnitude = std::abs(static_cast<double>(reference[i]));
    max_ref = magnitude > max_ref ? magnitude : max_ref;
  }
  result.rel = result.max_abs / max_ref;
  return result;
}

struct loaded_contract_fixture {
  std::vector<uint8_t> file_bytes;
  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  emel::model::needle::contract contract = {};
};

loaded_contract_fixture load_contract_fixture() {
  loaded_contract_fixture fixture{};
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  fixture.file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  REQUIRE(loader.process_event(emel::cact::loader::event::probe{
      std::span<const uint8_t>{fixture.file_bytes}, fixture.geometry,
      k_probe_done, k_probe_error}));
  fixture.tensors.resize(fixture.geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{fixture.tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{fixture.file_bytes}, k_parse_done,
      k_parse_error}));
  emel::model::needle::sm binder{};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      fixture.geometry,
      std::span<const emel::cact::loader::tensor_view>{fixture.tensors},
      fixture.contract, k_needle_done, k_needle_error}));
  return fixture;
}

void check_all_equal(const std::span<const float> values,
                     const float expected) {
  for (const float value : values)
    CHECK(value == expected);
}



uint64_t hash_ids(const std::span<const int32_t> ids) noexcept {
  uint64_t hash = 1469598103934665603ULL;
  for (const int32_t id : ids) {
    const uint32_t value = static_cast<uint32_t>(id);
    for (uint32_t byte = 0u; byte < 4u; ++byte) {
      hash ^= static_cast<uint8_t>(value >> (byte * 8u));
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

struct request_prompt_fixture {
  std::vector<int32_t> ids;
  std::string prompt;
};

std::vector<request_prompt_fixture> first_request_prompts() {
  const auto path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                    "tests/fixtures/cact/needle-heldout-prompts.tsv";
  std::ifstream input(path);
  REQUIRE(input.good());
  std::vector<request_prompt_fixture> rows;
  std::string line;
  while (rows.size() < 4u && std::getline(input, line)) {
    const size_t first = line.find('\t');
    const size_t second = line.find('\t', first + 1u);
    const size_t third = line.find('\t', second + 1u);
    REQUIRE(first != std::string::npos);
    REQUIRE(second != std::string::npos);
    REQUIRE(third != std::string::npos);
    request_prompt_fixture row{};
    const std::string ids = line.substr(second + 1u, third - second - 1u);
    size_t cursor = 0u;
    while (cursor < ids.size()) {
      char *end = nullptr;
      row.ids.push_back(static_cast<int32_t>(
          std::strtol(ids.c_str() + cursor, &end, 10)));
      REQUIRE(end != ids.c_str() + cursor);
      cursor = static_cast<size_t>(end - ids.c_str());
      while (cursor < ids.size() && ids[cursor] == ' ') ++cursor;
    }
    const std::string hex = line.substr(third + 1u);
    REQUIRE(hex.size() % 2u == 0u);
    row.prompt.reserve(hex.size() / 2u);
    for (size_t i = 0u; i < hex.size(); i += 2u)
      row.prompt.push_back(static_cast<char>(
          std::stoi(hex.substr(i, 2u), nullptr, 16)));
    rows.push_back(std::move(row));
  }
  REQUIRE(rows.size() == 4u);
  return rows;
}
constexpr std::string_view k_route_tools_json =
    R"json([{"name":"route","description":"Route a request to a domain queue with the minimum sufficient reasoning effort","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}}])json";

std::string replace_once(std::string input, const std::string_view from,
                         const std::string_view to) {
  const size_t at = input.find(from);
  REQUIRE(at != std::string::npos);
  input.replace(at, from.size(), to);
  return input;
}

std::string with_route_ancillary(const std::string_view value) {
  std::string result{k_route_tools_json};
  const size_t route_end = result.rfind('}');
  REQUIRE(route_end != std::string::npos);
  result.insert(route_end, value);
  return result;
}

struct request_callback_state {
  uint32_t done_count = 0u;
  uint32_t error_count = 0u;
  emel::error::type err =
      emel::error::cast(emel::model::needle::request::error::none);
};

request_callback_state *g_request_callback_state = nullptr;

void on_request_reset_done(
    const emel::model::needle::request::events::reset_done &) noexcept {
  ++g_request_callback_state->done_count;
}

void on_request_completed(
    const emel::model::needle::request::events::completed &) noexcept {
  ++g_request_callback_state->done_count;
}

void on_request_error(
    const emel::model::needle::request::events::request_error &ev) noexcept {
  ++g_request_callback_state->error_count;
  g_request_callback_state->err = ev.err;
}

TEST_CASE("needle request illegal runtime dispatch reports internal error without stale success") {
  namespace request = emel::model::needle::request;

  auto fixture = load_contract_fixture();
  constexpr std::string_view query = "hello";

  const auto check_failure = [](request_callback_state &callbacks,
                                const bool accepted) {
    CHECK_FALSE(accepted);
    CHECK(callbacks.done_count == 0u);
    CHECK(callbacks.error_count == 1u);
    CHECK(callbacks.err ==
          emel::error::cast(request::error::internal_error));
  };

  SUBCASE("wrapped and unwrapped runtime events set their originating context") {
    request::action::context ctx{
        request::action::dependencies{fixture.contract}};
    const request::event::configure configure{{}, k_route_tools_json};
    const request::event::reset reset{};
    const request::event::complete complete{query, 1u};
    request::event::configure_ctx configure_ctx{};
    request::event::reset_ctx reset_ctx{};
    request::event::complete_ctx complete_ctx{};
    const request::event::configure_run configure_run{configure,
                                                       configure_ctx};
    const request::event::reset_run reset_run{reset, reset_ctx};
    const request::event::complete_run complete_run{complete, complete_ctx};
    struct wrapped_runtime_event {
      const request::event::configure_run &event_;
    };

    const auto seed_outputs = [&ctx]() {
      ctx.prompt_size = 1u;
      ctx.prompt_id_count = 2u;
      ctx.generated_id_count = 3u;
      ctx.generated_text_size = 4u;
      ctx.normalized_envelope_size = 5u;
      ctx.prefill_nanoseconds = 6u;
      ctx.decode_nanoseconds = 7u;
    };
    const auto check_outputs_cleared = [&ctx]() {
      CHECK(ctx.prompt_size == 0u);
      CHECK(ctx.prompt_id_count == 0u);
      CHECK(ctx.generated_id_count == 0u);
      CHECK(ctx.generated_text_size == 0u);
      CHECK(ctx.normalized_envelope_size == 0u);
      CHECK(ctx.prefill_nanoseconds == 0u);
      CHECK(ctx.decode_nanoseconds == 0u);
    };

    emel::test::allocation::allocation_scope allocation_scope;
    seed_outputs();
    request::action::effect_on_unexpected{}(configure_run, ctx);
    CHECK(configure_ctx.err ==
          emel::error::cast(request::error::internal_error));
    check_outputs_cleared();
    seed_outputs();
    request::action::effect_on_unexpected{}(reset_run, ctx);
    CHECK(reset_ctx.err == emel::error::cast(request::error::internal_error));
    check_outputs_cleared();
    seed_outputs();
    request::action::effect_on_unexpected{}(complete_run, ctx);
    CHECK(complete_ctx.err ==
          emel::error::cast(request::error::internal_error));
    check_outputs_cleared();
    configure_ctx.err = emel::error::cast(request::error::none);
    seed_outputs();
    request::action::effect_on_unexpected{}(
        wrapped_runtime_event{configure_run}, ctx);
    CHECK(configure_ctx.err ==
          emel::error::cast(request::error::internal_error));
    check_outputs_cleared();
    CHECK(allocation_scope.allocations() == 0u);
  }

  SUBCASE("unwrapped runtime events fail from intermediate states") {
    request::action::context ctx{
        request::action::dependencies{fixture.contract}};
    stateforward::sml::sm<request::model, stateforward::sml::testing> machine{
        ctx};

    const auto check_runtime_failure = [&]<class state_type>(const auto &runtime,
                                                             auto &runtime_ctx) {
      machine.set_current_states(stateforward::sml::state<state_type>);
      runtime_ctx.err = emel::error::cast(request::error::none);
      ctx.prompt_size = 1u;
      ctx.prompt_id_count = 2u;
      ctx.generated_id_count = 3u;
      ctx.generated_text_size = 4u;
      ctx.normalized_envelope_size = 5u;
      ctx.prefill_nanoseconds = 6u;
      ctx.decode_nanoseconds = 7u;
      emel::test::allocation::allocation_scope allocation_scope;
      CHECK_FALSE(machine.process_event(runtime));
      CHECK(allocation_scope.allocations() == 0u);
      CHECK(runtime_ctx.err ==
            emel::error::cast(request::error::internal_error));
      CHECK(ctx.prompt_size == 0u);
      CHECK(ctx.prompt_id_count == 0u);
      CHECK(ctx.generated_id_count == 0u);
      CHECK(ctx.generated_text_size == 0u);
      CHECK(ctx.normalized_envelope_size == 0u);
      CHECK(ctx.prefill_nanoseconds == 0u);
      CHECK(ctx.decode_nanoseconds == 0u);
      CHECK(machine.is(stateforward::sml::state<request::state_errored>));
    };

    const request::event::configure configure{{}, k_route_tools_json};
    request::event::configure_ctx configure_ctx{};
    check_runtime_failure.template operator()<request::state_reset_decision>(
        request::event::configure_run{configure, configure_ctx}, configure_ctx);

    const request::event::reset reset{};
    request::event::reset_ctx reset_ctx{};
    check_runtime_failure.template operator()<request::state_configure_decision>(
        request::event::reset_run{reset, reset_ctx}, reset_ctx);

    const request::event::complete complete{query, 1u};
    request::event::complete_ctx complete_ctx{};
    check_runtime_failure.template operator()<request::state_reset_outcome>(
        request::event::complete_run{complete, complete_ctx}, complete_ctx);
  }

  SUBCASE("reset before configure reports error") {
    request::sm machine{fixture.contract};
    request_callback_state callbacks{};
    g_request_callback_state = &callbacks;
    check_failure(callbacks, machine.process_event(request::event::reset{
                                 &on_request_reset_done, &on_request_error}));
    g_request_callback_state = nullptr;
  }

  SUBCASE("complete before configure reports error") {
    request::sm machine{fixture.contract};
    request_callback_state callbacks{};
    g_request_callback_state = &callbacks;
    check_failure(callbacks, machine.process_event(request::event::complete{
                                 query, 1u, &on_request_completed,
                                 &on_request_error}));
    g_request_callback_state = nullptr;
  }

  SUBCASE("complete before reset reports error") {
    request::sm machine{fixture.contract};
    REQUIRE(machine.process_event(
        request::event::configure{{}, k_route_tools_json}));
    request_callback_state callbacks{};
    g_request_callback_state = &callbacks;
    check_failure(callbacks, machine.process_event(request::event::complete{
                                 query, 1u, &on_request_completed,
                                 &on_request_error}));
    g_request_callback_state = nullptr;
  }

  SUBCASE("illegal complete after success invalidates every prior response output") {
    request::sm machine{fixture.contract};
    const request_prompt_fixture row = first_request_prompts().front();
    constexpr std::string_view tools_end = "</tools>\n";
    constexpr std::string_view query_end =
        "<|im_end|>\n<|im_start|>assistant\n";
    const size_t tools_end_at = row.prompt.find(tools_end);
    REQUIRE(tools_end_at != std::string::npos);
    REQUIRE(row.prompt.ends_with(query_end));
    const size_t query_begin = tools_end_at + tools_end.size();
    const std::string_view successful_query{
        row.prompt.data() + query_begin,
        row.prompt.size() - query_begin - query_end.size()};

    REQUIRE(machine.process_event(
        request::event::configure{{}, k_route_tools_json}));
    REQUIRE(machine.process_event(request::event::reset{}));
    request_callback_state callbacks{};
    g_request_callback_state = &callbacks;
    REQUIRE(machine.process_event(request::event::complete{
        successful_query, 80u, &on_request_completed, &on_request_error}));
    REQUIRE(callbacks.done_count == 1u);
    REQUIRE(callbacks.error_count == 0u);
    REQUIRE_FALSE(machine.normalized_envelope().empty());
    REQUIRE_FALSE(machine.generated_token_ids().empty());
    REQUIRE(machine.prompt_tokens() > 0u);
    REQUIRE(machine.generated_tokens() > 0u);
    REQUIRE(machine.prefill_nanoseconds() > 0u);
    REQUIRE(machine.decode_nanoseconds() > 0u);

    callbacks = {};
    check_failure(callbacks, machine.process_event(request::event::complete{
                                 query, 1u, &on_request_completed,
                                 &on_request_error}));
    CHECK(machine.normalized_envelope().empty());
    CHECK(machine.generated_token_ids().empty());
    CHECK(machine.prompt_tokens() == 0u);
    CHECK(machine.generated_tokens() == 0u);
    CHECK(machine.prefill_nanoseconds() == 0u);
    CHECK(machine.decode_nanoseconds() == 0u);
    g_request_callback_state = nullptr;
  }

  SUBCASE("configure and reset dispatches invalidate prior response outputs") {
    request::sm machine{fixture.contract};
    const request_prompt_fixture row = first_request_prompts().front();
    constexpr std::string_view tools_end = "</tools>\n";
    constexpr std::string_view query_end =
        "<|im_end|>\n<|im_start|>assistant\n";
    const size_t tools_end_at = row.prompt.find(tools_end);
    REQUIRE(tools_end_at != std::string::npos);
    REQUIRE(row.prompt.ends_with(query_end));
    const size_t query_begin = tools_end_at + tools_end.size();
    const std::string_view successful_query{
        row.prompt.data() + query_begin,
        row.prompt.size() - query_begin - query_end.size()};
    REQUIRE(machine.process_event(
        request::event::configure{{}, k_route_tools_json}));
    request_callback_state callbacks{};
    g_request_callback_state = &callbacks;
    const auto complete_successfully = [&]() {
      REQUIRE(machine.process_event(request::event::reset{}));
      REQUIRE(machine.process_event(request::event::complete{
          successful_query, 80u, &on_request_completed, &on_request_error}));
      REQUIRE_FALSE(machine.normalized_envelope().empty());
      REQUIRE_FALSE(machine.generated_token_ids().empty());
      REQUIRE(machine.prompt_tokens() > 0u);
      REQUIRE(machine.generated_tokens() > 0u);
      REQUIRE(machine.prefill_nanoseconds() > 0u);
      REQUIRE(machine.decode_nanoseconds() > 0u);
    };
    const auto check_outputs_cleared = [&]() {
      CHECK(machine.normalized_envelope().empty());
      CHECK(machine.generated_token_ids().empty());
      CHECK(machine.prompt_tokens() == 0u);
      CHECK(machine.generated_tokens() == 0u);
      CHECK(machine.prefill_nanoseconds() == 0u);
      CHECK(machine.decode_nanoseconds() == 0u);
    };

    callbacks = {};
    complete_successfully();
    REQUIRE(machine.process_event(request::event::reset{}));
    check_outputs_cleared();

    complete_successfully();
    REQUIRE(machine.process_event(
        request::event::configure{{}, k_route_tools_json}));
    check_outputs_cleared();
    g_request_callback_state = nullptr;
  }

}

TEST_CASE("needle request source adapter preserves first-four rendered prompts and token ids") {
  auto fixture = load_contract_fixture();
  emel::model::needle::request::sm request{fixture.contract};
  const auto rows = first_request_prompts();
  const std::array<uint64_t, 4> expected_hashes = {
      0x27b46d6bd8eff67eULL, 0x381d6bf6b10d2fccULL,
      0xfce2b4b02fb723a3ULL, 0x4deefc5c4b77dc74ULL};

  for (size_t i = 0u; i < rows.size(); ++i) {
    const std::string_view prompt = rows[i].prompt;
    constexpr std::string_view tools_begin = "<tools>";
    constexpr std::string_view tools_end = "</tools>\n";
    constexpr std::string_view query_end =
        "<|im_end|>\n<|im_start|>assistant\n";
    const size_t tools_at = prompt.find(tools_begin);
    const size_t tools_end_at = prompt.find(tools_end, tools_at);
    REQUIRE(tools_at != std::string_view::npos);
    REQUIRE(tools_end_at != std::string_view::npos);
    REQUIRE(prompt.ends_with(query_end));
    const std::string_view tools = prompt.substr(
        tools_at + tools_begin.size(),
        tools_end_at - tools_at - tools_begin.size());
    const size_t query_begin = tools_end_at + tools_end.size();
    const std::string_view query = prompt.substr(
        query_begin, prompt.size() - query_begin - query_end.size());
    REQUIRE(request.process_event(
        emel::model::needle::request::event::configure{{}, tools}));
    emel::test::allocation::allocation_scope allocation_scope;
    REQUIRE(request.prepare(
        emel::model::needle::request::event::prepare{query, 80u}));
    CHECK(allocation_scope.allocations() == 0u);
    CHECK(request.rendered_prompt() == prompt);
    CHECK(request.prompt_token_ids().size() == rows[i].ids.size() + 1u);
    CHECK(request.prompt_token_ids().front() == 2);
    CHECK(hash_ids(request.prompt_token_ids().subspan(1u)) == expected_hashes[i]);
    CHECK(std::ranges::equal(request.prompt_token_ids().subspan(1u), rows[i].ids));
  }
}

TEST_CASE("needle request normalizes deterministic generated call envelopes") {
  auto fixture = load_contract_fixture();
  emel::model::needle::request::action::context ctx{
      emel::model::needle::request::action::dependencies{fixture.contract}};
  constexpr std::string_view generated =
      "<think>\nshort reason\n</think>\n<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\"}}]</tool_call>";
  REQUIRE(emel::model::needle::request::action::normalize_generated_response(
      ctx, generated));
  CHECK(std::string_view{ctx.normalized_envelope.data(),
                         ctx.normalized_envelope_size} ==
        "{\"error\":null,\"error_code\":null,\"function_calls\":[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\"}}],\"reason\":null,\"reasoning\":\"short reason\",\"success\":true,\"type\":\"call\",\"validation\":{\"negation\":false,\"ungrounded\":[]}}");
}

TEST_CASE("needle request rejects malformed tool-call JSON") {
  auto fixture = load_contract_fixture();
  emel::model::needle::request::action::context ctx{
      emel::model::needle::request::action::dependencies{fixture.contract}};
  constexpr std::array malformed = {
      "<think>reason</think><tool_call>[not-json]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\",}}]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\"},}]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\"}},]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\"}}]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"domain\":\"chat\",\"effort\":\"low\"}}]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\"}}]</tool_call>",
      "<tool_call>[{\"name\":\"route\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\",\"extra\":\"x\"}}]</tool_call>",
      "<tool_call>[{\"name\":\"ro\\ute\",\"arguments\":{\"domain\":\"other\",\"effort\":\"low\"}}]</tool_call>",
  };
  for (const std::string_view value : malformed) {
    CAPTURE(value);
    CHECK_FALSE(
        emel::model::needle::request::action::normalize_generated_response(
            ctx, value));
    CHECK(ctx.normalized_envelope_size == 0u);
  }
}

TEST_CASE("needle request rejects structurally invalid configured tools JSON") {
  constexpr std::array malformed = {
      R"json([{"name":"route","description":"decoy: properties domain effort enum required agentic-coding programming-qa math research writing extraction chat other low medium high xhigh","parameters":{"type":"object"}}])json",
      R"json([{"name":"route","description":"wrong path","parameters":{"type":"object","properties":{},"required":["domain","effort"]},"properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}}}])json",
      R"json([{"name":"route","name":"route","description":"duplicate","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}}])json",
      R"json([{"name":"route","description":"unknown nested key","parameters":{"type":"object","additionalProperties":false,"properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}}])json",
      R"json([{"description":"missing name","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}}])json",
      R"json([{"name":"route","description":"missing parameters"}])json",
      R"json([{"name":"route","description":"duplicate required","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"],"required":["domain","effort"]}}])json",
      R"json([{"name":"route","description":"duplicate enum key","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"],"enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}}])json",
      R"json([{"name":"route","description":"two tools","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}},{"name":"route","description":"second","parameters":{"type":"object","properties":{"domain":{"type":"string","enum":["agentic-coding","programming-qa","math","research","writing","extraction","chat","other"]},"effort":{"type":"string","enum":["low","medium","high","xhigh"]}},"required":["domain","effort"]}}])json",
  };
  for (const std::string_view value : malformed) {
    CAPTURE(value);
    CHECK_FALSE(
        emel::model::needle::request::action::validate_tools_json(value));
  }

  const std::array malformed_variants = {
      replace_once(std::string{k_route_tools_json},
                   R"json(,"required":["domain","effort"])json", ""),
      replace_once(std::string{k_route_tools_json}, R"json("other")json",
                   R"json("wrong")json"),
      replace_once(std::string{k_route_tools_json}, R"json("xhigh")json",
                   R"json("high")json"),
      replace_once(std::string{k_route_tools_json}, R"json("type":"object")json",
                   R"json("type":"array")json"),
      replace_once(std::string{k_route_tools_json},
                   R"json("required":["domain","effort"])json",
                   R"json("required":["domain","domain"])json"),
      with_route_ancillary(R"json(,"metadata":01)json"),
      with_route_ancillary(R"json(,"metadata":1.)json"),
      with_route_ancillary(R"json(,"metadata":1e)json"),
      with_route_ancillary(R"json(,"metadata":{"items":[],})json"),
      replace_once(std::string{k_route_tools_json}, "]", ",]"),
      std::string{k_route_tools_json} + "garbage",
      std::string{"{}"},
  };
  for (const std::string &value : malformed_variants) {
    CAPTURE(value);
    CHECK_FALSE(
        emel::model::needle::request::action::validate_tools_json(value));
  }
}

TEST_CASE("needle request accepts valid configured tool JSON grammar") {
  const std::string escaped_description = replace_once(
      std::string{k_route_tools_json},
      "Route a request to a domain queue with the minimum sufficient reasoning effort",
      R"json(Route \"quoted\" \\ slash \u0052oute\nnext)json");
  const std::string ancillary = with_route_ancillary(
      R"json(,"metadata":{"items":[],"object":{},"number":-1.25e+2,"enabled":true,"none":null},"examples":[])json");

  emel::test::allocation::allocation_scope allocation_scope;
  CHECK(emel::model::needle::request::action::validate_tools_json(
      k_route_tools_json));
  CHECK(emel::model::needle::request::action::validate_tools_json(
      escaped_description));
  CHECK(emel::model::needle::request::action::validate_tools_json(ancillary));
  CHECK(allocation_scope.allocations() == 0u);
}

TEST_CASE("needle request null timestamp dependency is safe") {
  auto fixture = load_contract_fixture();
  emel::model::needle::request::sm request{fixture.contract, nullptr};
  constexpr std::string_view tools = k_route_tools_json;
  constexpr std::string_view query = "hello";
  REQUIRE(request.process_event(
      emel::model::needle::request::event::configure{{}, tools}));
  REQUIRE(request.process_event(emel::model::needle::request::event::reset{}));
  CHECK_FALSE(request.process_event(
      emel::model::needle::request::event::complete{query, 1u}));
}

TEST_CASE("needle request invalid reconfigure clears prior reset state") {
  auto fixture = load_contract_fixture();
  emel::model::needle::request::sm request{fixture.contract};
  constexpr std::string_view tools = k_route_tools_json;
  REQUIRE(request.process_event(
      emel::model::needle::request::event::configure{{}, tools}));
  REQUIRE(request.process_event(emel::model::needle::request::event::reset{}));
  CHECK_FALSE(request.process_event(
      emel::model::needle::request::event::configure{{}, "{}"}));
  CHECK_FALSE(request.process_event(emel::model::needle::request::event::reset{}));
}

} // namespace

uint64_t g_timing_clock = 0u;

uint64_t fake_timestamp_now() noexcept {
  g_timing_clock += 10u;
  return g_timing_clock;
}

emel::model::needle::graph::sm *g_reentrant_graph = nullptr;
std::atomic<bool> g_reentrant_attempted = false;
std::atomic<bool> g_reentrant_accepted = false;

uint64_t reentrant_timestamp_now() noexcept {
  if (!g_reentrant_attempted.exchange(true, std::memory_order_acq_rel)) {
    emel::model::needle::graph::event::timing_breakdown nested{};
    g_reentrant_accepted.store(
        g_reentrant_graph->process_event(
            emel::model::needle::graph::event::capture_timing{nested}),
        std::memory_order_release);
  }
  return fake_timestamp_now();
}

std::atomic<bool> g_blocking_clock_entered = false;
std::atomic<bool> g_release_blocking_clock = false;

uint64_t blocking_timestamp_now() noexcept {
  if (!g_blocking_clock_entered.exchange(true, std::memory_order_acq_rel))
    while (!g_release_blocking_clock.load(std::memory_order_acquire)) {
    }
  return fake_timestamp_now();
}
TEST_CASE("needle graph default f32 route matches the committed JAX logits "
          "fixture on all cases") {
  // Load the pinned .cact through the maintained loader chain.
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);

  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));

  // Bind the named contract.
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));
  REQUIRE(contract.geo.vocab_size == k_vocab);

  // The graph machine allocates all runtime storage at construction.
  emel::model::needle::graph::sm graph{contract};
  std::vector<float> logits(k_vocab);
  double worst_abs = 0.0;
  double worst_rel = 0.0;

  for (const auto &parity : k_cases) {
    // Default init is the authoritative heldout CQ4/f32 generation route.
    REQUIRE(graph.process_event(emel::model::needle::graph::event::init{}));

    const auto reference = read_reference_logits(parity.file);
    std::array<int32_t, k_steps> greedy = {};

    REQUIRE(graph.process_event(emel::model::needle::graph::event::prefill{
        std::span<const int32_t>{parity.prompt_ids},
        std::span<float>{logits}}));
    for (uint32_t step = 0u; step < k_steps; ++step) {
      const std::span<const float> reference_step{
          reference.data() + static_cast<size_t>(step) * k_vocab, k_vocab};
      const step_error err = compare_step(logits, reference_step);
      worst_abs = err.max_abs > worst_abs ? err.max_abs : worst_abs;
      worst_rel = err.rel > worst_rel ? err.rel : worst_rel;
      greedy[step] = static_cast<int32_t>(
          argmax(std::span<const float>{logits.data(), k_vocab}));
      MESSAGE("case ", std::string_view{parity.file}, " step ", step,
              ": max_abs=", err.max_abs, " rel=", err.rel,
              " argmax=", greedy[step]);
      // Target tolerance: rel <= 1e-3 against the max reference magnitude.
      CHECK(err.rel <= 1e-3);
      CHECK(greedy[step] == parity.greedy[step]);
      if (step + 1u < k_steps) {
        REQUIRE(graph.process_event(emel::model::needle::graph::event::decode{
            greedy[step], std::span<float>{logits}}));
      }
    }
  }
  MESSAGE("worst-case parity across 3 cases x 3 steps: max_abs=", worst_abs,
          " rel=", worst_rel);
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  uint64_t gqa2_calls = 0u;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_swa_diagnostics{gqa2_calls}));
  CHECK(gqa2_calls > 0u);
#endif
}
TEST_CASE("needle graph explicit A8 training route matches authoritative JAX "
          "fixture") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  std::vector<float> logits(k_vocab);
  double worst_abs = 0.0;
  double worst_rel = 0.0;
  for (const auto &parity : k_a8_cases) {
    REQUIRE(graph.process_event(emel::model::needle::graph::event::init{true}));
    const auto reference = read_reference_logits(parity.file);
    std::array<int32_t, k_steps> greedy = {};
    REQUIRE(graph.process_event(emel::model::needle::graph::event::prefill{
        std::span<const int32_t>{parity.prompt_ids},
        std::span<float>{logits}}));
    for (uint32_t step = 0u; step < k_steps; ++step) {
      const std::span<const float> reference_step{
          reference.data() + static_cast<size_t>(step) * k_vocab, k_vocab};
      const step_error err = compare_step(logits, reference_step);
      worst_abs = err.max_abs > worst_abs ? err.max_abs : worst_abs;
      worst_rel = err.rel > worst_rel ? err.rel : worst_rel;
      greedy[step] = static_cast<int32_t>(
          argmax(std::span<const float>{logits.data(), k_vocab}));
      MESSAGE("A8 case ", std::string_view{parity.file}, " step ", step,
              ": max_abs=", err.max_abs, " rel=", err.rel,
              " argmax=", greedy[step]);
      // The native graph keeps exact CQ operands and greedy identity. Its
      // scalar stage ordering differs from XLA, so use the generated fixture's
      // observed 1.5e-2 envelope rather than the f32 route's 1e-3.
      CHECK(err.rel <= 1.5e-2);
      CHECK(greedy[step] == parity.greedy[step]);
      if (step + 1u < k_steps) {
        REQUIRE(graph.process_event(emel::model::needle::graph::event::decode{
            greedy[step], std::span<float>{logits}}));
      }
    }
  }
  MESSAGE("worst-case W4A8 parity across 3 cases x 3 steps: max_abs=",
          worst_abs, " rel=", worst_rel);

  uint64_t quantize_calls = 0u;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_a8_diagnostics{
          quantize_calls}));
  CHECK(quantize_calls > 0u);
}

TEST_CASE("needle graph prepares CQ4 storage once and selects prepared route") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  uint64_t prepare_calls = 0u;
  uint64_t prepared_calls = 0u;
  size_t prepared_index_bytes = 0u;
  size_t prepared_input32_bytes = 0u;
  size_t prepared_norm_bytes = 0u;
  size_t prepared_group32_norm_bytes = 0u;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepare_calls > 0u);
  CHECK(prepared_calls == 0u);
  CHECK(prepared_index_bytes > 0u);
  CHECK(prepared_input32_bytes == prepared_index_bytes);
  CHECK(prepared_norm_bytes > 0u);
  CHECK(prepared_group32_norm_bytes > 0u);

  std::vector<float> logits(k_vocab);
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  const uint64_t prepared_after_decode = prepared_calls;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepared_calls > prepared_after_decode);
  const uint64_t preparation_count = prepare_calls;

  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepare_calls == preparation_count);

  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepare_calls == preparation_count);
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
}

TEST_CASE(
    "needle graph component timing is explicit, resettable, and reconciled") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  std::vector<float> logits(k_vocab);
  REQUIRE(graph.process_event(emel::model::needle::graph::event::init{}));
  g_timing_clock = 0u;
  REQUIRE(
      graph.process_event(emel::model::needle::graph::event::configure_timing{
          true, &fake_timestamp_now}));
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  emel::model::needle::graph::event::timing_breakdown timing{};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_timing{timing}));
  CHECK(timing.steps == 1u);
  CHECK(timing.total_nanoseconds > 0u);
  const uint64_t split =
      timing.cq_nanoseconds + timing.graph_overhead_nanoseconds +
      timing.engram_nanoseconds + timing.norm_nanoseconds +
      timing.mhc_pre_nanoseconds + timing.mhc_post_nanoseconds +
      timing.attention_rope_nanoseconds + timing.attention_cache_nanoseconds +
      timing.attention_attend_nanoseconds + timing.attention_gate_nanoseconds +
      timing.hadamard_nanoseconds + timing.lane_copy_mean_nanoseconds +
      timing.sampling_nanoseconds;
  CHECK(split == timing.total_nanoseconds);
  REQUIRE(
      graph.process_event(emel::model::needle::graph::event::reset_timing{}));
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_timing{timing}));
  CHECK(timing.steps == 0u);
  CHECK(timing.total_nanoseconds == 0u);
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::configure_timing{false, nullptr}));
}

TEST_CASE("needle graph AVX2 route requires every CQ tensor group to be 128") {
  emel::model::needle::contract contract{};
  contract.layer_count = 1u;
  contract.engram_site_count = 1u;
  contract.geo.d_model = 512u;
  contract.geo.hada_n = 512u;
  contract.embedding.group = 128u;
  contract.mhc.phi_pre.group = 128u;
  contract.mhc.phi_post.group = 128u;
  contract.mhc.phi_res.group = 128u;
  auto &layer = contract.layers[0];
  layer.q_proj.group = 128u;
  layer.k_proj.group = 128u;
  layer.v_proj.group = 128u;
  layer.gate_proj.group = 128u;
  layer.out_proj.group = 128u;
  auto &site = contract.engram_sites[0];
  site.tables.group = 128u;
  site.key_proj.group = 128u;
  site.value_proj.group = 128u;

  emel::model::needle::graph::action::context ctx{contract};
  ctx.storage_valid = true;
  emel::model::needle::graph::event::step_ctx step{};
  const emel::model::needle::graph::event::step_run run{step};
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__) && defined(__F16C__)
  ctx.avx2_fma_available = true;
  CHECK(emel::model::needle::graph::guard::guard_route_avx2{}(run, ctx));
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));
#else
  ctx.avx2_fma_available = false;
  CHECK_FALSE(emel::model::needle::graph::guard::guard_route_avx2{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));
#endif

  contract.layers[0].out_proj.group = 64u;
  emel::model::needle::graph::action::context group_ctx{contract};
  group_ctx.storage_valid = true;
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_route_avx2{}(run, group_ctx));
  CHECK(
      emel::model::needle::graph::guard::guard_route_scalar{}(run, group_ctx));

  contract.layers[0].out_proj.group = 128u;
  contract.geo.d_model = 256u;
  emel::model::needle::graph::action::context geometry_ctx{contract};
  geometry_ctx.storage_valid = true;
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_route_avx2{}(run, geometry_ctx));
  CHECK(emel::model::needle::graph::guard::guard_route_scalar{}(run,
                                                                geometry_ctx));
}

TEST_CASE("needle graph rejects an out-of-vocab step token") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);

  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  std::vector<float> logits(k_vocab);
  CHECK_FALSE(graph.process_event(emel::model::needle::graph::event::decode{
      static_cast<int32_t>(k_vocab), std::span<float>{logits}}));
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_errored>));

  // Re-init recovers the machine.
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_ready>));
}

TEST_CASE(
    "needle graph serial and parallel4 routes are exact deterministic peers") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__) && defined(__F16C__)
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));
  emel::model::needle::graph::serial_sm serial{contract};
  emel::model::needle::graph::parallel4_sm parallel4{contract};
  for (const bool activation_quant : {false, true}) {
    REQUIRE(serial.process_event(
        emel::model::needle::graph::event::init{activation_quant}));
    REQUIRE(parallel4.process_event(
        emel::model::needle::graph::event::init{activation_quant}));
    std::vector<float> serial_logits(k_vocab);
    std::vector<float> parallel4_logits(k_vocab);
    const std::array<int32_t, 4u> prompt = {2, 1544, 1663, 2328};
    REQUIRE(serial.process_event(
        emel::model::needle::graph::event::prefill{prompt, serial_logits}));
    REQUIRE(parallel4.process_event(
        emel::model::needle::graph::event::prefill{prompt, parallel4_logits}));
    CHECK(parallel4_logits == serial_logits);
    const int32_t next = static_cast<int32_t>(argmax(serial_logits));
    REQUIRE(serial.process_event(
        emel::model::needle::graph::event::decode{next, serial_logits}));
    REQUIRE(parallel4.process_event(
        emel::model::needle::graph::event::decode{next, parallel4_logits}));
    CHECK(parallel4_logits == serial_logits);
    std::array<uint64_t, 3u> calls{};
    uint64_t submitted = 0u;
    uint64_t joined = 0u;
    uint64_t live = 1u;
    REQUIRE(parallel4.process_event(
        emel::model::needle::graph::event::capture_projection_diagnostics{
            calls, submitted, joined, live}));
    CHECK(calls[0] > 0u);
    CHECK(calls[1] > 0u);
    CHECK(calls[2] > 0u);
    CHECK(submitted == joined);
    CHECK(live == 0u);
  }
#endif
}

TEST_CASE("needle graph rejects work before init without writing outputs") {
  auto fixture = load_contract_fixture();
  emel::model::needle::graph::sm graph{fixture.contract};
  std::vector<float> logits(k_vocab, 17.0f);
  const std::array<int32_t, 1u> prompt = {2};

  CHECK_FALSE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  check_all_equal(logits, 17.0f);
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_errored>));

  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  std::fill(logits.begin(), logits.end(), 23.0f);
  CHECK_FALSE(graph.process_event(emel::model::needle::graph::event::prefill{
      std::span<const int32_t>{}, std::span<float>{logits}}));
  check_all_equal(logits, 23.0f);
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_ready>));

  REQUIRE(graph.process_event(emel::model::needle::graph::event::prefill{
      prompt, std::span<float>{logits}}));
}

TEST_CASE(
    "needle graph invalid requests are fail-closed and reinit clears them") {
  auto fixture = load_contract_fixture();

  SUBCASE("negative token preserves caller logits") {
    emel::model::needle::graph::sm graph{fixture.contract};
    REQUIRE(graph.process_event(
        emel::model::needle::graph::event::init{.activation_quant = false}));
    std::vector<float> logits(k_vocab, 31.0f);
    CHECK_FALSE(graph.process_event(emel::model::needle::graph::event::decode{
        -1, std::span<float>{logits}}));
    check_all_equal(logits, 31.0f);
    CHECK(graph.is(
        stateforward::sml::state<emel::model::needle::graph::state_errored>));
    REQUIRE(graph.process_event(
        emel::model::needle::graph::event::init{.activation_quant = false}));
    CHECK(graph.is(
        stateforward::sml::state<emel::model::needle::graph::state_ready>));
  }

  SUBCASE("undersized logits preserve caller storage") {
    emel::model::needle::graph::sm graph{fixture.contract};
    REQUIRE(graph.process_event(
        emel::model::needle::graph::event::init{.activation_quant = false}));
    std::vector<float> logits(k_vocab - 1u, 37.0f);
    CHECK_FALSE(graph.process_event(emel::model::needle::graph::event::decode{
        2, std::span<float>{logits}}));
    check_all_equal(logits, 37.0f);
    CHECK(graph.is(
        stateforward::sml::state<emel::model::needle::graph::state_errored>));
  }
}

TEST_CASE("needle graph enforces sequence capacity before any output write") {
  auto fixture = load_contract_fixture();
  fixture.contract.geo.max_seq_len = 2u;
  emel::model::needle::graph::sm graph{fixture.contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  std::vector<float> logits(k_vocab);
  const std::array<int32_t, 2u> prompt = {2, 1544};
  REQUIRE(graph.process_event(emel::model::needle::graph::event::prefill{
      prompt, std::span<float>{logits}}));

  std::fill(logits.begin(), logits.end(), 41.0f);
  CHECK_FALSE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  check_all_equal(logits, 41.0f);
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_errored>));
}

TEST_CASE("needle graph rejects unsupported construction geometry") {
  auto fixture = load_contract_fixture();
  fixture.contract.geo.hada_n = 384u;
  const auto result = emel::model::needle::graph::sm::create(fixture.contract);
  CHECK(result.machine == nullptr);
  CHECK(result.err ==
        emel::error::cast(
            emel::model::needle::graph::error::geometry_unsupported));
}

TEST_CASE("needle graph scalar route covers guards and activation phases") {
  auto fixture = load_contract_fixture();
  fixture.contract.geo.d_model = 513u;
  emel::model::needle::graph::action::context ctx{fixture.contract};
  emel::model::needle::graph::event::step_ctx step{};
  step.token = 2;
  step.want_logits = true;
  std::vector<float> logits(k_vocab);
  step.logits_out = logits;
  const emel::model::needle::graph::event::step_run run{step};

  CHECK(emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_step_valid_scalar{}(run, ctx));
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_step_valid_avx2{}(run, ctx));

  step.activation_quant = true;
  CHECK(emel::model::needle::graph::guard::guard_deployment_a8{}(run, ctx));
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_deployment_f32{}(run, ctx));
  step.activation_quant = false;
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_deployment_a8{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_deployment_f32{}(run, ctx));
}

TEST_CASE("needle graph construction factory returns typed failure") {
  auto fixture = load_contract_fixture();
  using graph_type = emel::model::needle::graph::sm;
  const graph_type::construction_factory fail_construction =
      [](const emel::model::needle::contract &) noexcept
      -> graph_type::construction_result {
    return {.machine = {},
            .err = emel::error::cast(
                emel::model::needle::graph::error::internal_error)};
  };

  const auto result = graph_type::create(fixture.contract, fail_construction);
  CHECK(result.machine == nullptr);
  CHECK(result.err ==
        emel::error::cast(emel::model::needle::graph::error::internal_error));
}

TEST_CASE("needle graph construction factory rejects inconsistent result") {
  auto fixture = load_contract_fixture();
  using graph_type = emel::model::needle::graph::sm;
  const graph_type::construction_factory inconsistent_construction =
      [](const emel::model::needle::contract &) noexcept
      -> graph_type::construction_result { return {}; };

  const auto result =
      graph_type::create(fixture.contract, inconsistent_construction);
  CHECK(result.machine == nullptr);
  CHECK(result.err ==
        emel::error::cast(emel::model::needle::graph::error::internal_error));
}

TEST_CASE(
    "needle graph construction factory preserves explicit capacity failure") {
  auto fixture = load_contract_fixture();
  using graph_type = emel::model::needle::graph::sm;
  const graph_type::construction_factory fail_construction =
      [](const emel::model::needle::contract &) noexcept
      -> graph_type::construction_result {
    return {.machine = {},
            .err = emel::error::cast(
                emel::model::needle::graph::error::capacity_exceeded)};
  };

  const auto result = graph_type::create(fixture.contract, fail_construction);
  CHECK(result.machine == nullptr);
  CHECK(
      result.err ==
      emel::error::cast(emel::model::needle::graph::error::capacity_exceeded));
}

TEST_CASE("needle graph construction factory preserves normal creation") {
  auto fixture = load_contract_fixture();

  const auto result = emel::model::needle::graph::sm::create(fixture.contract);
  REQUIRE(result.machine != nullptr);
  CHECK(result.err ==
        emel::error::cast(emel::model::needle::graph::error::none));
}

TEST_CASE("needle graph bounded construction rejects extreme geometry typed") {
  auto fixture = load_contract_fixture();
  fixture.contract.geo.max_seq_len = UINT32_MAX;

  const auto result = emel::model::needle::graph::sm::create(fixture.contract);
  CHECK(result.machine == nullptr);
  CHECK(result.err ==
        emel::error::cast(
            emel::model::needle::graph::error::geometry_unsupported));
}

TEST_CASE(
    "needle graph bounded construction rejects practical allocation cap") {
  auto fixture = load_contract_fixture();
  fixture.contract.geo.kv_window = 65536u;

  const auto result = emel::model::needle::graph::sm::create(fixture.contract);
  CHECK(result.machine == nullptr);
  CHECK(result.err ==
        emel::error::cast(
            emel::model::needle::graph::error::geometry_unsupported));
}

TEST_CASE("needle graph GQA2 route falls back when runtime AVX2 is disabled") {
  auto fixture = load_contract_fixture();
  emel::model::needle::graph::action::context ctx{fixture.contract};
  ctx.avx2_fma_available = false;
  emel::model::needle::graph::event::step_ctx step{};
  const emel::model::needle::graph::event::step_run run{step};

  CHECK_FALSE(emel::model::needle::graph::guard::guard_attend_gqa2{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_attend_generic{}(run, ctx));
}

TEST_CASE("needle graph engram dilation gathers exact dilated tap positions") {
  std::array<uint32_t, 4u> gathered{};
  for (uint32_t tap = 0u; tap < gathered.size(); ++tap) {
    gathered[tap] =
        emel::model::needle::graph::action::compute_engram_tap_position(
            11u, tap, 3u);
  }
  CHECK(gathered == std::array<uint32_t, 4u>{11u, 8u, 5u, 2u});
}

TEST_CASE(
    "needle graph pinned engram sites retain exact layer-to-site mapping") {
  const auto fixture = load_contract_fixture();
  const auto &geo = fixture.contract.geo;
  REQUIRE(geo.num_engram_sites == fixture.contract.engram_site_count);
  REQUIRE(geo.num_engram_sites == 2u);
  CHECK(geo.engram_sites[0] == 2u);
  CHECK(geo.engram_sites[1] == 15u);

  uint32_t expected_max_order = 0u;
  for (uint32_t i = 0u; i < geo.num_engram_orders; ++i) {
    expected_max_order = geo.engram_orders[i] > expected_max_order
                             ? geo.engram_orders[i]
                             : expected_max_order;
  }
  const uint32_t expected_history_extent =
      (geo.engram_conv_taps - 1u) * geo.engram_conv_dilation;
  uint32_t history_extent = 0u;
  uint32_t max_order = 0u;
  uint32_t window = 0u;
  uint32_t positions = 0u;
  REQUIRE(emel::model::needle::detail::compute_engram_hash_geometry(
      geo, history_extent, max_order, window, positions));
  CHECK(history_extent == expected_history_extent);
  CHECK(max_order == expected_max_order);
  CHECK(window == expected_history_extent + expected_max_order - 1u);
  CHECK(positions == window + 1u);

  for (uint32_t expected = 0u; expected < geo.num_engram_sites; ++expected) {
    uint32_t actual = emel::model::needle::k_max_engram_sites;
    CHECK(emel::model::needle::detail::find_engram_site_index(
        geo, geo.engram_sites[expected], actual));
    CHECK(actual == expected);
  }

  for (uint32_t layer = 0u; layer < geo.num_layers; ++layer) {
    uint32_t site = emel::model::needle::k_max_engram_sites;
    const bool found =
        emel::model::needle::detail::find_engram_site_index(geo, layer, site);
    CHECK(found ==
          emel::model::needle::graph::guard::layer_is_engram_site(geo, layer));
    if (!found) {
      CHECK(site == emel::model::needle::k_max_engram_sites);
    }
  }
}

TEST_CASE(
    "needle graph diagnostics reject invalid clocks and stay observational") {
  auto fixture = load_contract_fixture();
  emel::model::needle::graph::sm graph{fixture.contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  CHECK_FALSE(graph.process_event(
      emel::model::needle::graph::event::configure_timing{true, nullptr}));
  CHECK_FALSE(graph.process_event(
      emel::model::needle::graph::event::configure_cq_timing{true, nullptr}));

  emel::model::needle::graph::event::timing_breakdown graph_timing{};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_timing{graph_timing}));
  CHECK(graph_timing.steps == 0u);
  emel::kernel::cq::event::timing_breakdown cq_timing{};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_timing{cq_timing}));
  CHECK(cq_timing.quantize_nanoseconds == 0u);
  CHECK(cq_timing.dot_full_nanoseconds == 0u);
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_ready>));

  REQUIRE(
      graph.process_event(emel::model::needle::graph::event::reset_timing{}));
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_ready>));
}

TEST_CASE(
    "needle graph owns contract metadata while borrowing tensor payloads") {
  auto fixture = load_contract_fixture();
  const uint8_t *const borrowed_embedding = fixture.contract.embedding.data;
  auto graph =
      std::make_unique<emel::model::needle::graph::sm>(fixture.contract);
  fixture.contract = {};

  REQUIRE(graph->process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  std::vector<float> logits(k_vocab);
  REQUIRE(graph->process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  CHECK(borrowed_embedding != nullptr);
}

TEST_CASE("needle graph construction rejects forged tensor payload views") {
  auto fixture = load_contract_fixture();
  using graph_type = emel::model::needle::graph::sm;

  SUBCASE("null CQ backing") {
    fixture.contract.layers[0].q_proj.data = nullptr;
    const auto result = graph_type::create(fixture.contract);
    CHECK(result.machine == nullptr);
  }
  SUBCASE("short CQ backing") {
    fixture.contract.embedding.nbytes -= 1u;
    const auto result = graph_type::create(fixture.contract);
    CHECK(result.machine == nullptr);
  }
  SUBCASE("short fp16 backing") {
    fixture.contract.layers[0].norm_in.nbytes = 1u;
    const auto result = graph_type::create(fixture.contract);
    CHECK(result.machine == nullptr);
  }
  SUBCASE("misaligned fp16 backing") {
    fixture.contract.final_norm.data += 1u;
    const auto result = graph_type::create(fixture.contract);
    CHECK(result.machine == nullptr);
  }
  SUBCASE("wrong CQ bit width") {
    fixture.contract.mhc.phi_pre.bits = 3u;
    const auto result = graph_type::create(fixture.contract);
    CHECK(result.machine == nullptr);
  }
}

TEST_CASE("needle graph rejects callback-reentrant decode and diagnostics") {
  auto fixture = load_contract_fixture();
  emel::model::needle::graph::sm graph{fixture.contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  g_reentrant_graph = &graph;
  g_reentrant_attempted.store(false, std::memory_order_release);
  g_reentrant_accepted.store(true, std::memory_order_release);
  REQUIRE(
      graph.process_event(emel::model::needle::graph::event::configure_timing{
          true, &reentrant_timestamp_now}));

  std::vector<float> logits(k_vocab);
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  CHECK(g_reentrant_attempted.load(std::memory_order_acquire));
  CHECK_FALSE(g_reentrant_accepted.load(std::memory_order_acquire));
  g_reentrant_graph = nullptr;
}

TEST_CASE("needle graph decode rejects concurrent decode and diagnostics") {
  auto fixture = load_contract_fixture();
  emel::model::needle::graph::sm graph{fixture.contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  g_blocking_clock_entered.store(false, std::memory_order_release);
  g_release_blocking_clock.store(false, std::memory_order_release);
  REQUIRE(
      graph.process_event(emel::model::needle::graph::event::configure_timing{
          true, &blocking_timestamp_now}));

  std::vector<float> first(k_vocab);
  std::atomic<bool> decode_accepted = false;
  std::thread decode([&]() {
    decode_accepted.store(
        graph.process_event(emel::model::needle::graph::event::decode{
            2, std::span<float>{first}}),
        std::memory_order_release);
  });
  while (!g_blocking_clock_entered.load(std::memory_order_acquire)) {
  }

  std::vector<float> rejected(k_vocab, 73.0f);
  CHECK_FALSE(graph.process_event(emel::model::needle::graph::event::decode{
      2, std::span<float>{rejected}}));
  check_all_equal(rejected, 73.0f);
  emel::model::needle::graph::event::timing_breakdown diagnostics{};
  CHECK_FALSE(graph.process_event(
      emel::model::needle::graph::event::capture_timing{diagnostics}));
  CHECK(diagnostics.steps == 0u);

  g_release_blocking_clock.store(true, std::memory_order_release);
  decode.join();
  CHECK(decode_accepted.load(std::memory_order_acquire));
}
