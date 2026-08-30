// Heldout accuracy eval for the native needle runtime.
//
// Drives the maintained chain end to end: cact loader (probe/bind/parse) ->
// needle binder -> needle tokenizer blob loader -> shared SPM tokenizer ->
// native graph (public init/prefill/decode events only). For every heldout
// row it consumes the prompt text plus reference tokenizer IDs produced by
// scripts/gen_needle_heldout_eval.py, asserts native parity, and feeds the
// native IDs to the graph.
//
// Prediction protocol matches the JAX reference eval that produced the
// 0.840 domain / 0.760 effort numbers (/shared/effortless/scripts/eval_jax.py):
// greedy decode (temperature 0) up to 80 new tokens, stop at EOS (id 1),
// detokenize, parse the first <tool_call>{...}</tool_call> arguments object.
// The probe heads bound in the phase-2 contract (head manifest codes
// 1 contrastive / 2 confidence) are NOT part of this accuracy path in the
// maintained reference: the contrastive head embeds tool schemas and the
// confidence head is uncalibrated for tuned weights (train/REPORT.md), so
// domain/effort accuracy is a text-generation metric.
//
// Usage:
//   needle_eval <model.cact> <prompts.tsv> [row_begin row_end]
// Output: one "row i=..." line per example plus a final summary line.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/cact/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/needle/graph/sm.hpp"
#include "emel/model/needle/sm.hpp"
#include "emel/text/tokenizer/needle/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

namespace cact_loader = emel::cact::loader;
namespace needle = emel::model::needle;

constexpr int32_t k_bos_id = 2;
constexpr int32_t k_eos_id = 1;
constexpr uint32_t k_max_new_tokens = 80u;

[[noreturn]] void die(const char *what) {
  std::fprintf(stderr, "error: needle_eval: %s\n", what);
  std::exit(1);
}

void on_probe_done(const cact_loader::events::probe_done &) {}
void on_probe_error(const cact_loader::events::probe_error &) {}
void on_bind_done(const cact_loader::events::bind_done &) {}
void on_bind_error(const cact_loader::events::bind_error &) {}
void on_parse_done(const cact_loader::events::parse_done &) {}
void on_parse_error(const cact_loader::events::parse_error &) {}
void on_needle_done(const needle::events::bind_done &) {}
void on_needle_error(const needle::events::bind_error &) {}
void on_tok_load_done(const emel::text::tokenizer::needle::events::load_done &) {}
void on_tok_load_error(const emel::text::tokenizer::needle::events::load_error &) {}

std::vector<uint8_t> read_file_bytes(const char *path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) die("open model");
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  if (size <= 0) die("model size");
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!input.good()) die("read model");
  return bytes;
}

uint32_t argmax(const std::span<const float> logits) {
  uint32_t best = 0u;
  for (uint32_t i = 1u; i < logits.size(); ++i)
    best = logits[i] > logits[best] ? i : best;
  return best;
}

struct eval_row {
  std::string gold_domain;
  std::string gold_effort;
  std::vector<int32_t> ref_ids;
  std::string prompt;
};

std::vector<eval_row> read_rows(const char *path) {
  std::ifstream input(path);
  if (!input.good()) die("open prompts tsv");
  std::vector<eval_row> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    eval_row row;
    size_t a = line.find('\t');
    size_t b = line.find('\t', a + 1);
    size_t c = line.find('\t', b + 1);
    if (a == std::string::npos || b == std::string::npos ||
        c == std::string::npos)
      die("malformed tsv row");
    row.gold_domain = line.substr(0, a);
    row.gold_effort = line.substr(a + 1, b - a - 1);
    const std::string ids_text = line.substr(b + 1, c - b - 1);
    size_t cursor = 0;
    while (cursor < ids_text.size()) {
      char *end = nullptr;
      const long value = std::strtol(ids_text.c_str() + cursor, &end, 10);
      if (end == ids_text.c_str() + cursor) break;
      row.ref_ids.push_back(static_cast<int32_t>(value));
      cursor = static_cast<size_t>(end - ids_text.c_str());
      while (cursor < ids_text.size() && ids_text[cursor] == ' ') ++cursor;
    }
    const std::string hex = line.substr(c + 1);
    if (hex.size() % 2u != 0u) die("odd prompt hex");
    row.prompt.reserve(hex.size() / 2u);
    for (size_t i = 0; i < hex.size(); i += 2u) {
      row.prompt.push_back(static_cast<char>(
          std::stoi(hex.substr(i, 2u), nullptr, 16)));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

std::string_view vocab_piece(const emel::model::data::vocab &vocab,
                             const uint32_t id) {
  return std::string_view{vocab.token_storage.data() +
                              vocab.entries[id].text_offset,
                          vocab.entries[id].text_length};
}

// Detokenize per export.py RefTokenizer.decode: byte pieces append the raw
// byte, CONTROL/UNKNOWN skipped, everything else appends the surface; the
// SentencePiece meta-space becomes ' '. Piece types use the shared vocab
// (GGUF) codes: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 6 BYTE.
std::string detokenize(const emel::model::data::vocab &vocab,
                       const std::span<const int32_t> ids) {
  std::string out;
  for (const int32_t id : ids) {
    if (id < 0 || static_cast<uint32_t>(id) >= vocab.n_tokens) continue;
    const int32_t type = vocab.entries[static_cast<uint32_t>(id)].type;
    if (type == 2 || type == 3) continue;
    const std::string_view piece =
        vocab_piece(vocab, static_cast<uint32_t>(id));
    if (type == 6 && piece.size() == 6u && piece.rfind("<0x", 0u) == 0u) {
      out.push_back(static_cast<char>(
          std::stoi(std::string{piece.substr(3u, 2u)}, nullptr, 16)));
      continue;
    }
    out += piece;
  }
  // UTF-8 meta space U+2581 (e2 96 81) -> ' '.
  std::string replaced;
  replaced.reserve(out.size());
  for (size_t i = 0; i < out.size();) {
    if (i + 3u <= out.size() && static_cast<uint8_t>(out[i]) == 0xE2u &&
        static_cast<uint8_t>(out[i + 1u]) == 0x96u &&
        static_cast<uint8_t>(out[i + 2u]) == 0x81u) {
      replaced.push_back(' ');
      i += 3u;
      continue;
    }
    replaced.push_back(out[i]);
    ++i;
  }
  return replaced;
}

// Extracts the string value of `key` inside the first <tool_call>...</tool_call>
// block. Returns empty when absent.
std::string extract_call_value(const std::string &text, const char *key) {
  const size_t call_begin = text.find("<tool_call>");
  if (call_begin == std::string::npos) return {};
  const size_t call_end = text.find("</tool_call>", call_begin);
  const std::string_view body{
      text.data() + call_begin,
      (call_end == std::string::npos ? text.size() : call_end) - call_begin};
  const std::string quoted_key = std::string{"\""} + key + "\"";
  const size_t key_pos = body.find(quoted_key);
  if (key_pos == std::string_view::npos) return {};
  size_t cursor = key_pos + quoted_key.size();
  while (cursor < body.size() &&
         (body[cursor] == ' ' || body[cursor] == ':' || body[cursor] == '\t'))
    ++cursor;
  if (cursor >= body.size() || body[cursor] != '"') return {};
  ++cursor;
  const size_t value_end = body.find('"', cursor);
  if (value_end == std::string_view::npos) return {};
  return std::string{body.substr(cursor, value_end - cursor)};
}

int effort_rank(const std::string &effort) {
  if (effort == "low") return 0;
  if (effort == "medium") return 1;
  if (effort == "high") return 2;
  if (effort == "xhigh") return 3;
  return -1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: needle_eval <model.cact> <prompts.tsv> "
                 "[row_begin row_end]\n");
    return 2;
  }
  const std::vector<uint8_t> file_bytes = read_file_bytes(argv[1]);
  std::vector<eval_row> rows = read_rows(argv[2]);
  size_t row_begin = 0u;
  size_t row_end = rows.size();
  if (argc >= 5) {
    row_begin = static_cast<size_t>(std::strtoul(argv[3], nullptr, 10));
    row_end = static_cast<size_t>(std::strtoul(argv[4], nullptr, 10));
    if (row_begin >= rows.size() || row_end > rows.size() ||
        row_begin >= row_end)
      die("bad row range");
  }

  cact_loader::sm loader{};
  cact_loader::geometry geometry = {};
  if (!loader.process_event(cact_loader::event::probe{
          std::span<const uint8_t>{file_bytes}, geometry,
          cact_loader::event::probe_done_fn::from<&on_probe_done>(),
          cact_loader::event::probe_error_fn::from<&on_probe_error>()}))
    die("loader probe");
  std::vector<cact_loader::tensor_view> tensors(geometry.num_tensors);
  if (!loader.process_event(cact_loader::event::bind_storage{
          std::span<cact_loader::tensor_view>{tensors},
          cact_loader::event::bind_done_fn::from<&on_bind_done>(),
          cact_loader::event::bind_error_fn::from<&on_bind_error>()}))
    die("loader bind");
  if (!loader.process_event(cact_loader::event::parse{
          std::span<const uint8_t>{file_bytes},
          cact_loader::event::parse_done_fn::from<&on_parse_done>(),
          cact_loader::event::parse_error_fn::from<&on_parse_error>()}))
    die("loader parse");

  needle::sm binder{};
  needle::contract contract = {};
  if (!binder.process_event(needle::event::bind{
          geometry, std::span<const cact_loader::tensor_view>{tensors},
          contract, needle::event::bind_done_fn::from<&on_needle_done>(),
          needle::event::bind_error_fn::from<&on_needle_error>()}))
    die("needle bind");
  if (!contract.has_tokenizer) die("fixture has no tokenizer blob");

  // Tokenizer: blob -> shared vocab -> shared SPM tokenizer machine.
  auto vocab = std::make_unique<emel::model::data::vocab>();
  emel::text::tokenizer::needle::sm blob_loader{};
  if (!blob_loader.process_event(emel::text::tokenizer::needle::event::load{
          std::span<const uint8_t>{
              contract.tokenizer_blob.data,
              static_cast<size_t>(contract.tokenizer_blob.nbytes)},
          *vocab,
          emel::text::tokenizer::needle::event::load_done_fn::from<
              &on_tok_load_done>(),
          emel::text::tokenizer::needle::event::load_error_fn::from<
              &on_tok_load_error>()}))
    die("tokenizer blob load");

  emel::text::tokenizer::sm tokenizer{};
  int32_t bind_err =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = vocab.get();
  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::spm;
  bind_ev.error_out = &bind_err;
  if (!tokenizer.process_event(bind_ev)) die("tokenizer bind");

  needle::graph::sm graph{contract};
  std::vector<float> logits(contract.geo.vocab_size);
  std::vector<int32_t> token_buffer(contract.geo.max_seq_len);
  std::vector<int32_t> prefill_ids;
  std::vector<int32_t> generated;
  prefill_ids.reserve(contract.geo.max_seq_len);
  generated.reserve(k_max_new_tokens);

  size_t evaluated = 0u, domain_ok = 0u, effort_ok = 0u, joint_ok = 0u;
  size_t no_parse = 0u, within1 = 0u, tokenizer_mismatch = 0u;

  for (size_t index = row_begin; index < row_end; ++index) {
    const eval_row &row = rows[index];

    int32_t native_count = 0;
    int32_t tok_err =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
    emel::text::tokenizer::event::tokenize tok_ev = {};
    tok_ev.vocab = vocab.get();
    tok_ev.text = std::string_view{row.prompt};
    tok_ev.add_special = false;
    tok_ev.parse_special = true;
    tok_ev.token_ids_out = token_buffer.data();
    tok_ev.token_capacity = static_cast<int32_t>(token_buffer.size());
    tok_ev.token_count_out = &native_count;
    tok_ev.error_out = &tok_err;
    if (!tokenizer.process_event(tok_ev)) die("tokenize");

    bool ids_match =
        static_cast<size_t>(native_count) == row.ref_ids.size();
    for (int32_t i = 0; ids_match && i < native_count; ++i)
      ids_match = token_buffer[static_cast<size_t>(i)] ==
                  row.ref_ids[static_cast<size_t>(i)];
    if (!ids_match) {
      ++tokenizer_mismatch;
      die("native tokenizer ids differ from reference");
    }

    prefill_ids.clear();
    prefill_ids.push_back(k_bos_id);
    const size_t prompt_cap =
        contract.geo.max_seq_len > k_max_new_tokens
            ? contract.geo.max_seq_len - k_max_new_tokens
            : 1u;
    for (int32_t i = 0; i < native_count && prefill_ids.size() < prompt_cap;
         ++i)
      prefill_ids.push_back(token_buffer[static_cast<size_t>(i)]);

    if (!graph.process_event(needle::graph::event::init{})) die("graph init");
    if (!graph.process_event(needle::graph::event::prefill{
            std::span<const int32_t>{prefill_ids},
            std::span<float>{logits}}))
      die("graph prefill");

    generated.clear();
    for (uint32_t step = 0u; step < k_max_new_tokens; ++step) {
      const int32_t next = static_cast<int32_t>(
          argmax(std::span<const float>{logits.data(), logits.size()}));
      if (next == k_eos_id) break;
      generated.push_back(next);
      if (step + 1u < k_max_new_tokens) {
        if (!graph.process_event(needle::graph::event::decode{
                next, std::span<float>{logits}}))
          die("graph decode");
      }
    }

    const std::string text =
        detokenize(*vocab, std::span<const int32_t>{generated});
    const std::string pred_domain = extract_call_value(text, "domain");
    const std::string pred_effort = extract_call_value(text, "effort");

    ++evaluated;
    if (pred_domain.empty() && pred_effort.empty()) {
      ++no_parse;
    } else {
      if (pred_domain == row.gold_domain) ++domain_ok;
      if (pred_effort == row.gold_effort) ++effort_ok;
      if (pred_domain == row.gold_domain && pred_effort == row.gold_effort)
        ++joint_ok;
      const int gold_rank = effort_rank(row.gold_effort);
      const int pred_rank = effort_rank(pred_effort);
      if (gold_rank >= 0 && pred_rank >= 0 &&
          (gold_rank - pred_rank <= 1 && pred_rank - gold_rank <= 1))
        ++within1;
    }
    std::printf("row i=%zu gold=%s/%s pred=%s/%s ids_match=%d new_tokens=%zu\n",
                index, row.gold_domain.c_str(), row.gold_effort.c_str(),
                pred_domain.empty() ? "-" : pred_domain.c_str(),
                pred_effort.empty() ? "-" : pred_effort.c_str(),
                ids_match ? 1 : 0, generated.size());
    std::fflush(stdout);
  }

  std::printf("needle_eval_summary rows=%zu no_parse=%zu domain_acc=%.4f "
              "effort_acc=%.4f joint_acc=%.4f effort_within1=%.4f "
              "tokenizer_id_mismatch_rows=%zu\n",
              evaluated, no_parse,
              evaluated ? static_cast<double>(domain_ok) / evaluated : 0.0,
              evaluated ? static_cast<double>(effort_ok) / evaluated : 0.0,
              evaluated ? static_cast<double>(joint_ok) / evaluated : 0.0,
              evaluated ? static_cast<double>(within1) / evaluated : 0.0,
              tokenizer_mismatch);
  return tokenizer_mismatch == 0u ? 0 : 1;
}
