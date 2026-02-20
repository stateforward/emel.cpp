#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "emel/batch/sanitizer/sm.hpp"
#include "emel/batch/splitter/context.hpp"
#include "emel/emel.h"

#include "llama-batch.h"
#include "llama-vocab.h"

namespace {

constexpr int32_t k_sanitize_token_count = 128;
constexpr int32_t k_sanitize_seq_count = 4;

struct sanitize_inputs {
  std::array<int32_t, k_sanitize_token_count> tokens = {};
  std::array<int32_t, k_sanitize_token_count> seq_primary_ids = {};
};

sanitize_inputs make_sanitize_inputs() {
  sanitize_inputs inputs{};
  for (int32_t i = 0; i < k_sanitize_token_count; ++i) {
    inputs.tokens[static_cast<size_t>(i)] = i;
    inputs.seq_primary_ids[static_cast<size_t>(i)] = i % k_sanitize_seq_count;
  }
  return inputs;
}

struct reference_sanitize_state {
  static constexpr uint32_t k_embd = 1;

  llama_batch_allocr allocr;
  llama_vocab vocab;
  llama_batch batch = {};
  std::vector<float> embd;

  reference_sanitize_state(uint32_t n_tokens)
      : allocr(1),
        embd(static_cast<size_t>(n_tokens) * k_embd, 0.0f) {
    batch.n_tokens = static_cast<int32_t>(n_tokens);
    batch.token = nullptr;
    batch.embd = embd.data();
    batch.pos = nullptr;
    batch.n_seq_id = nullptr;
    batch.seq_id = nullptr;
    batch.logits = nullptr;
  }

  void reset() {
    batch.pos = nullptr;
    batch.n_seq_id = nullptr;
    batch.seq_id = nullptr;
    batch.logits = nullptr;
  }
};

}  // namespace

namespace emel::bench {

void append_emel_batch_sanitizer_cases(std::vector<result> & results, const config & cfg) {
  emel::batch::sanitizer::sm machine{};
  const auto inputs = make_sanitize_inputs();
  std::array<int32_t, k_sanitize_token_count> seq_primary = {};
  std::array<uint64_t,
             k_sanitize_token_count * emel::batch::splitter::action::SEQ_WORDS>
      seq_masks = {};
  std::array<int32_t, k_sanitize_token_count * 3> positions = {};
  std::array<int8_t, k_sanitize_token_count> output_mask = {};
  int32_t outputs_total = 0;
  int32_t seq_mask_words = 0;
  int32_t positions_count = 0;
  int32_t err = EMEL_OK;

  emel::batch::sanitizer::event::sanitize_decode request = {
    .token_ids = inputs.tokens.data(),
    .n_tokens = k_sanitize_token_count,
    .seq_primary_ids = inputs.seq_primary_ids.data(),
    .seq_primary_ids_count = k_sanitize_token_count,
    .seq_mask_words = 1,
    .seq_primary_ids_out = seq_primary.data(),
    .seq_primary_ids_capacity = static_cast<int32_t>(seq_primary.size()),
    .seq_masks_out = seq_masks.data(),
    .seq_masks_capacity = static_cast<int32_t>(seq_masks.size()),
    .positions_out = positions.data(),
    .positions_capacity = static_cast<int32_t>(positions.size()),
    .output_mask_out = output_mask.data(),
    .output_mask_capacity = static_cast<int32_t>(output_mask.size()),
    .outputs_total_out = &outputs_total,
    .seq_mask_words_out = &seq_mask_words,
    .positions_count_out = &positions_count,
    .error_out = &err,
  };

  auto fn = [&]() { (void)machine.process_event(request); };
  results.push_back(measure_case("batch/sanitizer_decode", cfg, fn));
}

void append_reference_batch_sanitizer_cases(std::vector<result> & results, const config & cfg) {
  reference_sanitize_state state(k_sanitize_token_count);
  auto fn = [&]() {
    state.reset();
    if (!state.allocr.init(
          state.batch,
          state.vocab,
          nullptr,
          reference_sanitize_state::k_embd,
          k_sanitize_seq_count,
          false)) {
      std::fprintf(stderr, "error: llama batch init failed\n");
      std::abort();
    }
  };
  results.push_back(measure_case("batch/sanitizer_decode", cfg, fn));
}

}  // namespace emel::bench
