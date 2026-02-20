#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "emel/batch/splitter/events.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/emel.h"

#include "llama-batch.h"
#include "llama-vocab.h"

namespace {

constexpr int32_t k_split_token_count = 128;
constexpr int32_t k_split_ubatch = 32;
constexpr int32_t k_split_seq_count = 4;

void splitter_done(const emel::batch::splitter::events::splitting_done &) noexcept {}
void splitter_error(const emel::batch::splitter::events::splitting_error &) noexcept {}

struct split_inputs {
  std::array<int32_t, k_split_token_count> tokens = {};
  std::array<int32_t, k_split_token_count> seq_primary_ids = {};
};

split_inputs make_split_inputs() {
  split_inputs inputs{};
  for (int32_t i = 0; i < k_split_token_count; ++i) {
    inputs.tokens[static_cast<size_t>(i)] = i;
    inputs.seq_primary_ids[static_cast<size_t>(i)] = i % k_split_seq_count;
  }
  return inputs;
}

struct reference_split_state {
  static constexpr uint32_t k_embd = 1;

  llama_batch_allocr allocr;
  llama_vocab vocab;
  llama_batch batch = {};
  std::vector<float> embd;
  std::vector<int32_t> n_seq_id;
  std::vector<llama_seq_id> seq_ids;
  std::vector<llama_seq_id *> seq_id_ptrs;
  uint32_t n_embd = k_embd;

  reference_split_state(uint32_t n_tokens, uint32_t n_seq_max, uint32_t seq_count)
      : allocr(1),
        embd(static_cast<size_t>(n_tokens) * k_embd, 0.0f),
        n_seq_id(static_cast<size_t>(n_tokens), 1),
        seq_ids(static_cast<size_t>(n_tokens), 0),
        seq_id_ptrs(static_cast<size_t>(n_tokens) + 1, nullptr) {
    if (seq_count == 0) {
      seq_count = 1;
    }
    if (n_seq_max < seq_count) {
      n_seq_max = seq_count;
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
      seq_ids[i] = static_cast<llama_seq_id>(i % seq_count);
      seq_id_ptrs[i] = &seq_ids[i];
    }
    seq_id_ptrs[n_tokens] = nullptr;

    batch.n_tokens = static_cast<int32_t>(n_tokens);
    batch.token = nullptr;
    batch.embd = embd.data();
    batch.pos = nullptr;
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id_ptrs.data();
    batch.logits = nullptr;

    if (!allocr.init(batch, vocab, nullptr, k_embd, n_seq_max, false)) {
      std::fprintf(stderr, "error: llama batch init failed\n");
      std::abort();
    }
  }
};

}  // namespace

namespace emel::bench {

void append_emel_batch_splitter_cases(std::vector<result> & results, const config & cfg) {
  {
    emel::batch::splitter::sm machine{};
    const auto inputs = make_split_inputs();
    const auto on_done =
        emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
            &splitter_done>();
    const auto on_error =
        emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
            &splitter_error>();

    emel::batch::splitter::event::split request = {
      .token_ids = inputs.tokens.data(),
      .n_tokens = k_split_token_count,
      .n_ubatch = k_split_ubatch,
      .mode = emel::batch::splitter::event::split_mode::simple,
      .seq_masks = nullptr,
      .seq_masks_count = 0,
      .seq_primary_ids = nullptr,
      .seq_primary_ids_count = 0,
      .equal_sequential = true,
      .seq_mask_words = 1,
      .output_mask = nullptr,
      .output_mask_count = 0,
      .output_all = false,
      .on_done = on_done,
      .on_error = on_error,
    };

    auto fn = [&]() { (void)machine.process_event(request); };
    results.push_back(measure_case("batch/splitter_simple", cfg, fn));
  }

  {
    emel::batch::splitter::sm machine{};
    const auto inputs = make_split_inputs();
    const auto on_done =
        emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
            &splitter_done>();
    const auto on_error =
        emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
            &splitter_error>();

    emel::batch::splitter::event::split request = {
      .token_ids = inputs.tokens.data(),
      .n_tokens = k_split_token_count,
      .n_ubatch = k_split_ubatch,
      .mode = emel::batch::splitter::event::split_mode::equal,
      .seq_masks = nullptr,
      .seq_masks_count = 0,
      .seq_primary_ids = inputs.seq_primary_ids.data(),
      .seq_primary_ids_count = k_split_token_count,
      .equal_sequential = true,
      .seq_mask_words = 1,
      .output_mask = nullptr,
      .output_mask_count = 0,
      .output_all = false,
      .on_done = on_done,
      .on_error = on_error,
    };

    auto fn = [&]() { (void)machine.process_event(request); };
    results.push_back(measure_case("batch/splitter_equal", cfg, fn));
  }

  {
    emel::batch::splitter::sm machine{};
    const auto inputs = make_split_inputs();
    const auto on_done =
        emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
            &splitter_done>();
    const auto on_error =
        emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
            &splitter_error>();

    emel::batch::splitter::event::split request = {
      .token_ids = inputs.tokens.data(),
      .n_tokens = k_split_token_count,
      .n_ubatch = k_split_ubatch,
      .mode = emel::batch::splitter::event::split_mode::seq,
      .seq_masks = nullptr,
      .seq_masks_count = 0,
      .seq_primary_ids = inputs.seq_primary_ids.data(),
      .seq_primary_ids_count = k_split_token_count,
      .equal_sequential = true,
      .seq_mask_words = 1,
      .output_mask = nullptr,
      .output_mask_count = 0,
      .output_all = false,
      .on_done = on_done,
      .on_error = on_error,
    };

    auto fn = [&]() { (void)machine.process_event(request); };
    results.push_back(measure_case("batch/splitter_seq", cfg, fn));
  }
}

void append_reference_batch_splitter_cases(std::vector<result> & results, const config & cfg) {
  {
    reference_split_state state(k_split_token_count, 1, 1);
    auto fn = [&]() {
      state.allocr.split_reset();
      while (true) {
        const auto ubatch = state.allocr.split_simple(k_split_ubatch);
        if (ubatch.n_tokens == 0) {
          break;
        }
      }
    };
    results.push_back(measure_case("batch/splitter_simple", cfg, fn));
  }

  {
    reference_split_state state(k_split_token_count, k_split_seq_count, k_split_seq_count);
    auto fn = [&]() {
      state.allocr.split_reset();
      while (true) {
        const auto ubatch = state.allocr.split_equal(k_split_ubatch, true);
        if (ubatch.n_tokens == 0) {
          break;
        }
      }
    };
    results.push_back(measure_case("batch/splitter_equal", cfg, fn));
  }

  {
    reference_split_state state(k_split_token_count, k_split_seq_count, k_split_seq_count);
    auto fn = [&]() {
      state.allocr.split_reset();
      while (true) {
        const auto ubatch = state.allocr.split_seq(k_split_ubatch);
        if (ubatch.n_tokens == 0) {
          break;
        }
      }
    };
    results.push_back(measure_case("batch/splitter_seq", cfg, fn));
  }
}

}  // namespace emel::bench
