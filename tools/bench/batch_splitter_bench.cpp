#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "emel/batch/splitter/context.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/emel.h"

#include "llama-batch.h"
#include "llama-vocab.h"

namespace {

constexpr int32_t k_split_token_count = 128;
constexpr int32_t k_split_ubatch = 32;
constexpr int32_t k_split_seq_count = 4;

struct split_result {
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_sizes = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_token_indices = {};
  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES + 1> ubatch_offsets = {};
  int32_t ubatch_count = 0;
  int32_t token_indices_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;
};

split_result * g_split_result = nullptr;

void splitter_done(const emel::batch::splitter::events::splitting_done & done) noexcept {
  if (g_split_result == nullptr) {
    return;
  }
  auto & out = *g_split_result;
  out.err = EMEL_OK;
  out.ubatch_count = done.ubatch_count;
  out.token_indices_count = done.ubatch_token_indices_count;
  out.total_outputs = done.total_outputs;

  for (int32_t i = 0; i < done.ubatch_count; ++i) {
    out.ubatch_sizes[static_cast<size_t>(i)] = done.ubatch_sizes[i];
  }
  for (int32_t i = 0; i < done.ubatch_token_indices_count; ++i) {
    out.ubatch_token_indices[static_cast<size_t>(i)] = done.ubatch_token_indices[i];
  }
  for (int32_t i = 0; i <= done.ubatch_count; ++i) {
    out.ubatch_offsets[static_cast<size_t>(i)] = done.ubatch_token_offsets[i];
  }
}

void splitter_error(const emel::batch::splitter::events::splitting_error & err) noexcept {
  if (g_split_result == nullptr) {
    return;
  }
  g_split_result->err = err.err;
}

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
  std::vector<llama_pos> pos;
  std::vector<int32_t> n_seq_id;
  std::vector<llama_seq_id> seq_ids;
  std::vector<llama_seq_id *> seq_id_ptrs;
  uint32_t n_embd = k_embd;

  reference_split_state(uint32_t n_tokens, uint32_t n_seq_max, uint32_t seq_count)
      : allocr(1),
        embd(static_cast<size_t>(n_tokens) * k_embd, 0.0f),
        pos(static_cast<size_t>(n_tokens), 0),
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
      pos[i] = static_cast<llama_pos>(i / seq_count);
    }
    seq_id_ptrs[n_tokens] = nullptr;

    batch.n_tokens = static_cast<int32_t>(n_tokens);
    batch.token = nullptr;
    batch.embd = embd.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id_ptrs.data();
    batch.logits = nullptr;

    if (!allocr.init(batch, vocab, nullptr, k_embd, n_seq_max, false)) {
      std::fprintf(stderr, "error: llama batch init failed\n");
      std::abort();
    }
  }
};

void reset_split_result(split_result & out) {
  out.ubatch_sizes.fill(0);
  out.ubatch_token_indices.fill(0);
  out.ubatch_offsets.fill(0);
  out.ubatch_count = 0;
  out.token_indices_count = 0;
  out.total_outputs = 0;
  out.err = EMEL_OK;
}

bool collect_emel_split(emel::batch::splitter::event::split_mode mode,
                        const split_inputs & inputs,
                        split_result & out) {
  reset_split_result(out);
  emel::batch::splitter::sm machine{};
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
    .mode = mode,
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

  if (mode != emel::batch::splitter::event::split_mode::simple) {
    request.seq_primary_ids = inputs.seq_primary_ids.data();
    request.seq_primary_ids_count = k_split_token_count;
  }

  g_split_result = &out;
  (void)machine.process_event(request);
  g_split_result = nullptr;

  return out.err == EMEL_OK;
}

bool collect_llama_split(emel::batch::splitter::event::split_mode mode,
                         split_result & out) {
  reset_split_result(out);
  const int32_t seq_count =
    mode == emel::batch::splitter::event::split_mode::simple ? 1 : k_split_seq_count;
  reference_split_state state(k_split_token_count,
                              static_cast<uint32_t>(seq_count),
                              static_cast<uint32_t>(seq_count));
  state.allocr.split_reset();

  auto split_next = [&]() -> llama_ubatch {
    switch (mode) {
      case emel::batch::splitter::event::split_mode::simple:
        return state.allocr.split_simple(k_split_ubatch);
      case emel::batch::splitter::event::split_mode::equal:
        return state.allocr.split_equal(k_split_ubatch, true);
      case emel::batch::splitter::event::split_mode::seq:
        return state.allocr.split_seq(k_split_ubatch);
    }
    return {};
  };

  while (true) {
    const auto ubatch = split_next();
    if (ubatch.n_tokens == 0) {
      break;
    }
    if (out.ubatch_count >= emel::batch::splitter::action::MAX_UBATCHES) {
      out.err = EMEL_ERR_BACKEND;
      return false;
    }
    out.ubatch_sizes[static_cast<size_t>(out.ubatch_count)] =
      static_cast<int32_t>(ubatch.n_tokens);
    out.ubatch_count += 1;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
      if (out.token_indices_count >= emel::batch::splitter::action::MAX_UBATCHES) {
        out.err = EMEL_ERR_BACKEND;
        return false;
      }
      const int32_t seq_id = ubatch.seq_id[i][0];
      const int32_t pos = static_cast<int32_t>(ubatch.pos[i]);
      out.ubatch_token_indices[static_cast<size_t>(out.token_indices_count)] =
        pos * seq_count + seq_id;
      out.token_indices_count += 1;
    }
    out.ubatch_offsets[static_cast<size_t>(out.ubatch_count)] = out.token_indices_count;
  }

  out.total_outputs = static_cast<int32_t>(state.allocr.get_out_ids().size());
  return true;
}

bool compare_split_results(const split_result & lhs,
                           const split_result & rhs,
                           const char * label) {
  if (lhs.err != EMEL_OK || rhs.err != EMEL_OK) {
    std::fprintf(stderr, "error: splitter parity failed (%s): err %d vs %d\n",
                 label, lhs.err, rhs.err);
    return false;
  }
  if (lhs.ubatch_count != rhs.ubatch_count) {
    std::fprintf(stderr, "error: splitter parity failed (%s): ubatch_count %d vs %d\n",
                 label, lhs.ubatch_count, rhs.ubatch_count);
    return false;
  }
  if (lhs.token_indices_count != rhs.token_indices_count) {
    std::fprintf(stderr, "error: splitter parity failed (%s): token_count %d vs %d\n",
                 label, lhs.token_indices_count, rhs.token_indices_count);
    return false;
  }
  if (lhs.total_outputs != rhs.total_outputs) {
    std::fprintf(stderr, "error: splitter parity failed (%s): total_outputs %d vs %d\n",
                 label, lhs.total_outputs, rhs.total_outputs);
    return false;
  }

  for (int32_t i = 0; i < lhs.ubatch_count; ++i) {
    if (lhs.ubatch_sizes[static_cast<size_t>(i)] != rhs.ubatch_sizes[static_cast<size_t>(i)]) {
      std::fprintf(stderr, "error: splitter parity failed (%s): ubatch_size[%d] %d vs %d\n",
                   label,
                   i,
                   lhs.ubatch_sizes[static_cast<size_t>(i)],
                   rhs.ubatch_sizes[static_cast<size_t>(i)]);
      return false;
    }
  }
  for (int32_t i = 0; i < lhs.token_indices_count; ++i) {
    if (lhs.ubatch_token_indices[static_cast<size_t>(i)] !=
        rhs.ubatch_token_indices[static_cast<size_t>(i)]) {
      std::fprintf(stderr, "error: splitter parity failed (%s): token_idx[%d] %d vs %d\n",
                   label,
                   i,
                   lhs.ubatch_token_indices[static_cast<size_t>(i)],
                   rhs.ubatch_token_indices[static_cast<size_t>(i)]);
      return false;
    }
  }
  for (int32_t i = 0; i <= lhs.ubatch_count; ++i) {
    if (lhs.ubatch_offsets[static_cast<size_t>(i)] != rhs.ubatch_offsets[static_cast<size_t>(i)]) {
      std::fprintf(stderr, "error: splitter parity failed (%s): offset[%d] %d vs %d\n",
                   label,
                   i,
                   lhs.ubatch_offsets[static_cast<size_t>(i)],
                   rhs.ubatch_offsets[static_cast<size_t>(i)]);
      return false;
    }
  }
  return true;
}

void ensure_splitter_parity() {
  static bool checked = false;
  if (checked) {
    return;
  }
  checked = true;

  const auto inputs = make_split_inputs();
  split_result emel_out;
  split_result llama_out;

  if (!collect_emel_split(emel::batch::splitter::event::split_mode::simple,
                          inputs,
                          emel_out) ||
      !collect_llama_split(emel::batch::splitter::event::split_mode::simple, llama_out) ||
      !compare_split_results(emel_out, llama_out, "simple")) {
    std::fprintf(stderr, "error: splitter parity check failed (simple)\n");
    std::exit(1);
  }

  if (!collect_emel_split(emel::batch::splitter::event::split_mode::equal,
                          inputs,
                          emel_out) ||
      !collect_llama_split(emel::batch::splitter::event::split_mode::equal, llama_out) ||
      !compare_split_results(emel_out, llama_out, "equal")) {
    std::fprintf(stderr, "error: splitter parity check failed (equal)\n");
    std::exit(1);
  }

  if (!collect_emel_split(emel::batch::splitter::event::split_mode::seq,
                          inputs,
                          emel_out) ||
      !collect_llama_split(emel::batch::splitter::event::split_mode::seq, llama_out) ||
      !compare_split_results(emel_out, llama_out, "seq")) {
    std::fprintf(stderr, "error: splitter parity check failed (seq)\n");
    std::exit(1);
  }
}

}  // namespace

namespace emel::bench {

void append_emel_batch_splitter_cases(std::vector<result> & results, const config & cfg) {
  ensure_splitter_parity();
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
