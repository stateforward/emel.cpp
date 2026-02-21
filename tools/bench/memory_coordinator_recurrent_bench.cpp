#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "emel/batch/splitter/context.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"

#include "llama-batch.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-vocab.h"

namespace {

namespace event = emel::memory::coordinator::event;

constexpr uint32_t k_recurrent_tokens = 64;
constexpr uint32_t k_recurrent_seq_count = 2;
constexpr uint32_t k_recurrent_layers = 1;
constexpr uint32_t k_recurrent_mem_size = 64;
constexpr uint32_t k_recurrent_ubatch = 16;

struct split_inputs {
  std::array<int32_t, k_recurrent_tokens> token_ids = {};
  std::array<int32_t, k_recurrent_tokens> seq_primary_ids = {};
  std::array<llama_pos, k_recurrent_tokens> pos = {};
};

split_inputs make_split_inputs() {
  split_inputs inputs{};
  for (uint32_t i = 0; i < k_recurrent_tokens; ++i) {
    inputs.token_ids[i] = static_cast<int32_t>(i);
    inputs.seq_primary_ids[i] = static_cast<int32_t>(i % k_recurrent_seq_count);
    inputs.pos[i] = static_cast<llama_pos>(i / k_recurrent_seq_count);
  }
  return inputs;
}

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

void reset_split_result(split_result & out) {
  out.ubatch_sizes.fill(0);
  out.ubatch_token_indices.fill(0);
  out.ubatch_offsets.fill(0);
  out.ubatch_count = 0;
  out.token_indices_count = 0;
  out.total_outputs = 0;
  out.err = EMEL_OK;
}

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

event::memory_status map_status(llama_memory_status status) noexcept {
  switch (status) {
    case LLAMA_MEMORY_STATUS_SUCCESS:
      return event::memory_status::success;
    case LLAMA_MEMORY_STATUS_NO_UPDATE:
      return event::memory_status::no_update;
    case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
      return event::memory_status::failed_prepare;
    case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
      return event::memory_status::failed_compute;
  }
  return event::memory_status::failed_prepare;
}

void ensure_backend_init() {
  static bool initialized = false;
  if (!initialized) {
    llama_backend_init();
    initialized = true;
  }
}

void configure_hparams(llama_hparams & hparams) {
  hparams.n_layer = k_recurrent_layers;
  hparams.n_embd = 1;
  hparams.n_ctx_train = k_recurrent_tokens;

  hparams.ssm_d_conv = 2;
  hparams.ssm_d_inner = 1;
  hparams.ssm_n_group = 1;
  hparams.ssm_d_state = 1;

  for (uint32_t i = 0; i < k_recurrent_layers; ++i) {
    hparams.n_head_arr[i] = 1;
    hparams.n_head_kv_arr[i] = 1;
    hparams.n_ff_arr[i] = 1;
    hparams.recurrent_layer_arr[i] = true;
  }
}

llama_model_params make_model_params() {
  llama_model_params params = llama_model_default_params();
  return params;
}

struct split_state {
  static constexpr uint32_t k_embd = 1;

  split_inputs inputs = make_split_inputs();
  llama_batch_allocr allocr;
  llama_vocab vocab;
  llama_batch batch = {};
  std::vector<float> embd;
  std::vector<llama_pos> pos;
  std::vector<int32_t> n_seq_id;
  std::vector<llama_seq_id> seq_ids;
  std::vector<llama_seq_id *> seq_id_ptrs;

  split_state()
      : allocr(1),
        embd(static_cast<size_t>(k_recurrent_tokens) * k_embd, 0.0f),
        pos(static_cast<size_t>(k_recurrent_tokens), 0),
        n_seq_id(static_cast<size_t>(k_recurrent_tokens), 1),
        seq_ids(static_cast<size_t>(k_recurrent_tokens), 0),
        seq_id_ptrs(static_cast<size_t>(k_recurrent_tokens) + 1, nullptr) {
    for (uint32_t i = 0; i < k_recurrent_tokens; ++i) {
      seq_ids[i] = static_cast<llama_seq_id>(inputs.seq_primary_ids[i]);
      seq_id_ptrs[i] = &seq_ids[i];
      pos[i] = inputs.pos[i];
    }
    seq_id_ptrs[k_recurrent_tokens] = nullptr;

    batch.n_tokens = static_cast<int32_t>(k_recurrent_tokens);
    batch.token = nullptr;
    batch.embd = embd.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id_ptrs.data();
    batch.logits = nullptr;

    if (!allocr.init(batch, vocab, nullptr, k_embd, k_recurrent_seq_count, false)) {
      std::fprintf(stderr, "error: llama batch init failed\n");
      std::abort();
    }
  }
};

struct reference_state {
  llama_model model;
  std::unique_ptr<llama_memory_recurrent> memory;
  split_state splitter;
  std::vector<llama_ubatch> ubatches;

  reference_state()
      : model(make_model_params()) {
    ensure_backend_init();
    configure_hparams(model.hparams);
    memory = std::make_unique<llama_memory_recurrent>(
      model,
      GGML_TYPE_F32,
      GGML_TYPE_F32,
      false,
      k_recurrent_mem_size,
      k_recurrent_seq_count,
      llama_memory_recurrent::layer_filter_cb{});
  }
};

bool run_emel_split_equal(const split_inputs & inputs, split_result & out) {
  reset_split_result(out);
  emel::batch::splitter::sm machine{};
  const auto on_done =
      emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
          &splitter_done>();
  const auto on_error =
      emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
          &splitter_error>();

  emel::batch::splitter::event::split request = {
    .token_ids = inputs.token_ids.data(),
    .n_tokens = static_cast<int32_t>(k_recurrent_tokens),
    .n_ubatch = static_cast<int32_t>(k_recurrent_ubatch),
    .mode = emel::batch::splitter::event::split_mode::equal,
    .seq_masks = nullptr,
    .seq_masks_count = 0,
    .seq_primary_ids = inputs.seq_primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(k_recurrent_tokens),
    .equal_sequential = true,
    .seq_mask_words = 1,
    .output_mask = nullptr,
    .output_mask_count = 0,
    .output_all = false,
    .on_done = on_done,
    .on_error = on_error,
  };

  g_split_result = &out;
  (void)machine.process_event(request);
  g_split_result = nullptr;
  return out.err == EMEL_OK;
}

bool run_llama_split_equal(split_state & state, split_result & out) {
  reset_split_result(out);
  state.allocr.split_reset();
  while (true) {
    const auto ubatch = state.allocr.split_equal(k_recurrent_ubatch, true);
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
        pos * static_cast<int32_t>(k_recurrent_seq_count) + seq_id;
      out.token_indices_count += 1;
    }
    out.ubatch_offsets[static_cast<size_t>(out.ubatch_count)] = out.token_indices_count;
  }
  out.total_outputs = static_cast<int32_t>(state.allocr.get_out_ids().size());
  return true;
}

bool compare_split_results(const split_result & lhs, const split_result & rhs) {
  if (lhs.err != EMEL_OK || rhs.err != EMEL_OK) {
    return false;
  }
  if (lhs.ubatch_count != rhs.ubatch_count ||
      lhs.token_indices_count != rhs.token_indices_count ||
      lhs.total_outputs != rhs.total_outputs) {
    return false;
  }
  for (int32_t i = 0; i < lhs.ubatch_count; ++i) {
    if (lhs.ubatch_sizes[static_cast<size_t>(i)] != rhs.ubatch_sizes[static_cast<size_t>(i)]) {
      return false;
    }
  }
  for (int32_t i = 0; i < lhs.token_indices_count; ++i) {
    if (lhs.ubatch_token_indices[static_cast<size_t>(i)] !=
        rhs.ubatch_token_indices[static_cast<size_t>(i)]) {
      return false;
    }
  }
  for (int32_t i = 0; i <= lhs.ubatch_count; ++i) {
    if (lhs.ubatch_offsets[static_cast<size_t>(i)] !=
        rhs.ubatch_offsets[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

void ensure_splitter_parity(split_state & state) {
  static bool checked = false;
  if (checked) {
    return;
  }
  checked = true;

  split_result emel_out;
  split_result llama_out;
  if (!run_emel_split_equal(state.inputs, emel_out) ||
      !run_llama_split_equal(state, llama_out) ||
      !compare_split_results(emel_out, llama_out)) {
    std::fprintf(stderr, "error: splitter parity check failed (recurrent equal)\n");
    std::exit(1);
  }
}

void build_ubatches_from_split(const split_inputs & inputs,
                               const split_result & split,
                               std::vector<llama_ubatch> & out) {
  out.clear();
  out.reserve(static_cast<size_t>(split.ubatch_count));

  for (int32_t ub = 0; ub < split.ubatch_count; ++ub) {
    const int32_t start = split.ubatch_offsets[static_cast<size_t>(ub)];
    const int32_t end = split.ubatch_offsets[static_cast<size_t>(ub + 1)];
    const int32_t n_tokens = end - start;
    if (n_tokens <= 0) {
      continue;
    }

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->pos.resize(static_cast<size_t>(n_tokens));
    data->n_seq_id.resize(static_cast<size_t>(n_tokens), 1);
    data->seq_id.resize(static_cast<size_t>(n_tokens));
    data->seq_id_data.reserve(static_cast<size_t>(n_tokens));
    data->seq_id_unq.clear();
    data->seq_idx.resize(LLAMA_MAX_SEQ, -1);
    data->output.resize(static_cast<size_t>(n_tokens), 0);

    std::array<bool, k_recurrent_seq_count> seen = {};

    for (int32_t i = 0; i < n_tokens; ++i) {
      const int32_t idx = split.ubatch_token_indices[static_cast<size_t>(start + i)];
      const int32_t seq_id = inputs.seq_primary_ids[static_cast<size_t>(idx)];
      data->pos[static_cast<size_t>(i)] = inputs.pos[static_cast<size_t>(idx)];
      data->seq_id_data.push_back(static_cast<llama_seq_id>(seq_id));
      if (seq_id >= 0 && seq_id < static_cast<int32_t>(k_recurrent_seq_count)) {
        if (!seen[static_cast<size_t>(seq_id)]) {
          seen[static_cast<size_t>(seq_id)] = true;
          data->seq_id_unq.push_back(static_cast<llama_seq_id>(seq_id));
        }
      }
    }

    llama_seq_id * seq_id_ptr = data->seq_id_data.data();
    for (int32_t i = 0; i < n_tokens; ++i) {
      data->seq_id[static_cast<size_t>(i)] = seq_id_ptr;
      seq_id_ptr += data->n_seq_id[static_cast<size_t>(i)];
    }

    for (size_t i = 0; i < data->seq_id_unq.size(); ++i) {
      const auto seq_id = data->seq_id_unq[i];
      if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
        data->seq_idx[static_cast<size_t>(seq_id)] = static_cast<int32_t>(i);
      }
    }

    const uint32_t n_seqs = static_cast<uint32_t>(data->seq_id_unq.size());
    if (n_seqs == 0 || n_tokens % static_cast<int32_t>(n_seqs) != 0) {
      std::fprintf(stderr, "error: invalid ubatch sequence layout\n");
      std::abort();
    }

    llama_ubatch ubatch{
      .b_equal_seqs = true,
      .n_tokens = static_cast<uint32_t>(n_tokens),
      .n_seq_tokens = static_cast<uint32_t>(n_tokens) / n_seqs,
      .n_seqs = n_seqs,
      .n_seqs_unq = static_cast<uint32_t>(data->seq_id_unq.size()),
      .n_pos = 1,
      .token = nullptr,
      .embd = nullptr,
      .pos = data->pos.data(),
      .n_seq_id = data->n_seq_id.data(),
      .seq_id = data->seq_id.data(),
      .seq_id_unq = data->seq_id_unq.data(),
      .seq_idx = data->seq_idx.data(),
      .output = data->output.data(),
      .data = std::move(data),
    };
    out.push_back(std::move(ubatch));
  }
}

event::memory_status prepare_update_reference(const event::prepare_update & request,
                                              void * user_data,
                                              int32_t * err_out) noexcept {
  auto * state = static_cast<reference_state *>(user_data);
  auto ctx = state->memory->init_update(nullptr, request.optimize);
  if (!ctx) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return event::memory_status::failed_prepare;
  }
  const auto status = ctx->get_status();
  if (err_out != nullptr) {
    *err_out = (status == LLAMA_MEMORY_STATUS_SUCCESS ||
                status == LLAMA_MEMORY_STATUS_NO_UPDATE)
      ? EMEL_OK
      : EMEL_ERR_BACKEND;
  }
  return map_status(status);
}

event::memory_status prepare_batch_reference(const event::prepare_batch & request,
                                             void * user_data,
                                             int32_t * err_out) noexcept {
  auto * state = static_cast<reference_state *>(user_data);
  if (request.n_ubatch <= 0 || request.n_ubatches_total <= 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return event::memory_status::failed_prepare;
  }
  const bool ok = state->memory->prepare(state->ubatches);
  if (err_out != nullptr) {
    *err_out = ok ? EMEL_OK : EMEL_ERR_BACKEND;
  }
  return ok ? event::memory_status::success : event::memory_status::failed_prepare;
}

event::memory_status prepare_full_reference(const event::prepare_full &,
                                            void * user_data,
                                            int32_t * err_out) noexcept {
  auto * state = static_cast<reference_state *>(user_data);
  auto ctx = state->memory->init_full();
  if (!ctx) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return event::memory_status::failed_prepare;
  }
  const auto status = ctx->get_status();
  if (err_out != nullptr) {
    *err_out = (status == LLAMA_MEMORY_STATUS_SUCCESS ||
                status == LLAMA_MEMORY_STATUS_NO_UPDATE)
      ? EMEL_OK
      : EMEL_ERR_BACKEND;
  }
  return map_status(status);
}

}  // namespace

namespace emel::bench {

void append_emel_memory_coordinator_recurrent_cases(std::vector<result> & results,
                                                    const config & cfg) {
  emel::memory::coordinator::recurrent::sm machine{};
  auto llama_state = std::make_shared<reference_state>();
  ensure_splitter_parity(llama_state->splitter);
  event::memory_status update_status = event::memory_status::failed_prepare;
  event::memory_status batch_status = event::memory_status::failed_prepare;
  event::memory_status full_status = event::memory_status::failed_prepare;
  int32_t update_error = EMEL_OK;
  int32_t batch_error = EMEL_OK;
  int32_t full_error = EMEL_OK;

  event::prepare_update update_request = {
    .optimize = true,
    .status_out = &update_status,
    .error_out = &update_error,
    .prepare_fn = &prepare_update_reference,
    .prepare_ctx = llama_state.get(),
  };
  event::prepare_batch batch_request = {
    .n_ubatch = 1,
    .n_ubatches_total = 1,
    .status_out = &batch_status,
    .error_out = &batch_error,
    .prepare_fn = &prepare_batch_reference,
    .prepare_ctx = llama_state.get(),
  };
  event::prepare_full full_request = {
    .status_out = &full_status,
    .error_out = &full_error,
    .prepare_fn = &prepare_full_reference,
    .prepare_ctx = llama_state.get(),
  };

  auto fn = [&]() {
    split_result split_out;
    if (!run_emel_split_equal(llama_state->splitter.inputs, split_out)) {
      return;
    }
    build_ubatches_from_split(llama_state->splitter.inputs, split_out, llama_state->ubatches);
    (void)machine.process_event(update_request);
    (void)machine.process_event(batch_request);
    (void)machine.process_event(full_request);
  };

  results.push_back(measure_case("memory/coordinator_recurrent_full", cfg, fn));
}

void append_reference_memory_coordinator_recurrent_cases(std::vector<result> & results,
                                                         const config & cfg) {
  auto state = std::make_shared<reference_state>();
  ensure_splitter_parity(state->splitter);
  event::prepare_update update_request = {
    .optimize = true,
  };
  event::prepare_batch batch_request = {
    .n_ubatch = 1,
    .n_ubatches_total = 1,
  };
  event::prepare_full full_request = {};
  auto fn = [&]() {
    split_result split_out;
    if (!run_llama_split_equal(state->splitter, split_out)) {
      return;
    }
    build_ubatches_from_split(state->splitter.inputs, split_out, state->ubatches);
    int32_t err = EMEL_OK;
    (void)prepare_update_reference(update_request, state.get(), &err);
    (void)prepare_batch_reference(batch_request, state.get(), &err);
    (void)prepare_full_reference(full_request, state.get(), &err);
  };
  results.push_back(measure_case("memory/coordinator_recurrent_full", cfg, fn));
}

}  // namespace emel::bench
