#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "emel/emel.h"
#include "emel/memory/events.hpp"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-memory.h"
#include "llama-model.h"

namespace emel::bench::memory_bench {

namespace event = emel::memory::event;

enum class bench_error : int32_t {
  none = 0,
  backend = (1 << 0),
  internal = (1 << 1),
};

constexpr int32_t error_code(const bench_error code) noexcept {
  return static_cast<int32_t>(code);
}

constexpr int32_t k_error_none = error_code(bench_error::none);

constexpr int32_t k_max_sequences = 64;
constexpr int32_t k_max_blocks = 1024;
constexpr int32_t k_block_tokens = 16;
constexpr int32_t k_parent_seq = 0;
constexpr int32_t k_branch_child_seq = 1;
constexpr int32_t k_work_seq = 2;
constexpr int32_t k_tokens_per_step = 16;

struct lifecycle_state {
  int32_t copy_calls = 0;
};

struct reference_batch {
  llama_batch batch = {};
  std::vector<llama_token> tokens;
  std::vector<llama_pos> pos;
  std::vector<int32_t> n_seq_id;
  std::vector<llama_seq_id> seq_id_data;
  std::vector<llama_seq_id *> seq_id_ptrs;
  std::vector<int8_t> output;

  void reset(const int32_t seq_id, const int32_t token_count) {
    const size_t count = static_cast<size_t>(token_count);
    tokens.assign(count, 0);
    pos.resize(count);
    n_seq_id.assign(count, 1);
    seq_id_data.resize(count);
    seq_id_ptrs.resize(count);
    output.assign(count, 0);

    for (size_t i = 0; i < count; ++i) {
      pos[i] = static_cast<llama_pos>(i);
      seq_id_data[i] = static_cast<llama_seq_id>(seq_id);
      seq_id_ptrs[i] = &seq_id_data[i];
    }

    batch.n_tokens = static_cast<int32_t>(count);
    batch.token = tokens.data();
    batch.embd = nullptr;
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id_ptrs.data();
    batch.logits = output.data();
  }
};

inline bool recurrent_copy_state(const int32_t, const int32_t, void * user_data,
                                 int32_t * error_out) {
  if (user_data != nullptr) {
    auto * calls = static_cast<int32_t *>(user_data);
    *calls += 1;
  }
  if (error_out != nullptr) {
    *error_out = k_error_none;
  }
  return true;
}

inline void must_succeed(const bool accepted, const int32_t err, const char * step) {
  if (accepted && err == k_error_none) {
    return;
  }
  std::fprintf(stderr, "error: memory bench setup failed at %s (accepted=%d err=%d)\n",
               step, accepted ? 1 : 0, err);
  std::abort();
}

inline void must_reference_succeed(const bool ok, const char * step) {
  if (ok) {
    return;
  }
  std::fprintf(stderr, "error: reference memory bench setup failed at %s\n", step);
  std::abort();
}

inline std::filesystem::path bench_root_path() {
  std::filesystem::path path = std::filesystem::path(__FILE__).parent_path();
  path = path.parent_path().parent_path().parent_path();
  return path;
}

inline std::string resolve_model_path(const char * env_var, const char * fallback_rel) {
  const char * override_path = std::getenv(env_var);
  if (override_path != nullptr && override_path[0] != '\0') {
    return std::string(override_path);
  }
  return (bench_root_path() / fallback_rel).string();
}

inline std::unique_ptr<llama_model, decltype(&llama_model_free)>
load_model(const std::string & path) {
  llama_model_params params = llama_model_default_params();
  params.no_alloc = true;
  params.use_mmap = false;

  llama_model * model = llama_model_load_from_file(path.c_str(), params);
  if (model == nullptr) {
    std::fprintf(stderr, "error: failed to load llama model from %s\n", path.c_str());
    std::abort();
  }
  return std::unique_ptr<llama_model, decltype(&llama_model_free)>(model, llama_model_free);
}

inline llama_cparams make_cparams() {
  llama_cparams cparams = {};
  cparams.n_ctx_seq = static_cast<uint32_t>(k_max_blocks * k_block_tokens);
  cparams.n_seq_max = static_cast<uint32_t>(k_max_sequences);
  cparams.n_ubatch = static_cast<uint32_t>(k_tokens_per_step);
  cparams.kv_unified = true;
  cparams.offload_kqv = false;
  cparams.flash_attn = false;
  return cparams;
}

inline llama_memory_params make_memory_params() {
  llama_memory_params params = {};
  params.type_k = GGML_TYPE_F16;
  params.type_v = GGML_TYPE_F16;
  params.swa_full = false;
  return params;
}

inline bool apply_batch(llama_memory_i & memory,
                        llama_batch_allocr & allocr,
                        const llama_model & model,
                        reference_batch & batch,
                        const int32_t seq_id,
                        const uint32_t token_count,
                        const uint32_t n_seq_max,
                        const uint32_t n_ubatch) {
  batch.reset(seq_id, static_cast<int32_t>(token_count));
  const uint32_t n_embd = model.hparams.n_embd_inp();

  if (!allocr.init(batch.batch, model.vocab, &memory, n_embd, n_seq_max, false)) {
    return false;
  }

  auto ctx = memory.init_batch(allocr, n_ubatch, false);
  if (!ctx || ctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
    return false;
  }

  if (!ctx->apply()) {
    return false;
  }
  while (ctx->next()) {
    if (!ctx->apply()) {
      return false;
    }
  }

  return true;
}

template <class machine_type>
void initialize_machine(machine_type & machine, lifecycle_state & state) {
  state.copy_calls = 0;

  int32_t err = k_error_none;
  must_succeed(machine.process_event(event::reserve{
                 .max_sequences = k_max_sequences,
                 .max_blocks = k_max_blocks,
                 .block_tokens = k_block_tokens,
                 .error_out = &err,
               }),
               err,
               "reserve");

  err = k_error_none;
  must_succeed(machine.process_event(event::allocate_sequence{
                 .seq_id = k_parent_seq,
                 .error_out = &err,
               }),
               err,
               "allocate_sequence(parent)");

  err = k_error_none;
  must_succeed(machine.process_event(event::allocate_slots{
                 .seq_id = k_parent_seq,
                 .token_count = k_tokens_per_step,
                 .error_out = &err,
               }),
               err,
               "allocate_slots(parent)");
}

template <class machine_type>
void run_lifecycle_cycle(machine_type & machine, lifecycle_state & state,
                         event::branch_sequence::copy_state_fn copy_state) {
  int32_t err = k_error_none;

  (void)machine.process_event(event::free_sequence{
    .seq_id = k_branch_child_seq,
    .error_out = &err,
  });

  err = k_error_none;
  (void)machine.process_event(event::allocate_sequence{
    .seq_id = k_work_seq,
    .error_out = &err,
  });

  err = k_error_none;
  (void)machine.process_event(event::allocate_slots{
    .seq_id = k_work_seq,
    .token_count = k_tokens_per_step,
    .error_out = &err,
  });

  err = k_error_none;
  (void)machine.process_event(event::branch_sequence{
    .parent_seq_id = k_parent_seq,
    .child_seq_id = k_branch_child_seq,
    .copy_state = copy_state,
    .copy_state_user_data = copy_state == nullptr ? nullptr : &state.copy_calls,
    .error_out = &err,
  });

  err = k_error_none;
  (void)machine.process_event(event::free_sequence{
    .seq_id = k_branch_child_seq,
    .error_out = &err,
  });

  err = k_error_none;
  (void)machine.process_event(event::free_sequence{
    .seq_id = k_work_seq,
    .error_out = &err,
  });
}

}  // namespace emel::bench::memory_bench
