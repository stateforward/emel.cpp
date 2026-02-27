#include "bench_cases.hpp"

#include <algorithm>

#include "memory/bench_common.hpp"

#include "emel/memory/hybrid/sm.hpp"

#include "llama-memory-hybrid.h"

namespace {

using memory_sm = emel::memory::hybrid::sm;
namespace memory_bench = emel::bench::memory_bench;

struct reference_state {
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model;
  std::unique_ptr<llama_memory_hybrid> memory;
  llama_batch_allocr allocr;
  memory_bench::reference_batch batch;
  llama_cparams cparams;

  reference_state()
      : model(memory_bench::load_model(memory_bench::resolve_model_path(
          "EMEL_BENCH_HYBRID_MODEL",
          "tests/models/rwkv7-0.1B-g1-F16.gguf"))),
        allocr(model->hparams.n_pos_per_embd()),
        cparams(memory_bench::make_cparams()) {
    const llama_memory_params mparams = memory_bench::make_memory_params();
    const uint32_t rs_size = std::max(1u, cparams.n_seq_max);
    memory = std::make_unique<llama_memory_hybrid>(
        *model,
        mparams.type_k,
        mparams.type_v,
        !cparams.flash_attn,
        cparams.n_ctx_seq,
        1,
        model->hparams.n_swa,
        model->hparams.swa_type,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        rs_size,
        cparams.n_seq_max,
        cparams.offload_kqv,
        cparams.kv_unified,
        nullptr,
        nullptr);
    memory_bench::must_reference_succeed(memory != nullptr, "create_memory(hybrid)");
    memory_bench::must_reference_succeed(
        memory_bench::apply_batch(*memory,
                                  allocr,
                                  *model,
                                  batch,
                                  memory_bench::k_parent_seq,
                                  memory_bench::k_tokens_per_step,
                                  cparams.n_seq_max,
                                  cparams.n_ubatch),
        "allocate parent");
  }
};

void run_reference_cycle(reference_state & state) {
  state.memory->seq_rm(memory_bench::k_branch_child_seq, 0, -1);

  memory_bench::must_reference_succeed(
      memory_bench::apply_batch(*state.memory,
                                state.allocr,
                                *state.model,
                                state.batch,
                                memory_bench::k_work_seq,
                                memory_bench::k_tokens_per_step,
                                state.cparams.n_seq_max,
                                state.cparams.n_ubatch),
      "allocate work");

  state.memory->seq_cp(memory_bench::k_parent_seq,
                       memory_bench::k_branch_child_seq,
                       0,
                       memory_bench::k_tokens_per_step);

  state.memory->seq_rm(memory_bench::k_branch_child_seq, 0, -1);
  state.memory->seq_rm(memory_bench::k_work_seq, 0, -1);
}

}  // namespace

namespace emel::bench {

void append_emel_memory_hybrid_cases(std::vector<result> & results, const config & cfg) {
  memory_sm machine = {};
  memory_bench::lifecycle_state state = {};
  memory_bench::initialize_machine(machine, state);

  auto fn = [&]() {
    memory_bench::run_lifecycle_cycle(machine, state, &memory_bench::recurrent_copy_state);
  };
  results.push_back(measure_case("memory/hybrid_full", cfg, fn));
}

void append_reference_memory_hybrid_cases(std::vector<result> & results, const config & cfg) {
  reference_state state = {};
  auto fn = [&]() { run_reference_cycle(state); };
  results.push_back(measure_case("memory/hybrid_full", cfg, fn));
}

}  // namespace emel::bench
