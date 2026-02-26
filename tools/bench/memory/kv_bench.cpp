#include "bench_cases.hpp"

#include "memory/bench_common.hpp"

#include "emel/memory/kv/sm.hpp"

namespace {

using memory_sm = emel::memory::kv::sm;
namespace memory_bench = emel::bench::memory_bench;

struct reference_state {
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model;
  llama_memory_ptr memory;
  llama_batch_allocr allocr;
  memory_bench::reference_batch batch;
  llama_cparams cparams;

  reference_state()
      : model(memory_bench::load_model(memory_bench::resolve_model_path(
          "EMEL_BENCH_KV_MODEL",
          "tests/models/Llama-68M-Chat-v1-Q2_K.gguf"))),
        allocr(model->hparams.n_pos_per_embd()),
        cparams(memory_bench::make_cparams()) {
    const llama_memory_params mparams = memory_bench::make_memory_params();
    memory.reset(model->create_memory(mparams, cparams));
    memory_bench::must_reference_succeed(memory != nullptr, "create_memory(kv)");
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

void append_emel_memory_kv_cases(std::vector<result> & results, const config & cfg) {
  memory_sm machine = {};
  memory_bench::lifecycle_state state = {};
  memory_bench::initialize_machine(machine, state);

  auto fn = [&]() { memory_bench::run_lifecycle_cycle(machine, state, nullptr); };
  results.push_back(measure_case("memory/kv_full", cfg, fn));
}

void append_reference_memory_kv_cases(std::vector<result> & results, const config & cfg) {
  reference_state state = {};
  auto fn = [&]() { run_reference_cycle(state); };
  results.push_back(measure_case("memory/kv_full", cfg, fn));
}

}  // namespace emel::bench
