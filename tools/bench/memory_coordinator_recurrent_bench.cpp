#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "emel/emel.h"
#include "emel/memory/coordinator/any.hpp"

namespace {

namespace event = emel::memory::coordinator::event;
using coordinator_any = emel::memory::coordinator::any;
using coordinator_kind = emel::memory::coordinator::coordinator_kind;

constexpr int32_t k_max_sequences = 64;
constexpr int32_t k_max_blocks = 1024;
constexpr int32_t k_block_tokens = 16;
constexpr int32_t k_parent_seq = 0;
constexpr int32_t k_branch_child_seq = 1;
constexpr int32_t k_work_seq = 2;
constexpr int32_t k_tokens_per_step = 16;

struct bench_state {
  coordinator_any machine = {};
  int32_t copy_calls = 0;
};

bool recurrent_copy_state(const int32_t, const int32_t, void * user_data, int32_t * error_out) {
  if (user_data != nullptr) {
    auto * calls = static_cast<int32_t *>(user_data);
    *calls += 1;
  }
  if (error_out != nullptr) {
    *error_out = EMEL_OK;
  }
  return true;
}

void must_succeed(const bool accepted, const int32_t err, const char * step) {
  if (accepted && err == EMEL_OK) {
    return;
  }
  std::fprintf(stderr, "error: memory bench setup failed at %s (accepted=%d err=%d)\n",
               step, accepted ? 1 : 0, err);
  std::abort();
}

void initialize_for_kind(bench_state & state, const coordinator_kind kind) {
  state.machine.set_kind(kind);
  state.copy_calls = 0;

  int32_t err = EMEL_OK;
  must_succeed(state.machine.process_event(event::reserve{
                 .max_sequences = k_max_sequences,
                 .max_blocks = k_max_blocks,
                 .block_tokens = k_block_tokens,
                 .error_out = &err,
               }),
               err,
               "reserve");

  err = EMEL_OK;
  must_succeed(state.machine.process_event(event::allocate_sequence{
                 .seq_id = k_parent_seq,
                 .error_out = &err,
               }),
               err,
               "allocate_sequence(parent)");

  err = EMEL_OK;
  must_succeed(state.machine.process_event(event::allocate_slots{
                 .seq_id = k_parent_seq,
                 .token_count = k_tokens_per_step,
                 .error_out = &err,
               }),
               err,
               "allocate_slots(parent)");
}

void run_lifecycle_cycle(bench_state & state) {
  int32_t err = EMEL_OK;

  (void)state.machine.process_event(event::free_sequence{
    .seq_id = k_branch_child_seq,
    .error_out = &err,
  });

  err = EMEL_OK;
  (void)state.machine.process_event(event::allocate_sequence{
    .seq_id = k_work_seq,
    .error_out = &err,
  });

  err = EMEL_OK;
  (void)state.machine.process_event(event::allocate_slots{
    .seq_id = k_work_seq,
    .token_count = k_tokens_per_step,
    .error_out = &err,
  });

  err = EMEL_OK;
  (void)state.machine.process_event(event::branch_sequence{
    .parent_seq_id = k_parent_seq,
    .child_seq_id = k_branch_child_seq,
    .copy_state = &recurrent_copy_state,
    .copy_state_user_data = &state.copy_calls,
    .error_out = &err,
  });

  err = EMEL_OK;
  (void)state.machine.process_event(event::free_sequence{
    .seq_id = k_branch_child_seq,
    .error_out = &err,
  });

  err = EMEL_OK;
  (void)state.machine.process_event(event::free_sequence{
    .seq_id = k_work_seq,
    .error_out = &err,
  });
}

template <size_t n>
void initialize_all(std::array<bench_state, n> & states) {
  initialize_for_kind(states[0], coordinator_kind::recurrent);
  initialize_for_kind(states[1], coordinator_kind::kv);
  initialize_for_kind(states[2], coordinator_kind::hybrid);
}

template <size_t n>
void run_next_mode(std::array<bench_state, n> & states, size_t & next_mode) {
  run_lifecycle_cycle(states[next_mode]);
  next_mode = (next_mode + 1) % states.size();
}

}  // namespace

namespace emel::bench {

void append_emel_memory_coordinator_recurrent_cases(std::vector<result> & results,
                                                    const config & cfg) {
  std::array<bench_state, 3> states = {};
  initialize_all(states);
  size_t next_mode = 0;

  auto fn = [&]() { run_next_mode(states, next_mode); };
  results.push_back(measure_case("memory/coordinator_recurrent_full", cfg, fn));
}

void append_reference_memory_coordinator_recurrent_cases(std::vector<result> & results,
                                                         const config & cfg) {
  std::array<bench_state, 3> states = {};
  initialize_all(states);
  size_t next_mode = 0;

  auto fn = [&]() { run_next_mode(states, next_mode); };
  results.push_back(measure_case("memory/coordinator_recurrent_full", cfg, fn));
}

}  // namespace emel::bench
