#include <array>
#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/decoder/sm.hpp"
#include "emel/emel.h"

namespace {

struct error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct retryable_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    if constexpr (requires { ev.retryable_out; }) {
      if (ev.retryable_out != nullptr) {
        *ev.retryable_out = true;
      }
    }
  }
};

}  // namespace

TEST_CASE("decoder_sm_on_entry_branches_take_error_paths") {
  emel::decoder::action::context ctx{};
  error_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<
    emel::decoder::model,
    boost::sml::testing,
    emel::decoder::Process> machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(decode));
  CHECK(err == EMEL_ERR_BACKEND);

  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::decoding_done{.request = &decode}));
}

TEST_CASE("decoder_sm_on_entry_retryable_prepare_error") {
  emel::decoder::action::context ctx{};
  retryable_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<
    emel::decoder::model,
    boost::sml::testing,
    emel::decoder::Process> machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));

  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_retryable_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::optimize_memory_done{.request = &decode}));
}
