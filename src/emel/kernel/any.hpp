#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/cuda/sm.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/metal/sm.hpp"
#include "emel/kernel/vulkan/sm.hpp"
#include "emel/kernel/wasm/sm.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "emel/sm.hpp"

namespace emel::kernel {

enum class kernel_kind : uint8_t {
  x86_64 = 0,
  aarch64 = 1,
  wasm = 2,
  cuda = 3,
  metal = 4,
  vulkan = 5,
};

class any {
 public:
  any() = default;
  explicit any(const kernel_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const kernel_kind kind) { core_.set_kind(kind); }

  kernel_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::scaffold & ev) { return core_.process_event(ev); }

  int32_t last_error() const noexcept {
    int32_t err = EMEL_ERR_BACKEND;
    core_.visit([&](const auto & sm) {
      if constexpr (requires { sm.last_error(); }) {
        err = sm.last_error();
      }
    });
    return err;
  }

 private:
  using sm_list = boost::sml::aux::type_list<x86_64::sm, aarch64::sm, wasm::sm, cuda::sm,
                                             metal::sm, vulkan::sm>;
  using event_list = boost::sml::aux::type_list<event::scaffold>;

  emel::sm_any<kernel_kind, sm_list, event_list> core_{};
};

}  // namespace emel::kernel
