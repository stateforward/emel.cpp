#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/encoder/events.hpp"
#include "emel/encoder/bpe/sm.hpp"
#include "emel/encoder/fallback/sm.hpp"
#include "emel/encoder/plamo2/sm.hpp"
#include "emel/encoder/rwkv/sm.hpp"
#include "emel/encoder/spm/sm.hpp"
#include "emel/encoder/ugm/sm.hpp"
#include "emel/encoder/wpm/sm.hpp"
#include "emel/sm.hpp"

namespace emel::encoder {

enum class encoder_kind : uint8_t {
  spm = 0,
  bpe = 1,
  wpm = 2,
  ugm = 3,
  rwkv = 4,
  plamo2 = 5,
  fallback = 6,
};

class any {
 public:
  any() = default;
  explicit any(const encoder_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const encoder_kind kind) { core_.set_kind(kind); }

  encoder_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::encode & ev) { return core_.process_event(ev); }

  int32_t last_error() const noexcept {
    int32_t err = EMEL_ERR_BACKEND;
    core_.visit([&](const auto & sm) { err = sm.last_error(); });
    return err;
  }

 private:
  using sm_list = boost::sml::aux::type_list<spm::sm, bpe::sm, wpm::sm, ugm::sm,
                                             rwkv::sm, plamo2::sm, fallback::sm>;
  using event_list = boost::sml::aux::type_list<event::encode>;

  emel::sm_any<encoder_kind, sm_list, event_list> core_{};
};

}  // namespace emel::encoder
