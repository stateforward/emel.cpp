#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/tokenizer/preprocessor/events.hpp"
#include "emel/tokenizer/preprocessor/bpe/sm.hpp"
#include "emel/tokenizer/preprocessor/fallback/sm.hpp"
#include "emel/tokenizer/preprocessor/plamo2/sm.hpp"
#include "emel/tokenizer/preprocessor/rwkv/sm.hpp"
#include "emel/tokenizer/preprocessor/spm/sm.hpp"
#include "emel/tokenizer/preprocessor/ugm/sm.hpp"
#include "emel/tokenizer/preprocessor/wpm/sm.hpp"

namespace emel::tokenizer::preprocessor {

enum class preprocessor_kind : uint8_t {
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
  explicit any(const preprocessor_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const preprocessor_kind kind) { core_.set_kind(kind); }

  preprocessor_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::preprocess & ev) { return core_.process_event(ev); }

  int32_t last_error() const noexcept {
    int32_t err = EMEL_ERR_BACKEND;
    core_.visit([&](const auto & sm) { err = sm.last_error(); });
    return err;
  }

 private:
  using sm_list = boost::sml::aux::type_list<spm::sm, bpe::sm, wpm::sm, ugm::sm,
                                             rwkv::sm, plamo2::sm, fallback::sm>;
  using event_list = boost::sml::aux::type_list<event::preprocess>;

  emel::sm_any<preprocessor_kind, sm_list, event_list> core_{};
};

}  // namespace emel::tokenizer::preprocessor
