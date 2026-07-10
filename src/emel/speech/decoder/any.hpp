#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/speech/decoder/events.hpp"
#include "emel/speech/decoder/whisper/sm.hpp"

namespace emel::speech::decoder {

// Component variant selector. `unsupported` is a sentinel used by owners (e.g.
// speech/transcriber dependencies) to model "no decoder injected"; owners must
// guard dispatch on a supported kind. sm_any clamps out-of-range kinds to the
// first variant, so constructing a facade with `unsupported` is safe as long as
// no event is dispatched to it.
enum class decoder_kind : uint8_t {
  whisper = 0,
  unsupported = 255,
};

class any {
public:
  any() = default;
  explicit any(const decoder_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any &operator=(const any &) = delete;
  any(any &&) = delete;
  any &operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const decoder_kind kind) { core_.set_kind(kind); }

  decoder_kind kind() const noexcept { return core_.kind(); }

  bool process_event(const event::decode &ev) {
    return core_.process_event(ev);
  }

private:
  using sm_list = stateforward::sml::aux::type_list<whisper::sm>;
  using event_list = stateforward::sml::aux::type_list<event::decode>;

  emel::sm_any<decoder_kind, sm_list, event_list> core_{};
};

} // namespace emel::speech::decoder
