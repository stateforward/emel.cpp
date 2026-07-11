#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/speech/decoder/events.hpp"
#include "emel/speech/decoder/whisper/sm.hpp"

namespace emel::speech::decoder {

// Component variant selector. `unsupported` is a sentinel used by owners (e.g.
// speech/transcriber dependencies) to model "no decoder injected"; owners must
// guard dispatch on is_supported(kind()). Constructing a facade with
// `unsupported` (or an out-of-range cast) is safe: the active machine clamps
// to the first variant, but kind() preserves the requested value so the guard
// stays truthful. No event may be dispatched to an unsupported facade.
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

  // True only for kinds that name a real variant; owners must reject every
  // other value (including `unsupported` and out-of-range casts) before
  // dispatch, since the facade would otherwise clamp them to the first variant.
  static constexpr bool is_supported(const decoder_kind kind) noexcept {
    return emel::sm_any<decoder_kind, sm_list, event_list>::is_supported_kind(
        kind);
  }

  bool process_event(const event::decode &ev) {
    return core_.process_event(ev);
  }

private:
  using sm_list = stateforward::sml::aux::type_list<whisper::sm>;
  using event_list = stateforward::sml::aux::type_list<event::decode>;

  emel::sm_any<decoder_kind, sm_list, event_list> core_{};
};

} // namespace emel::speech::decoder
