#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/speech/tokenizer/events.hpp"
#include "emel/speech/tokenizer/whisper/sm.hpp"

namespace emel::speech::tokenizer {

// Component variant selector. `unsupported` is a sentinel used by owners (e.g.
// speech/transcriber dependencies) to model "no tokenizer injected"; owners
// must guard dispatch on a supported kind. sm_any clamps out-of-range kinds to
// the first variant, so constructing a facade with `unsupported` is safe as
// long as no event is dispatched to it.
enum class tokenizer_kind : uint8_t {
  whisper = 0,
  unsupported = 255,
};

class any {
public:
  any() = default;
  explicit any(const tokenizer_kind kind) : core_(kind) {}

  any(const any &) = delete;
  any &operator=(const any &) = delete;
  any(any &&) = delete;
  any &operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const tokenizer_kind kind) { core_.set_kind(kind); }

  tokenizer_kind kind() const noexcept { return core_.kind(); }

  // True only for kinds that name a real variant; owners must reject every
  // other value (including `unsupported` and out-of-range casts) before
  // dispatch, since the facade would otherwise clamp them to the first variant.
  static constexpr bool is_supported(const tokenizer_kind kind) noexcept {
    return emel::sm_any<tokenizer_kind, sm_list, event_list>::is_supported_kind(
        kind);
  }

  bool process_event(const event::detokenize &ev) {
    return core_.process_event(ev);
  }

private:
  using sm_list = stateforward::sml::aux::type_list<whisper::sm>;
  using event_list = stateforward::sml::aux::type_list<event::detokenize>;

  emel::sm_any<tokenizer_kind, sm_list, event_list> core_{};
};

} // namespace emel::speech::tokenizer
