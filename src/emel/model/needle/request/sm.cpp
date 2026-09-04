#include "emel/model/needle/request/sm.hpp"

namespace emel::model::needle::request {


bool sm::process_event(const event::configure &ev) {
  action::reset_outputs(this->context_);
  event::configure_ctx ctx{};
  event::configure_run runtime{ev, ctx};
  const bool accepted = base_type::process_event(runtime);
  const bool ok = accepted && ctx.err == emel::error::cast(error::none);
  if (ok && ev.on_done != nullptr) ev.on_done(events::configured{ev});
  if (!ok && ev.on_error != nullptr) ev.on_error(events::request_error{ctx.err});
  return ok;
}

bool sm::process_event(const event::reset &ev) {
  action::reset_outputs(this->context_);
  event::reset_ctx ctx{};
  event::reset_run runtime{ev, ctx};
  const bool accepted = base_type::process_event(runtime);
  const bool ok = accepted && ctx.err == emel::error::cast(error::none);
  if (ok && ev.on_done != nullptr) ev.on_done(events::reset_done{ev});
  if (!ok && ev.on_error != nullptr) ev.on_error(events::request_error{ctx.err});
  return ok;
}

bool sm::process_event(const event::complete &ev) {
  action::reset_outputs(this->context_);
  event::complete_ctx ctx{};
  event::complete_run runtime{ev, ctx};
  const bool accepted = base_type::process_event(runtime);
  const bool ok = accepted && ctx.err == emel::error::cast(error::none);
  if (ok) {
    ctx.normalized_envelope = normalized_envelope();
    ctx.generated_token_ids = generated_token_ids();
    ctx.prompt_tokens = prompt_tokens();
    ctx.generated_tokens = generated_tokens();
    ctx.prefill_nanoseconds = prefill_nanoseconds();
    ctx.decode_nanoseconds = decode_nanoseconds();
    if (ev.on_done != nullptr)
      ev.on_done(events::completed{
          .request = ev,
          .normalized_envelope = ctx.normalized_envelope,
          .generated_token_ids = ctx.generated_token_ids,
          .prompt_tokens = ctx.prompt_tokens,
          .generated_tokens = ctx.generated_tokens,
          .prefill_nanoseconds = ctx.prefill_nanoseconds,
          .decode_nanoseconds = ctx.decode_nanoseconds,
      });
  } else if (ev.on_error != nullptr) {
    ev.on_error(events::request_error{ctx.err});
  }
  return ok;
}

} // namespace emel::model::needle::request
