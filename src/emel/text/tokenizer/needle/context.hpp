#pragma once

namespace emel::text::tokenizer::needle::action {

// The blob loader holds no persistent actor state: each load request carries
// its blob span and output vocab, and per-dispatch values cross phases via
// the typed runtime event.
struct context {};

} // namespace emel::text::tokenizer::needle::action
