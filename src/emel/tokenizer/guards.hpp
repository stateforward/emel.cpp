#pragma once

namespace emel::tokenizer::guard {

constexpr auto can_tokenize = [] { return true; };
constexpr auto has_more_fragments = [] { return false; };
constexpr auto no_more_fragments = [] { return !has_more_fragments(); };

}  // namespace emel::tokenizer::guard
