#pragma once

namespace emel::tokenizer::action {

constexpr auto on_tokenize_requested = [] {};
constexpr auto on_partitioning_special_done = [] {};
constexpr auto on_partitioning_special_error = [] {};
constexpr auto on_selecting_backend_done = [] {};
constexpr auto on_selecting_backend_error = [] {};
constexpr auto on_applying_special_prefix_done = [] {};
constexpr auto on_applying_special_prefix_error = [] {};
constexpr auto on_encoding_fragment_done = [] {};
constexpr auto on_encoding_fragment_error = [] {};
constexpr auto on_applying_special_suffix_done = [] {};
constexpr auto on_applying_special_suffix_error = [] {};
constexpr auto on_finalizing_done = [] {};
constexpr auto on_finalizing_error = [] {};

}  // namespace emel::tokenizer::action
