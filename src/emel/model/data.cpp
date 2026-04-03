#include "emel/model/data.hpp"

#include "emel/model/architecture/detail.hpp"
#include "emel/model/gemma4/detail.hpp"
#include "emel/model/lfm2/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model {

std::string_view tensor_name_view(const data & model_data,
                                  const data::tensor_record & tensor) noexcept {
  const size_t begin = static_cast<size_t>(tensor.name_offset);
  const size_t length = static_cast<size_t>(tensor.name_length);
  if (begin + length > model_data.name_storage.size()) {
    return {};
  }

  return std::string_view{model_data.name_storage.data() + begin, length};
}

bool try_parse_block_index(const std::string_view name, int32_t & block_index_out) noexcept {
  constexpr std::string_view k_prefix = "blk.";
  if (!name.starts_with(k_prefix)) {
    return false;
  }

  size_t cursor = k_prefix.size();
  if (cursor >= name.size()) {
    return false;
  }

  int32_t value = 0;
  bool saw_digit = false;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    saw_digit = true;
    value = value * 10 + static_cast<int32_t>(name[cursor] - '0');
    ++cursor;
  }

  if (!saw_digit || cursor >= name.size() || name[cursor] != '.') {
    return false;
  }

  block_index_out = value;
  return true;
}

std::string_view architecture_name_view(const data & model_data) noexcept {
  size_t length = 0u;
  while (length < model_data.architecture_name.size() &&
         model_data.architecture_name[length] != '\0') {
    ++length;
  }

  return std::string_view{model_data.architecture_name.data(), length};
}

bool is_supported_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::resolve_architecture(
             architecture, emel::model::default_architecture_span()) != nullptr;
}

bool is_lfm2_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::lfm2::detail::is_execution_architecture(architecture);
}

bool is_gemma4_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::gemma4::detail::is_execution_architecture(architecture);
}

emel::error::type validate_execution_contract(const data & model_data) noexcept {
  const auto architecture = emel::model::architecture_name_view(model_data);
  const auto * resolved = emel::model::resolve_architecture(
      architecture, emel::model::default_architecture_span());
  if (resolved == nullptr || resolved->validate_data == nullptr) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return resolved->validate_data(model_data);
}

}  // namespace emel::model
