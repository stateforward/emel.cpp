#include "emel/model/architecture/detail.hpp"

#include "emel/model/gemma4/detail.hpp"
#include "emel/model/lfm2/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/qwen3/detail.hpp"

namespace emel::model {

const std::array<architecture, 4> default_architectures = {{
    {
        .name = "llama",
        .load_hparams = &emel::model::llama::detail::load_hparams,
        .validate_data = &emel::model::llama::detail::validate_data,
        .build_view = &emel::model::llama::detail::build_view,
    },
    {
        .name = "qwen3",
        .load_hparams = &emel::model::qwen3::detail::load_hparams,
        .validate_data = &emel::model::qwen3::detail::validate_data,
        .build_view = &emel::model::qwen3::detail::build_view,
    },
    {
        .name = "lfm2",
        .load_hparams = &emel::model::lfm2::detail::load_hparams,
        .validate_data = &emel::model::lfm2::detail::validate_execution_contract,
        .build_view = &emel::model::lfm2::detail::build_view,
    },
    {
        .name = "gemma4",
        .load_hparams = &emel::model::gemma4::detail::load_hparams,
        .validate_data = &emel::model::gemma4::detail::validate_execution_contract,
        .build_view = &emel::model::gemma4::detail::build_view,
    },
}};

architectures default_architecture_span() noexcept {
  return architectures{default_architectures.data(), default_architectures.size()};
}

const architecture * resolve_architecture(
    const std::string_view name,
    const architectures available_architectures) noexcept {
  for (const auto & candidate : available_architectures) {
    if (candidate.name == name) {
      return &candidate;
    }
  }

  return nullptr;
}

}  // namespace emel::model
