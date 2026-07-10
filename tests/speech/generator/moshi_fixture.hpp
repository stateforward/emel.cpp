#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/detail.hpp"

namespace emel::speech::generator::moshi::test {

struct loaded_fixture {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};

  emel::model::detail::kv_binding binding() const {
    return emel::model::detail::kv_binding{
        .arena = std::span<const uint8_t>{kv_arena},
        .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
    };
  }
};

inline std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

inline std::filesystem::path fixture_path(const std::string_view name) {
  return repo_root() / "tests" / "models" / std::string{name};
}

inline std::vector<uint8_t>
read_binary_file(const std::filesystem::path &path) {
  std::ifstream file{path, std::ios::binary};
  REQUIRE(file.good());
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  REQUIRE(size > 0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(file.good());
  return bytes;
}

inline void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
inline void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
inline void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
inline void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
inline void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
inline void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

inline void
materialize_tensor_names_from_file(emel::model::data &model,
                                   const std::vector<uint8_t> &file) {
  model.name_bytes_used = 0;
  for (uint32_t index = 0; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    REQUIRE(source_offset + length <= file.size());
    REQUIRE(static_cast<size_t>(model.name_bytes_used) + length <=
            model.name_storage.size());
    std::copy_n(file.data() + source_offset, length,
                model.name_storage.data() + model.name_bytes_used);
    tensor.name_offset = model.name_bytes_used;
    model.name_bytes_used += static_cast<uint32_t>(length);
  }
}

inline loaded_fixture load_fixture_or_skip(const std::string_view name) {
  const auto path = fixture_path(name);
  if (!std::filesystem::exists(path)) {
    MESSAGE("skipping Moshi fixture test because fixture is missing: "
            << path.string());
    return {};
  }

  loaded_fixture loaded{};
  loaded.model = std::make_unique<emel::model::data>();
  loaded.file_bytes = read_binary_file(path);

  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements{};
  REQUIRE(loader.process_event(emel::gguf::loader::event::probe{
      std::span<const uint8_t>{loaded.file_bytes},
      requirements,
      emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>(),
      emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>(),
  }));

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  REQUIRE(arena_bytes != UINT64_MAX);
  loaded.kv_arena.resize(static_cast<size_t>(arena_bytes));
  loaded.kv_entries.resize(requirements.kv_count);
  loaded.model->n_tensors = requirements.tensor_count;

  REQUIRE(loader.process_event(emel::gguf::loader::event::bind_storage{
      std::span<uint8_t>{loaded.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{loaded.kv_entries},
      std::span<emel::model::data::tensor_record>{loaded.model->tensors.data(),
                                                  loaded.model->n_tensors},
      emel::gguf::loader::event::bind_done_fn::from<&noop_bind_done>(),
      emel::gguf::loader::event::bind_error_fn::from<&noop_bind_error>(),
  }));
  REQUIRE(loader.process_event(emel::gguf::loader::event::parse{
      std::span<const uint8_t>{loaded.file_bytes},
      emel::gguf::loader::event::parse_done_fn::from<&noop_parse_done>(),
      emel::gguf::loader::event::parse_error_fn::from<&noop_parse_error>(),
  }));
  REQUIRE(emel::model::detail::load_hparams_from_gguf(loaded.binding(),
                                                      *loaded.model));
  loaded.model->weights_data = loaded.file_bytes.data();
  loaded.model->weights_size = loaded.file_bytes.size();
  materialize_tensor_names_from_file(*loaded.model, loaded.file_bytes);
  return loaded;
}

} // namespace emel::speech::generator::moshi::test
