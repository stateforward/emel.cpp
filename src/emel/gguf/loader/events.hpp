#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/model/data.hpp"

namespace emel::gguf::loader {

struct requirements {
  uint32_t tensor_count = 0;
  uint32_t kv_count = 0;
  uint32_t max_key_bytes = 0;
  uint32_t max_value_bytes = 0;
};

struct kv_entry {
  uint32_t key_offset = 0;
  uint32_t key_length = 0;
  uint32_t value_offset = 0;
  uint32_t value_length = 0;
  uint32_t value_type = 0;
};

namespace events {

struct probe_done;
struct probe_error;
struct bind_done;
struct bind_error;
struct parse_done;
struct parse_error;

}  // namespace events

namespace event {

using probe_done_fn = emel::callback<void(const events::probe_done &)>;
using probe_error_fn = emel::callback<void(const events::probe_error &)>;
using bind_done_fn = emel::callback<void(const events::bind_done &)>;
using bind_error_fn = emel::callback<void(const events::bind_error &)>;
using parse_done_fn = emel::callback<void(const events::parse_done &)>;
using parse_error_fn = emel::callback<void(const events::parse_error &)>;

struct probe {
  std::span<const uint8_t> file_image = {};
  requirements & requirements_out;
  const probe_done_fn & on_done;
  const probe_error_fn & on_error;

  probe(std::span<const uint8_t> file_image_in,
        requirements & requirements_out_in,
        const probe_done_fn & on_done_in,
        const probe_error_fn & on_error_in) noexcept
      : file_image(file_image_in),
        requirements_out(requirements_out_in),
        on_done(on_done_in),
        on_error(on_error_in) {}
};

struct bind_storage {
  std::span<uint8_t> kv_arena = {};
  std::span<kv_entry> kv_entries = {};
  std::span<emel::model::data::tensor_record> tensors = {};
  const bind_done_fn & on_done;
  const bind_error_fn & on_error;

  bind_storage(std::span<uint8_t> kv_arena_in,
               std::span<kv_entry> kv_entries_in,
               std::span<emel::model::data::tensor_record> tensors_in,
               const bind_done_fn & on_done_in,
               const bind_error_fn & on_error_in) noexcept
      : kv_arena(kv_arena_in),
        kv_entries(kv_entries_in),
        tensors(tensors_in),
        on_done(on_done_in),
        on_error(on_error_in) {}
};

struct parse {
  std::span<const uint8_t> file_image = {};
  const parse_done_fn & on_done;
  const parse_error_fn & on_error;

  parse(std::span<const uint8_t> file_image_in,
        const parse_done_fn & on_done_in,
        const parse_error_fn & on_error_in) noexcept
      : file_image(file_image_in),
        on_done(on_done_in),
        on_error(on_error_in) {}
};

struct probe_ctx {
  emel::error::type err = emel::error::cast(error::none);
  requirements requirements_out = {};
};

struct probe_runtime {
  const probe & request;
  probe_ctx & ctx;
};

struct bind_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_runtime {
  const bind_storage & request;
  bind_ctx & ctx;
};

struct parse_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct parse_runtime {
  const parse & request;
  parse_ctx & ctx;
};

}  // namespace event

namespace events {

struct probe_done {
  const event::probe & request;
  const requirements & requirements_out;
};

struct probe_error {
  const event::probe & request;
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_done {
  const event::bind_storage & request;
};

struct bind_error {
  const event::bind_storage & request;
  emel::error::type err = emel::error::cast(error::none);
};

struct parse_done {
  const event::parse & request;
};

struct parse_error {
  const event::parse & request;
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace events

}  // namespace emel::gguf::loader
