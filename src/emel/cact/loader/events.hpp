#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/cact/loader/errors.hpp"
#include "emel/callback.hpp"
#include "emel/error/error.hpp"

namespace emel::cact::loader {

// Positional (nameless) tensor view: geometry only, zero-copy pointer into the
// caller-supplied file image. No dequantization or allocation happens here.
struct tensor_view {
  uint8_t dtype = 0u;
  uint8_t ndim = 0u;
  std::array<uint32_t, 4> shape = {0u, 0u, 0u, 0u};
  uint64_t offset = 0u;
  uint64_t nbytes = 0u;
  uint32_t group = 0u;
  uint32_t bits = 0u;
  const uint8_t *data = nullptr;
};

inline constexpr uint32_t k_codebook_len = 28u;

// Fixed architecture geometry decoded from the 120-byte `.cact` header.
struct geometry {
  uint32_t num_tensors = 0u;
  uint32_t kv_window = 0u;
  uint32_t kv_bits = 0u;
  uint32_t vocab_size = 0u;
  uint32_t d_model = 0u;
  uint32_t num_heads = 0u;
  uint32_t num_kv_heads = 0u;
  uint32_t num_layers = 0u;
  uint32_t head_dim = 0u;
  uint32_t max_seq_len = 0u;
  uint32_t hada_n = 0u;
  uint32_t mhc_lanes = 0u;
  uint32_t engram_slots = 0u;
  uint32_t engram_sub_dim = 0u;
  uint32_t num_engram_tables = 0u;
  uint32_t engram_conv_taps = 0u;
  uint32_t engram_conv_dilation = 0u;
  uint32_t num_engram_orders = 0u;
  std::array<uint32_t, 4> engram_orders = {0u, 0u, 0u, 0u};
  uint32_t num_engram_sites = 0u;
  std::array<uint32_t, 4> engram_sites = {0u, 0u, 0u, 0u};
  float rope_theta = 0.0f;
  std::array<float, k_codebook_len> codebook = {};
};

namespace events {

struct probe_done;
struct probe_error;
struct bind_done;
struct bind_error;
struct parse_done;
struct parse_error;

} // namespace events

namespace event {

using probe_done_fn = emel::callback<void(const events::probe_done &)>;
using probe_error_fn = emel::callback<void(const events::probe_error &)>;
using bind_done_fn = emel::callback<void(const events::bind_done &)>;
using bind_error_fn = emel::callback<void(const events::bind_error &)>;
using parse_done_fn = emel::callback<void(const events::parse_done &)>;
using parse_error_fn = emel::callback<void(const events::parse_error &)>;

// Probes header + directory geometry from a zero-copy file image. Does not
// touch io::mmap; the caller is responsible for producing file_image via the
// maintained mmap/file route before invoking this actor.
struct probe {
  std::span<const uint8_t> file_image = {};
  geometry &geometry_out;
  const probe_done_fn &on_done;
  const probe_error_fn &on_error;

  probe(std::span<const uint8_t> file_image_in, geometry &geometry_out_in,
        const probe_done_fn &on_done_in,
        const probe_error_fn &on_error_in) noexcept
      : file_image(file_image_in), geometry_out(geometry_out_in),
        on_done(on_done_in), on_error(on_error_in) {}
};

struct bind_storage {
  std::span<tensor_view> tensors = {};
  const bind_done_fn &on_done;
  const bind_error_fn &on_error;

  bind_storage(std::span<tensor_view> tensors_in,
               const bind_done_fn &on_done_in,
               const bind_error_fn &on_error_in) noexcept
      : tensors(tensors_in), on_done(on_done_in), on_error(on_error_in) {}
};

struct parse {
  std::span<const uint8_t> file_image = {};
  const parse_done_fn &on_done;
  const parse_error_fn &on_error;

  parse(std::span<const uint8_t> file_image_in, const parse_done_fn &on_done_in,
        const parse_error_fn &on_error_in) noexcept
      : file_image(file_image_in), on_done(on_done_in), on_error(on_error_in) {}
};

struct probe_ctx {
  emel::error::type err = emel::error::cast(error::none);
  geometry geometry_out = {};
};

struct probe_runtime {
  const probe &request;
  probe_ctx &ctx;
};

struct bind_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_runtime {
  const bind_storage &request;
  bind_ctx &ctx;
};

struct parse_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct parse_runtime {
  const parse &request;
  parse_ctx &ctx;
};

} // namespace event

namespace events {

struct probe_done {
  const event::probe &request;
  const geometry &geometry_out;
};

struct probe_error {
  const event::probe &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_done {
  const event::bind_storage &request;
};

struct bind_error {
  const event::bind_storage &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct parse_done {
  const event::parse &request;
};

struct parse_error {
  const event::parse &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace events

} // namespace emel::cact::loader
