#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"

namespace emel::embeddings::generator::action {

enum class text_route_kind : uint8_t {
  none = 0u,
  encoder = 1u,
};

enum class image_route_kind : uint8_t {
  none = 0u,
  encoder = 1u,
};

enum class audio_route_kind : uint8_t {
  none = 0u,
  encoder = 1u,
};

struct embedding_execution_contract {
  const emel::model::data * model = nullptr;
  int32_t embedding_length = 0;
  int32_t image_encoder_length = 0;
  int32_t audio_encoder_length = 0;
  uint32_t matryoshka_dimension_count = 0;
  std::array<int32_t, emel::model::data::k_max_matryoshka_dims> matryoshka_dimensions = {};
};

struct text_route_status {
  bool ready = false;
  text_route_kind route_kind = text_route_kind::none;
  int32_t max_positions = 0;
  int32_t shared_embedding_size = 0;
};

struct image_route_status {
  bool ready = false;
  image_route_kind route_kind = image_route_kind::none;
};

struct audio_route_status {
  bool ready = false;
  audio_route_kind route_kind = audio_route_kind::none;
};

struct scratch_status {
  int32_t * token_ids = nullptr;
  float * full_embedding = nullptr;
  bool ready = false;
};

struct route_state {
  using destroy_fn = void (*)(void *) noexcept;

  void * data = nullptr;
  destroy_fn destroy = nullptr;

  route_state() = default;
  route_state(const route_state &) = delete;
  route_state & operator=(const route_state &) = delete;

  route_state(route_state && other) noexcept
      : data(other.data),
        destroy(other.destroy) {
    other.data = nullptr;
    other.destroy = nullptr;
  }

  route_state & operator=(route_state && other) noexcept {
    if (this != &other) {
      reset();
      data = other.data;
      destroy = other.destroy;
      other.data = nullptr;
      other.destroy = nullptr;
    }
    return *this;
  }

  ~route_state() {
    reset();
  }

  void reset(void * next = nullptr, destroy_fn next_destroy = nullptr) noexcept {
    if (data != nullptr && destroy != nullptr) {
      destroy(data);
    }
    data = next;
    destroy = next_destroy;
  }
};

struct context {
  const emel::model::data * model = nullptr;
  emel::text::conditioner::sm * conditioner = nullptr;
  void * formatter_ctx = nullptr;
  emel::text::formatter::format_fn format_prompt =
      emel::text::formatter::format_raw;
  embedding_execution_contract execution_contract = {};

  text_route_status text_status = {};
  image_route_status image_status = {};
  audio_route_status audio_status = {};
  scratch_status scratch_status = {};
  route_state route = {};
  bool initialized = false;
};

}  // namespace emel::embeddings::generator::action
