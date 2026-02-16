#pragma once

#include <array>
#include <cstdint>

namespace emel::model {

struct data {
  static constexpr int32_t k_max_tensors = 65536;
  static constexpr int32_t k_max_name_bytes = 4 * 1024 * 1024;
  static constexpr int32_t k_max_architecture_name = 64;

  struct tensor_record {
    uint32_t name_offset = 0;
    uint32_t name_length = 0;
    int32_t type = 0;
    int32_t n_dims = 0;
    std::array<int64_t, 4> dims = {};
    uint64_t data_offset = 0;
    uint64_t data_size = 0;
    const void * data = nullptr;
  };

  int32_t n_layers = 0;
  uint32_t n_tensors = 0;
  uint32_t name_bytes_used = 0;
  std::array<char, k_max_architecture_name> architecture_name = {};
  std::array<char, k_max_name_bytes> name_storage = {};
  std::array<tensor_record, k_max_tensors> tensors = {};
  const void * weights_data = nullptr;
  uint64_t weights_size = 0;
  bool weights_mapped = false;
};

}  // namespace emel::model
