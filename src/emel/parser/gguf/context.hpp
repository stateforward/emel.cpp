#pragma once

#include <array>
#include <cstdint>
#include <cstdio>

#include "emel/model/data.hpp"

namespace emel::parser::gguf {

inline constexpr uint32_t k_default_alignment = 32;
inline constexpr uint32_t k_max_architecture = 64;

struct context {
  std::FILE * file = nullptr;
  bool owns_file = false;
  uint32_t version = 0;
  uint32_t alignment = k_default_alignment;
  uint64_t data_offset = 0;
  uint64_t data_size = 0;
  uint16_t split_count = 1;
  uint16_t split_no = 0;
  uint16_t split_tensors_count = 0;
  std::array<char, k_max_architecture> architecture = {};
  uint32_t architecture_len = 0;
  uint32_t block_count = 0;
  std::array<char, k_max_architecture> pending_arch = {};
  uint32_t pending_arch_len = 0;
  uint32_t pending_block_count = 0;
  bool pending_block_count_valid = false;
  const void * mapped_data = nullptr;
  uint64_t mapped_size = 0;
  std::array<const void *, emel::model::data::k_max_split_files> mapped_splits = {};
  std::array<uint64_t, emel::model::data::k_max_split_files> mapped_sizes = {};
  uint16_t mapped_count = 0;
  bool has_tokenizer_model = false;
  bool has_tokenizer_pre = false;
  bool has_bos_id = false;
  bool has_eos_id = false;
  bool has_unk_id = false;
  bool has_sep_id = false;
  bool has_pad_id = false;
  bool has_mask_id = false;
  bool has_add_bos = false;
  bool has_add_eos = false;
  bool has_add_sep = false;
  bool has_add_space_prefix = false;
  bool has_remove_extra_whitespaces = false;
  bool has_escape_whitespaces = false;
  bool has_treat_whitespace_as_suffix = false;
  bool has_ignore_merges = false;
};

}  // namespace emel::parser::gguf
