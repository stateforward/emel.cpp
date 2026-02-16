#pragma once

#include <array>
#include <cstdint>

namespace emel::model {

struct data {
  static constexpr int32_t k_max_tensors = 65536;
  static constexpr int32_t k_max_name_bytes = 4 * 1024 * 1024;
  static constexpr int32_t k_max_architecture_name = 64;
  static constexpr int32_t k_max_vocab_tokens = 320000;
  static constexpr int32_t k_max_vocab_bytes = 8 * 1024 * 1024;
  static constexpr int32_t k_max_tokenizer_model = 64;
  static constexpr int32_t k_max_tokenizer_pre = 64;
  static constexpr int32_t k_max_merges = 400000;
  static constexpr int32_t k_max_merge_bytes = 8 * 1024 * 1024;
  static constexpr int32_t k_max_precompiled_charsmap_bytes = 1024 * 1024;
  static constexpr int32_t k_max_split_files = 128;

  struct tensor_record {
    uint32_t name_offset = 0;
    uint32_t name_length = 0;
    int32_t type = 0;
    int32_t n_dims = 0;
    std::array<int64_t, 4> dims = {};
    uint64_t data_offset = 0;
    uint64_t file_offset = 0;
    uint64_t data_size = 0;
    const void * data = nullptr;
    uint16_t file_index = 0;
  };

  struct hparams {
    int32_t n_ctx = 0;
    int32_t n_embd = 0;
    int32_t n_embd_out = 0;
    int32_t n_ff = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_rot = 0;
    int32_t n_layer = 0;
    int32_t n_vocab = 0;
    float rope_freq_base = 0.0f;
    float rope_scale_linear = 0.0f;
    float rope_scaling_factor = 0.0f;
    float rope_scaling_attn_factor = 0.0f;
    int32_t rope_scaling_orig_ctx_len = 0;
  };

  struct vocab_entry {
    uint32_t text_offset = 0;
    uint32_t text_length = 0;
    float score = 0.0f;
    int32_t type = 0;
  };

  struct vocab {
    uint32_t n_tokens = 0;
    uint32_t n_token_types = 0;
    uint32_t token_bytes_used = 0;
    uint32_t n_merges = 0;
    uint32_t merge_bytes_used = 0;
    uint32_t precompiled_charsmap_size = 0;

    std::array<char, k_max_tokenizer_model> tokenizer_model = {};
    std::array<char, k_max_tokenizer_pre> tokenizer_pre = {};
    std::array<char, k_max_vocab_bytes> token_storage = {};
    std::array<char, k_max_merge_bytes> merge_storage = {};

    std::array<vocab_entry, k_max_vocab_tokens> entries = {};
    std::array<uint32_t, k_max_merges> merge_offsets = {};
    std::array<uint32_t, k_max_merges> merge_lengths = {};
    std::array<uint8_t, k_max_precompiled_charsmap_bytes> precompiled_charsmap = {};

    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t eot_id = -1;
    int32_t eom_id = -1;
    int32_t unk_id = -1;
    int32_t sep_id = -1;
    int32_t pad_id = -1;
    int32_t cls_id = -1;
    int32_t mask_id = -1;
    int32_t prefix_id = -1;
    int32_t suffix_id = -1;
    int32_t middle_id = -1;
    int32_t fim_pre_id = -1;
    int32_t fim_suf_id = -1;
    int32_t fim_mid_id = -1;
    int32_t fim_pad_id = -1;
    int32_t fim_rep_id = -1;
    int32_t fim_sep_id = -1;

    bool add_bos = false;
    bool add_eos = false;
    bool add_sep = false;
    bool add_space_prefix = false;
    bool remove_extra_whitespaces = false;
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
  uint16_t weights_split_count = 1;
  std::array<uint64_t, k_max_split_files> weights_split_sizes = {};
  std::array<uint64_t, k_max_split_files> weights_split_offsets = {};
  hparams params = {};
  vocab vocab_data = {};
};

}  // namespace emel::model
