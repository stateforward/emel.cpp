#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "doctest/doctest.h"
#include "emel/parser/gguf/actions.hpp"

namespace {

bool make_temp_path(char * out, const size_t capacity) {
  if (out == nullptr || capacity == 0) {
    return false;
  }
  const char * tmp = nullptr;
#if defined(_WIN32)
  tmp = std::getenv("TEMP");
  if (tmp == nullptr || tmp[0] == '\0') {
    tmp = ".";
  }
  const int pid = _getpid();
#else
  tmp = std::getenv("TMPDIR");
  if (tmp == nullptr || tmp[0] == '\0') {
    tmp = "/tmp";
  }
  const int pid = getpid();
#endif
  static uint64_t counter = 0;
  const uint64_t stamp = ++counter;
#if defined(_WIN32)
  const int written =
    std::snprintf(out, capacity, "%s\\emel_gguf_%d_%llu.gguf", tmp, pid,
                  static_cast<unsigned long long>(stamp));
#else
  const int written =
    std::snprintf(out, capacity, "%s/emel_gguf_%d_%llu.gguf", tmp, pid,
                  static_cast<unsigned long long>(stamp));
#endif
  return written > 0 && static_cast<size_t>(written) < capacity;
}

bool write_u64(std::FILE * file, const uint64_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_u32(std::FILE * file, const uint32_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_i32(std::FILE * file, const int32_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_string(std::FILE * file, const char * value) {
  const uint64_t len = std::strlen(value);
  if (!write_u64(file, len)) {
    return false;
  }
  return std::fwrite(value, 1, len, file) == len;
}

bool write_header(std::FILE * file, const uint32_t version, const int64_t n_tensors, const int64_t n_kv) {
  const char magic[4] = {'G', 'G', 'U', 'F'};
  if (std::fwrite(magic, 1, sizeof(magic), file) != sizeof(magic)) {
    return false;
  }
  if (!write_u32(file, version)) {
    return false;
  }
  if (std::fwrite(&n_tensors, 1, sizeof(n_tensors), file) != sizeof(n_tensors)) {
    return false;
  }
  if (std::fwrite(&n_kv, 1, sizeof(n_kv), file) != sizeof(n_kv)) {
    return false;
  }
  return true;
}

bool write_kv_bool(std::FILE * file, const char * key, const bool value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_bool);
  if (!write_i32(file, type)) {
    return false;
  }
  const uint8_t raw = value ? 1 : 0;
  return std::fwrite(&raw, 1, sizeof(raw), file) == sizeof(raw);
}

bool write_kv_i32(std::FILE * file, const char * key, const int32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_i32);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_i32(file, value);
}

bool write_kv_u32(std::FILE * file, const char * key, const uint32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_u32);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_u32(file, value);
}

bool write_kv_f32(std::FILE * file, const char * key, const float value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_f32);
  if (!write_i32(file, type)) {
    return false;
  }
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_kv_string(std::FILE * file, const char * key, const char * value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_string);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_string(file, value);
}

bool write_kv_string_array(std::FILE * file, const char * key,
                           const std::array<const char *, 2> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_string);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (const char * value : values) {
    if (!write_string(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_string_array3(std::FILE * file, const char * key,
                            const std::array<const char *, 3> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_string);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (const char * value : values) {
    if (!write_string(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_u32_array(std::FILE * file, const char * key,
                        const std::array<uint32_t, 3> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_u32);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (uint32_t value : values) {
    if (!write_u32(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_i32_array(std::FILE * file, const char * key,
                        const std::array<int32_t, 2> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_i32);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (int32_t value : values) {
    if (!write_i32(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_f32_array(std::FILE * file, const char * key,
                        const std::array<float, 3> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_f32);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (float value : values) {
    if (std::fwrite(&value, 1, sizeof(value), file) != sizeof(value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_f32_array2(std::FILE * file, const char * key,
                         const std::array<float, 2> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_f32);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (float value : values) {
    if (std::fwrite(&value, 1, sizeof(value), file) != sizeof(value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_bool_array(std::FILE * file, const char * key,
                         const std::array<uint8_t, 3> & values) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_bool);
  if (!write_i32(file, elem_type)) {
    return false;
  }
  if (!write_u64(file, values.size())) {
    return false;
  }
  for (uint8_t value : values) {
    if (std::fwrite(&value, 1, sizeof(value), file) != sizeof(value)) {
      return false;
    }
  }
  return true;
}

bool write_tensor_info(std::FILE * file, const char * name, const int32_t type,
                       const std::array<int64_t, 4> & dims, const uint64_t offset) {
  if (!write_string(file, name)) {
    return false;
  }
  const uint32_t n_dims = 2;
  if (!write_u32(file, n_dims)) {
    return false;
  }
  if (std::fwrite(&dims[0], 1, sizeof(int64_t), file) != sizeof(int64_t)) {
    return false;
  }
  if (std::fwrite(&dims[1], 1, sizeof(int64_t), file) != sizeof(int64_t)) {
    return false;
  }
  if (!write_i32(file, type)) {
    return false;
  }
  return write_u64(file, offset);
}

bool write_vocab_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 29;
  if (!write_header(file, emel::parser::gguf::k_gguf_version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_bool(file, "tokenizer.ggml.add_bos_token", true) ||
      !write_kv_bool(file, "tokenizer.ggml.add_eos_token", false) ||
      !write_kv_bool(file, "tokenizer.ggml.add_sep_token", true) ||
      !write_kv_bool(file, "tokenizer.ggml.add_space_prefix", true) ||
      !write_kv_bool(file, "tokenizer.ggml.remove_extra_whitespaces", true) ||
      !write_kv_i32(file, "tokenizer.ggml.padding_token_id", 11) ||
      !write_kv_i32(file, "tokenizer.ggml.cls_token_id", 12) ||
      !write_kv_i32(file, "tokenizer.ggml.mask_token_id", 13) ||
      !write_kv_i32(file, "tokenizer.ggml.prefix_token_id", 14) ||
      !write_kv_i32(file, "tokenizer.ggml.suffix_token_id", 15) ||
      !write_kv_i32(file, "tokenizer.ggml.middle_token_id", 16) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_pre_token_id", 17) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_suf_token_id", 18) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_mid_token_id", 19) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_pad_token_id", 20) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_rep_token_id", 21) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_sep_token_id", 22) ||
      !write_kv_u32(file, "test.context_length", 1024) ||
      !write_kv_u32(file, "test.embedding_length", 512) ||
      !write_kv_u32(file, "test.embedding_length_out", 256) ||
      !write_kv_u32(file, "test.feed_forward_length", 2048) ||
      !write_kv_u32(file, "test.attention.head_count", 8) ||
      !write_kv_u32(file, "test.attention.head_count_kv", 4) ||
      !write_kv_u32(file, "test.rope.dimension_count", 64) ||
      !write_kv_f32(file, "test.rope.freq_base", 20000.0f) ||
      !write_kv_f32(file, "test.rope.scale_linear", 1.5f) ||
      !write_kv_f32(file, "test.rope.scaling.factor", 2.0f) ||
      !write_kv_f32(file, "test.rope.scaling.attn_factor", 1.25f) ||
      !write_kv_u32(file, "test.rope.scaling.original_context_length", 2048)) {
    std::fclose(file);
    return false;
  }

  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

std::string_view meta_view(const emel::model::data::metadata & meta,
                           const emel::model::data::metadata::string_view & view) {
  if (view.length == 0) {
    return {};
  }
  return std::string_view(meta.blob.data() + view.offset, view.length);
}

bool write_metadata_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  constexpr int64_t n_tensors = 1;
  constexpr int64_t n_kv = 222;
  if (!write_header(file, emel::parser::gguf::k_gguf_version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, "general.architecture", "test") ||
      !write_kv_string(file, "general.type", "model") ||
      !write_kv_u32(file, "general.quantization_version", 2) ||
      !write_kv_u32(file, "general.file_type", 1) ||
      !write_kv_string(file, "general.sampling.sequence", "top_k,top_p") ||
      !write_kv_i32(file, "general.sampling.top_k", 50) ||
      !write_kv_f32(file, "general.sampling.top_p", 0.9f) ||
      !write_kv_f32(file, "general.sampling.min_p", 0.05f) ||
      !write_kv_f32(file, "general.sampling.xtc_probability", 0.4f) ||
      !write_kv_f32(file, "general.sampling.xtc_threshold", 0.2f) ||
      !write_kv_f32(file, "general.sampling.temp", 0.7f) ||
      !write_kv_i32(file, "general.sampling.penalty_last_n", 64) ||
      !write_kv_f32(file, "general.sampling.penalty_repeat", 1.1f) ||
      !write_kv_i32(file, "general.sampling.mirostat", 2) ||
      !write_kv_f32(file, "general.sampling.mirostat_tau", 5.0f) ||
      !write_kv_f32(file, "general.sampling.mirostat_eta", 0.1f) ||
      !write_kv_string(file, "general.name", "test-model") ||
      !write_kv_string(file, "general.author", "author") ||
      !write_kv_string(file, "general.version", "v1") ||
      !write_kv_string(file, "general.organization", "org") ||
      !write_kv_string(file, "general.finetune", "ft") ||
      !write_kv_string(file, "general.basename", "base") ||
      !write_kv_string(file, "general.description", "desc") ||
      !write_kv_string(file, "general.quantized_by", "quant") ||
      !write_kv_string(file, "general.size_label", "size") ||
      !write_kv_string(file, "general.license", "MIT") ||
      !write_kv_string(file, "general.license.name", "MIT") ||
      !write_kv_string(file, "general.license.link", "https://license") ||
      !write_kv_string(file, "general.url", "https://model") ||
      !write_kv_string(file, "general.doi", "10.1234/xyz") ||
      !write_kv_string(file, "general.uuid", "uuid") ||
      !write_kv_string(file, "general.repo_url", "https://repo") ||
      !write_kv_string(file, "general.source.url", "https://source") ||
      !write_kv_string(file, "general.source.doi", "10.0000/src") ||
      !write_kv_string(file, "general.source.uuid", "srcuuid") ||
      !write_kv_string(file, "general.source.repo_url", "https://src-repo") ||
      !write_kv_string(file, "general.source.huggingface.repository", "hf/repo") ||
      !write_kv_u32(file, "general.base_model.count", 1) ||
      !write_kv_string(file, "general.base_model.0.name", "base0") ||
      !write_kv_string(file, "general.base_model.0.author", "bauthor") ||
      !write_kv_string(file, "general.base_model.0.version", "bver") ||
      !write_kv_string(file, "general.base_model.0.organization", "borg") ||
      !write_kv_string(file, "general.base_model.0.description", "bdesc") ||
      !write_kv_string(file, "general.base_model.0.url", "https://b") ||
      !write_kv_string(file, "general.base_model.0.doi", "10.0000/b") ||
      !write_kv_string(file, "general.base_model.0.uuid", "buuid") ||
      !write_kv_string(file, "general.base_model.0.repo_url", "https://b-repo") ||
      !write_kv_u32(file, "general.dataset.count", 1) ||
      !write_kv_string(file, "general.dataset.0.name", "data0") ||
      !write_kv_string(file, "general.dataset.0.author", "dauthor") ||
      !write_kv_string(file, "general.dataset.0.version", "dver") ||
      !write_kv_string(file, "general.dataset.0.organization", "dorg") ||
      !write_kv_string(file, "general.dataset.0.description", "ddesc") ||
      !write_kv_string(file, "general.dataset.0.url", "https://d") ||
      !write_kv_string(file, "general.dataset.0.doi", "10.0000/d") ||
      !write_kv_string(file, "general.dataset.0.uuid", "duuid") ||
      !write_kv_string(file, "general.dataset.0.repo_url", "https://d-repo") ||
      !write_kv_string_array(
        file, "general.tags", std::array<const char *, 2>{"tag1", "tag2"}) ||
      !write_kv_string_array(
        file, "general.languages", std::array<const char *, 2>{"en", "fr"}) ||
      !write_kv_string(file, "tokenizer.huggingface.json", "{\"a\":1}") ||
      !write_kv_string(file, "tokenizer.rwkv.world", "rwkv") ||
      !write_kv_string(file, "tokenizer.chat_template", "tmpl") ||
      !write_kv_string_array3(
        file, "tokenizer.chat_templates",
        std::array<const char *, 3>{"default", "alt", "fast"}) ||
      !write_kv_string(file, "tokenizer.chat_template.default", "dflt") ||
      !write_kv_string(file, "adapter.type", "lora") ||
      !write_kv_f32(file, "adapter.lora.alpha", 8.0f) ||
      !write_kv_string(file, "adapter.lora.task_name", "task") ||
      !write_kv_string(file, "adapter.lora.prompt_prefix", "prefix") ||
      !write_kv_u32_array(
        file, "adapter.alora.invocation_tokens", std::array<uint32_t, 3>{3, 4, 5}) ||
      !write_kv_u32(file, "imatrix.chunk_count", 2) ||
      !write_kv_u32(file, "imatrix.chunk_size", 128) ||
      !write_kv_string_array(
        file, "imatrix.datasets", std::array<const char *, 2>{"ds0", "ds1"}) ||
      !write_kv_bool(file, "clip.has_vision_encoder", true) ||
      !write_kv_bool(file, "clip.has_audio_encoder", true) ||
      !write_kv_bool(file, "clip.has_llava_projector", false) ||
      !write_kv_string(file, "clip.projector_type", "proj") ||
      !write_kv_bool(file, "clip.use_gelu", true) ||
      !write_kv_bool(file, "clip.use_silu", false) ||
      !write_kv_string(file, "clip.vision.projector_type", "vproj") ||
      !write_kv_u32(file, "clip.vision.image_size", 224) ||
      !write_kv_u32(file, "clip.vision.image_min_pixels", 16) ||
      !write_kv_u32(file, "clip.vision.image_max_pixels", 4096) ||
      !write_kv_u32(file, "clip.vision.preproc_image_size", 256) ||
      !write_kv_u32(file, "clip.vision.patch_size", 14) ||
      !write_kv_u32(file, "clip.vision.embedding_length", 768) ||
      !write_kv_u32(file, "clip.vision.feed_forward_length", 3072) ||
      !write_kv_u32(file, "clip.vision.projection_dim", 512) ||
      !write_kv_u32(file, "clip.vision.block_count", 12) ||
      !write_kv_u32(file, "clip.vision.spatial_merge_size", 2) ||
      !write_kv_u32(file, "clip.vision.n_wa_pattern", 4) ||
      !write_kv_u32(file, "clip.vision.window_size", 7) ||
      !write_kv_u32(file, "clip.vision.attention.head_count", 8) ||
      !write_kv_f32(file, "clip.vision.attention.layer_norm_epsilon", 1e-5f) ||
      !write_kv_u32(file, "clip.vision.projector.scale_factor", 2) ||
      !write_kv_f32_array(
        file, "clip.vision.image_mean", std::array<float, 3>{0.5f, 0.5f, 0.5f}) ||
      !write_kv_f32_array(
        file, "clip.vision.image_std", std::array<float, 3>{0.2f, 0.2f, 0.2f}) ||
      !write_kv_u32_array(
        file, "clip.vision.wa_layer_indexes", std::array<uint32_t, 3>{1, 3, 5}) ||
      !write_kv_bool_array(
        file, "clip.vision.is_deepstack_layers", std::array<uint8_t, 3>{1, 0, 1}) ||
      !write_kv_string(file, "clip.audio.projector_type", "aproj") ||
      !write_kv_u32(file, "clip.audio.num_mel_bins", 80) ||
      !write_kv_u32(file, "clip.audio.embedding_length", 256) ||
      !write_kv_u32(file, "clip.audio.feed_forward_length", 1024) ||
      !write_kv_u32(file, "clip.audio.projection_dim", 128) ||
      !write_kv_u32(file, "clip.audio.block_count", 4) ||
      !write_kv_u32(file, "clip.audio.attention.head_count", 4) ||
      !write_kv_f32(file, "clip.audio.attention.layer_norm_epsilon", 1e-6f) ||
      !write_kv_u32(file, "clip.audio.projector.stack_factor", 2) ||
      !write_kv_bool(file, "diffusion.shift_logits", true) ||
      !write_kv_f32_array2(
        file, "xielu.alpha_p", std::array<float, 2>{0.1f, 0.2f}) ||
      !write_kv_f32_array2(
        file, "xielu.alpha_n", std::array<float, 2>{0.3f, 0.4f}) ||
      !write_kv_f32_array2(
        file, "xielu.beta", std::array<float, 2>{0.5f, 0.6f}) ||
      !write_kv_f32_array2(
        file, "xielu.eps", std::array<float, 2>{0.7f, 0.8f}) ||
      !write_kv_u32(file, "test.vocab_size", 32000) ||
      !write_kv_u32(file, "test.context_length", 4096) ||
      !write_kv_u32(file, "test.embedding_length", 4096) ||
      !write_kv_u32(file, "test.embedding_length_out", 2048) ||
      !write_kv_u32(file, "test.features_length", 128) ||
      !write_kv_u32(file, "test.feed_forward_length", 8192) ||
      !write_kv_u32(file, "test.leading_dense_block_count", 2) ||
      !write_kv_u32(file, "test.expert_feed_forward_length", 1024) ||
      !write_kv_u32(file, "test.expert_shared_feed_forward_length", 512) ||
      !write_kv_u32(file, "test.expert_chunk_feed_forward_length", 256) ||
      !write_kv_bool(file, "test.use_parallel_residual", true) ||
      !write_kv_string(file, "test.tensor_data_layout", "layout") ||
      !write_kv_u32(file, "test.expert_count", 8) ||
      !write_kv_u32(file, "test.expert_used_count", 2) ||
      !write_kv_u32(file, "test.expert_shared_count", 1) ||
      !write_kv_u32(file, "test.expert_group_count", 4) ||
      !write_kv_u32(file, "test.expert_group_used_count", 2) ||
      !write_kv_f32(file, "test.expert_weights_scale", 0.7f) ||
      !write_kv_bool(file, "test.expert_weights_norm", true) ||
      !write_kv_u32(file, "test.expert_gating_func", 3) ||
      !write_kv_f32(file, "test.expert_group_scale", 0.5f) ||
      !write_kv_u32(file, "test.experts_per_group", 2) ||
      !write_kv_u32(file, "test.moe_every_n_layers", 4) ||
      !write_kv_u32(file, "test.nextn_predict_layers", 3) ||
      !write_kv_u32(file, "test.n_deepstack_layers", 2) ||
      !write_kv_u32(file, "test.pooling_type", 1) ||
      !write_kv_f32(file, "test.logit_scale", 1.3f) ||
      !write_kv_u32(file, "test.decoder_start_token_id", 2) ||
      !write_kv_u32(file, "test.decoder_block_count", 6) ||
      !write_kv_f32(file, "test.attn_logit_softcapping", 0.9f) ||
      !write_kv_f32(file, "test.router_logit_softcapping", 1.1f) ||
      !write_kv_f32(file, "test.final_logit_softcapping", 1.2f) ||
      !write_kv_bool(file, "test.swin_norm", true) ||
      !write_kv_u32(file, "test.rescale_every_n_layers", 8) ||
      !write_kv_u32(file, "test.time_mix_extra_dim", 16) ||
      !write_kv_u32(file, "test.time_decay_extra_dim", 32) ||
      !write_kv_f32(file, "test.residual_scale", 0.8f) ||
      !write_kv_f32(file, "test.embedding_scale", 0.9f) ||
      !write_kv_u32(file, "test.token_shift_count", 4) ||
      !write_kv_u32(file, "test.interleave_moe_layer_step", 3) ||
      !write_kv_u32(file, "test.full_attention_interval", 2) ||
      !write_kv_f32(file, "test.activation_sparsity_scale", 0.6f) ||
      !write_kv_u32(file, "test.altup.active_idx", 1) ||
      !write_kv_u32(file, "test.altup.num_inputs", 2) ||
      !write_kv_u32(file, "test.embedding_length_per_layer_input", 64) ||
      !write_kv_u32(file, "test.dense_2_feat_in", 128) ||
      !write_kv_u32(file, "test.dense_2_feat_out", 256) ||
      !write_kv_u32(file, "test.dense_3_feat_in", 512) ||
      !write_kv_u32(file, "test.dense_3_feat_out", 1024) ||
      !write_kv_u32(file, "test.attention.head_count", 12) ||
      !write_kv_u32(file, "test.attention.head_count_kv", 6) ||
      !write_kv_f32(file, "test.attention.max_alibi_bias", 1.4f) ||
      !write_kv_f32(file, "test.attention.clamp_kqv", 0.8f) ||
      !write_kv_u32(file, "test.attention.key_length", 64) ||
      !write_kv_u32(file, "test.attention.value_length", 64) ||
      !write_kv_f32(file, "test.attention.layer_norm_epsilon", 1e-5f) ||
      !write_kv_f32(file, "test.attention.layer_norm_rms_epsilon", 1e-6f) ||
      !write_kv_f32(file, "test.attention.group_norm_epsilon", 1e-4f) ||
      !write_kv_u32(file, "test.attention.group_norm_groups", 2) ||
      !write_kv_bool(file, "test.attention.causal", true) ||
      !write_kv_u32(file, "test.attention.q_lora_rank", 8) ||
      !write_kv_u32(file, "test.attention.kv_lora_rank", 4) ||
      !write_kv_u32(file, "test.attention.decay_lora_rank", 2) ||
      !write_kv_u32(file, "test.attention.iclr_lora_rank", 1) ||
      !write_kv_u32(file, "test.attention.value_residual_mix_lora_rank", 3) ||
      !write_kv_u32(file, "test.attention.gate_lora_rank", 5) ||
      !write_kv_u32(file, "test.attention.relative_buckets_count", 32) ||
      !write_kv_u32(file, "test.attention.sliding_window", 128) ||
      !write_kv_bool_array(
        file, "test.attention.sliding_window_pattern", std::array<uint8_t, 3>{1, 0, 1}) ||
      !write_kv_f32(file, "test.attention.scale", 1.5f) ||
      !write_kv_f32(file, "test.attention.output_scale", 1.1f) ||
      !write_kv_u32(file, "test.attention.temperature_length", 16) ||
      !write_kv_f32(file, "test.attention.temperature_scale", 0.9f) ||
      !write_kv_u32(file, "test.attention.key_length_mla", 32) ||
      !write_kv_u32(file, "test.attention.value_length_mla", 32) ||
      !write_kv_u32(file, "test.attention.indexer.head_count", 2) ||
      !write_kv_u32(file, "test.attention.indexer.key_length", 16) ||
      !write_kv_u32(file, "test.attention.indexer.top_k", 4) ||
      !write_kv_u32(file, "test.attention.shared_kv_layers", 1) ||
      !write_kv_u32(file, "test.rope.dimension_count", 128) ||
      !write_kv_i32_array(
        file, "test.rope.dimension_sections", std::array<int32_t, 2>{64, 64}) ||
      !write_kv_f32(file, "test.rope.freq_base", 5000.0f) ||
      !write_kv_f32(file, "test.rope.freq_base_swa", 6000.0f) ||
      !write_kv_f32(file, "test.rope.scale_linear", 1.2f) ||
      !write_kv_string(file, "test.rope.scaling.type", "linear") ||
      !write_kv_f32(file, "test.rope.scaling.factor", 2.2f) ||
      !write_kv_f32(file, "test.rope.scaling.attn_factor", 1.4f) ||
      !write_kv_u32(file, "test.rope.scaling.original_context_length", 1024) ||
      !write_kv_bool(file, "test.rope.scaling.finetuned", true) ||
      !write_kv_f32(file, "test.rope.scaling.yarn_log_multiplier", 0.2f) ||
      !write_kv_f32(file, "test.rope.scaling.yarn_ext_factor", 0.3f) ||
      !write_kv_f32(file, "test.rope.scaling.yarn_attn_factor", 0.4f) ||
      !write_kv_f32(file, "test.rope.scaling.yarn_beta_fast", 0.5f) ||
      !write_kv_f32(file, "test.rope.scaling.yarn_beta_slow", 0.6f) ||
      !write_kv_u32(file, "test.ssm.conv_kernel", 7) ||
      !write_kv_u32(file, "test.ssm.inner_size", 8) ||
      !write_kv_u32(file, "test.ssm.state_size", 9) ||
      !write_kv_u32(file, "test.ssm.time_step_rank", 10) ||
      !write_kv_u32(file, "test.ssm.group_count", 11) ||
      !write_kv_bool(file, "test.ssm.dt_b_c_rms", true) ||
      !write_kv_u32(file, "test.kda.head_dim", 32) ||
      !write_kv_u32(file, "test.wkv.head_size", 64) ||
      !write_kv_u32(file, "test.posnet.embedding_length", 48) ||
      !write_kv_u32(file, "test.posnet.block_count", 2) ||
      !write_kv_u32(file, "test.convnext.embedding_length", 96) ||
      !write_kv_u32(file, "test.convnext.block_count", 3) ||
      !write_kv_u32(file, "test.shortconv.l_cache", 4) ||
      !write_kv_f32_array(
        file, "test.swiglu_clamp_exp", std::array<float, 3>{0.1f, 0.2f, 0.3f}) ||
      !write_kv_f32_array(
        file, "test.swiglu_clamp_shexp", std::array<float, 3>{0.4f, 0.5f, 0.6f}) ||
      !write_kv_string_array(
        file, "test.classifier.output_labels", std::array<const char *, 2>{"cat", "dog"})) {
    std::fclose(file);
    return false;
  }

  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

}  // namespace

TEST_CASE("gguf parser reads tokenizer flags and ids") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_vocab_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(model.vocab_data.add_bos);
  CHECK(!model.vocab_data.add_eos);
  CHECK(model.vocab_data.add_sep);
  CHECK(model.vocab_data.add_space_prefix);
  CHECK(model.vocab_data.remove_extra_whitespaces);
  CHECK(model.vocab_data.pad_id == 11);
  CHECK(model.vocab_data.cls_id == 12);
  CHECK(model.vocab_data.mask_id == 13);
  CHECK(model.vocab_data.prefix_id == 14);
  CHECK(model.vocab_data.suffix_id == 15);
  CHECK(model.vocab_data.middle_id == 16);
  CHECK(model.vocab_data.fim_pre_id == 17);
  CHECK(model.vocab_data.fim_suf_id == 18);
  CHECK(model.vocab_data.fim_mid_id == 19);
  CHECK(model.vocab_data.fim_pad_id == 20);
  CHECK(model.vocab_data.fim_rep_id == 21);
  CHECK(model.vocab_data.fim_sep_id == 22);
  CHECK(model.params.n_ctx == 1024);
  CHECK(model.params.n_embd == 512);
  CHECK(model.params.n_embd_out == 256);
  CHECK(model.params.n_ff == 2048);
  CHECK(model.params.n_head == 8);
  CHECK(model.params.n_head_kv == 4);
  CHECK(model.params.n_rot == 64);
  CHECK(model.params.rope_freq_base == doctest::Approx(20000.0f));
  CHECK(model.params.rope_scale_linear == doctest::Approx(1.5f));
  CHECK(model.params.rope_scaling_factor == doctest::Approx(2.0f));
  CHECK(model.params.rope_scaling_attn_factor == doctest::Approx(1.25f));
  CHECK(model.params.rope_scaling_orig_ctx_len == 2048);

  std::remove(path);
}

TEST_CASE("gguf parser reads extended metadata") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_metadata_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(meta_view(model.meta, model.meta.general_data.type) == "model");
  CHECK(model.meta.general_data.quantization_version == 2);
  CHECK(model.meta.general_data.file_type == 1);
  CHECK(meta_view(model.meta, model.meta.sampling_data.sequence) == "top_k,top_p");
  CHECK(model.meta.sampling_data.top_k == 50);
  CHECK(model.meta.sampling_data.top_p == doctest::Approx(0.9f));
  CHECK(model.meta.sampling_data.min_p == doctest::Approx(0.05f));
  CHECK(model.meta.sampling_data.xtc_probability == doctest::Approx(0.4f));
  CHECK(model.meta.sampling_data.xtc_threshold == doctest::Approx(0.2f));
  CHECK(model.meta.sampling_data.temp == doctest::Approx(0.7f));
  CHECK(model.meta.sampling_data.penalty_last_n == 64);
  CHECK(model.meta.sampling_data.penalty_repeat == doctest::Approx(1.1f));
  CHECK(model.meta.sampling_data.mirostat == 2);
  CHECK(model.meta.sampling_data.mirostat_tau == doctest::Approx(5.0f));
  CHECK(model.meta.sampling_data.mirostat_eta == doctest::Approx(0.1f));
  CHECK(meta_view(model.meta, model.meta.general_data.name) == "test-model");
  CHECK(meta_view(model.meta, model.meta.general_data.author) == "author");
  CHECK(meta_view(model.meta, model.meta.general_data.version) == "v1");
  CHECK(meta_view(model.meta, model.meta.general_data.organization) == "org");
  CHECK(meta_view(model.meta, model.meta.general_data.finetune) == "ft");
  CHECK(meta_view(model.meta, model.meta.general_data.basename) == "base");
  CHECK(meta_view(model.meta, model.meta.general_data.description) == "desc");
  CHECK(meta_view(model.meta, model.meta.general_data.quantized_by) == "quant");
  CHECK(meta_view(model.meta, model.meta.general_data.size_label) == "size");
  CHECK(meta_view(model.meta, model.meta.general_data.license) == "MIT");
  CHECK(meta_view(model.meta, model.meta.general_data.license_name) == "MIT");
  CHECK(meta_view(model.meta, model.meta.general_data.license_link) == "https://license");
  CHECK(meta_view(model.meta, model.meta.general_data.url) == "https://model");
  CHECK(meta_view(model.meta, model.meta.general_data.doi) == "10.1234/xyz");
  CHECK(meta_view(model.meta, model.meta.general_data.uuid) == "uuid");
  CHECK(meta_view(model.meta, model.meta.general_data.repo_url) == "https://repo");
  CHECK(meta_view(model.meta, model.meta.general_data.source_url) == "https://source");
  CHECK(meta_view(model.meta, model.meta.general_data.source_doi) == "10.0000/src");
  CHECK(meta_view(model.meta, model.meta.general_data.source_uuid) == "srcuuid");
  CHECK(meta_view(model.meta, model.meta.general_data.source_repo_url) == "https://src-repo");
  CHECK(meta_view(model.meta, model.meta.general_data.source_hf_repo) == "hf/repo");
  CHECK(model.meta.general_data.base_model_count == 1);
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].name) == "base0");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].author) == "bauthor");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].version) == "bver");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].organization) == "borg");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].description) == "bdesc");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].url) == "https://b");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].doi) == "10.0000/b");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].uuid) == "buuid");
  CHECK(meta_view(model.meta, model.meta.general_data.base_models[0].repo_url) ==
        "https://b-repo");
  CHECK(model.meta.general_data.dataset_count == 1);
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].name) == "data0");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].author) == "dauthor");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].version) == "dver");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].organization) == "dorg");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].description) == "ddesc");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].url) == "https://d");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].doi) == "10.0000/d");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].uuid) == "duuid");
  CHECK(meta_view(model.meta, model.meta.general_data.datasets[0].repo_url) ==
        "https://d-repo");
  CHECK(meta_view(model.meta, model.meta.general_data.tags[0]) == "tag1");
  CHECK(meta_view(model.meta, model.meta.general_data.languages[1]) == "fr");
  CHECK(meta_view(model.meta, model.meta.tokenizer_data.hf_json) == "{\"a\":1}");
  CHECK(meta_view(model.meta, model.meta.tokenizer_data.rwkv_world) == "rwkv");
  CHECK(meta_view(model.meta, model.meta.tokenizer_data.chat_template) == "tmpl");
  CHECK(meta_view(model.meta, model.meta.tokenizer_data.chat_template_names[0]) ==
        "default");
  CHECK(meta_view(model.meta, model.meta.tokenizer_data.chat_template_values[0]) ==
        "dflt");
  CHECK(meta_view(model.meta, model.meta.adapter_data.type) == "lora");
  CHECK(model.meta.adapter_data.lora_alpha == doctest::Approx(8.0f));
  CHECK(meta_view(model.meta, model.meta.adapter_data.lora_task_name) == "task");
  CHECK(meta_view(model.meta, model.meta.adapter_data.lora_prompt_prefix) == "prefix");
  CHECK(model.meta.adapter_data.alora_invocation_count == 3);
  CHECK(model.meta.adapter_data.alora_invocation_tokens[1] == 4);
  CHECK(model.meta.imatrix_data.chunk_count == 2);
  CHECK(model.meta.imatrix_data.chunk_size == 128);
  CHECK(meta_view(model.meta, model.meta.imatrix_data.datasets[0]) == "ds0");
  CHECK(model.meta.clip_data.has_vision_encoder);
  CHECK(model.meta.clip_data.has_audio_encoder);
  CHECK(!model.meta.clip_data.has_llava_projector);
  CHECK(meta_view(model.meta, model.meta.clip_data.projector_type) == "proj");
  CHECK(model.meta.clip_data.use_gelu);
  CHECK(!model.meta.clip_data.use_silu);
  CHECK(meta_view(model.meta, model.meta.clip_vision_data.projector_type) == "vproj");
  CHECK(model.meta.clip_vision_data.image_size == 224);
  CHECK(model.meta.clip_vision_data.image_min_pixels == 16);
  CHECK(model.meta.clip_vision_data.image_max_pixels == 4096);
  CHECK(model.meta.clip_vision_data.preproc_image_size == 256);
  CHECK(model.meta.clip_vision_data.patch_size == 14);
  CHECK(model.meta.clip_vision_data.embedding_length == 768);
  CHECK(model.meta.clip_vision_data.feed_forward_length == 3072);
  CHECK(model.meta.clip_vision_data.projection_dim == 512);
  CHECK(model.meta.clip_vision_data.block_count == 12);
  CHECK(model.meta.clip_vision_data.spatial_merge_size == 2);
  CHECK(model.meta.clip_vision_data.n_wa_pattern == 4);
  CHECK(model.meta.clip_vision_data.window_size == 7);
  CHECK(model.meta.clip_vision_data.attention_head_count == 8);
  CHECK(model.meta.clip_vision_data.attention_layer_norm_epsilon ==
        doctest::Approx(1e-5f));
  CHECK(model.meta.clip_vision_data.projector_scale_factor == 2);
  CHECK(model.meta.clip_vision_data.image_mean_count == 3);
  CHECK(model.meta.clip_vision_data.image_mean[0] == doctest::Approx(0.5f));
  CHECK(model.meta.clip_vision_data.image_std[2] == doctest::Approx(0.2f));
  CHECK(model.meta.clip_vision_data.wa_layer_index_count == 3);
  CHECK(model.meta.clip_vision_data.wa_layer_indexes[2] == 5);
  CHECK(model.meta.clip_vision_data.deepstack_layer_count == 3);
  CHECK(model.meta.clip_vision_data.deepstack_layers[0] == 1);
  CHECK(meta_view(model.meta, model.meta.clip_audio_data.projector_type) == "aproj");
  CHECK(model.meta.clip_audio_data.num_mel_bins == 80);
  CHECK(model.meta.clip_audio_data.embedding_length == 256);
  CHECK(model.meta.clip_audio_data.feed_forward_length == 1024);
  CHECK(model.meta.clip_audio_data.projection_dim == 128);
  CHECK(model.meta.clip_audio_data.block_count == 4);
  CHECK(model.meta.clip_audio_data.attention_head_count == 4);
  CHECK(model.meta.clip_audio_data.attention_layer_norm_epsilon ==
        doctest::Approx(1e-6f));
  CHECK(model.meta.clip_audio_data.projector_stack_factor == 2);
  CHECK(model.meta.diffusion_data.shift_logits);
  CHECK(model.meta.xielu_data.alpha_p_count == 2);
  CHECK(model.meta.xielu_data.alpha_p[1] == doctest::Approx(0.2f));
  CHECK(model.meta.xielu_data.eps[0] == doctest::Approx(0.7f));
  CHECK(model.params.n_vocab == 32000);
  CHECK(model.params.n_ctx == 4096);
  CHECK(model.params.n_embd == 4096);
  CHECK(model.params.n_embd_out == 2048);
  CHECK(model.params.n_features == 128);
  CHECK(model.params.n_ff == 8192);
  CHECK(model.params.n_leading_dense_block == 2);
  CHECK(model.params.n_expert_ff == 1024);
  CHECK(model.params.n_expert_shared_ff == 512);
  CHECK(model.params.n_expert_chunk_ff == 256);
  CHECK(model.params.use_parallel_residual);
  CHECK(meta_view(model.meta, model.meta.llm_strings_data.tensor_data_layout) == "layout");
  CHECK(model.params.n_expert == 8);
  CHECK(model.params.n_expert_used == 2);
  CHECK(model.params.n_expert_shared == 1);
  CHECK(model.params.n_expert_group == 4);
  CHECK(model.params.n_expert_group_used == 2);
  CHECK(model.params.expert_weights_scale == doctest::Approx(0.7f));
  CHECK(model.params.expert_weights_norm);
  CHECK(model.params.expert_gating_func == 3);
  CHECK(model.params.expert_group_scale == doctest::Approx(0.5f));
  CHECK(model.params.experts_per_group == 2);
  CHECK(model.params.moe_every_n_layers == 4);
  CHECK(model.params.nextn_predict_layers == 3);
  CHECK(model.params.n_deepstack_layers == 2);
  CHECK(model.params.pooling_type == 1);
  CHECK(model.params.logit_scale == doctest::Approx(1.3f));
  CHECK(model.params.decoder_start_token_id == 2);
  CHECK(model.params.decoder_block_count == 6);
  CHECK(model.params.attn_logit_softcapping == doctest::Approx(0.9f));
  CHECK(model.params.router_logit_softcapping == doctest::Approx(1.1f));
  CHECK(model.params.final_logit_softcapping == doctest::Approx(1.2f));
  CHECK(model.params.swin_norm);
  CHECK(model.params.rescale_every_n_layers == 8);
  CHECK(model.params.time_mix_extra_dim == 16);
  CHECK(model.params.time_decay_extra_dim == 32);
  CHECK(model.params.residual_scale == doctest::Approx(0.8f));
  CHECK(model.params.embedding_scale == doctest::Approx(0.9f));
  CHECK(model.params.token_shift_count == 4);
  CHECK(model.params.interleave_moe_layer_step == 3);
  CHECK(model.params.full_attention_interval == 2);
  CHECK(model.params.activation_sparsity_scale == doctest::Approx(0.6f));
  CHECK(model.params.altup_active_idx == 1);
  CHECK(model.params.altup_num_inputs == 2);
  CHECK(model.params.embd_length_per_layer_input == 64);
  CHECK(model.params.dense_2_feat_in == 128);
  CHECK(model.params.dense_2_feat_out == 256);
  CHECK(model.params.dense_3_feat_in == 512);
  CHECK(model.params.dense_3_feat_out == 1024);
  CHECK(model.params.n_head == 12);
  CHECK(model.params.n_head_kv == 6);
  CHECK(model.params.attention_max_alibi_bias == doctest::Approx(1.4f));
  CHECK(model.params.attention_clamp_kqv == doctest::Approx(0.8f));
  CHECK(model.params.attention_key_length == 64);
  CHECK(model.params.attention_value_length == 64);
  CHECK(model.params.attention_layer_norm_epsilon == doctest::Approx(1e-5f));
  CHECK(model.params.attention_layer_norm_rms_epsilon == doctest::Approx(1e-6f));
  CHECK(model.params.attention_group_norm_epsilon == doctest::Approx(1e-4f));
  CHECK(model.params.attention_group_norm_groups == 2);
  CHECK(model.params.attention_causal);
  CHECK(model.params.attention_q_lora_rank == 8);
  CHECK(model.params.attention_kv_lora_rank == 4);
  CHECK(model.params.attention_decay_lora_rank == 2);
  CHECK(model.params.attention_iclr_lora_rank == 1);
  CHECK(model.params.attention_value_residual_mix_lora_rank == 3);
  CHECK(model.params.attention_gate_lora_rank == 5);
  CHECK(model.params.attention_relative_buckets_count == 32);
  CHECK(model.params.attention_sliding_window == 128);
  CHECK(model.params.attention_sliding_window_pattern_count == 3);
  CHECK(model.params.attention_sliding_window_pattern_flags[1] == 0);
  CHECK(model.params.attention_scale == doctest::Approx(1.5f));
  CHECK(model.params.attention_output_scale == doctest::Approx(1.1f));
  CHECK(model.params.attention_temperature_length == 16);
  CHECK(model.params.attention_temperature_scale == doctest::Approx(0.9f));
  CHECK(model.params.attention_key_length_mla == 32);
  CHECK(model.params.attention_value_length_mla == 32);
  CHECK(model.params.attention_indexer_head_count == 2);
  CHECK(model.params.attention_indexer_key_length == 16);
  CHECK(model.params.attention_indexer_top_k == 4);
  CHECK(model.params.attention_shared_kv_layers == 1);
  CHECK(model.params.n_rot == 128);
  CHECK(model.params.rope_dimension_sections_count == 2);
  CHECK(model.params.rope_dimension_sections[1] == 64);
  CHECK(model.params.rope_freq_base == doctest::Approx(5000.0f));
  CHECK(model.params.rope_freq_base_swa == doctest::Approx(6000.0f));
  CHECK(model.params.rope_scale_linear == doctest::Approx(1.2f));
  CHECK(meta_view(model.meta, model.meta.rope_data.scaling_type) == "linear");
  CHECK(model.params.rope_scaling_factor == doctest::Approx(2.2f));
  CHECK(model.params.rope_scaling_attn_factor == doctest::Approx(1.4f));
  CHECK(model.params.rope_scaling_orig_ctx_len == 1024);
  CHECK(model.params.rope_scaling_finetuned);
  CHECK(model.params.rope_scaling_yarn_log_multiplier == doctest::Approx(0.2f));
  CHECK(model.params.rope_scaling_yarn_ext_factor == doctest::Approx(0.3f));
  CHECK(model.params.rope_scaling_yarn_attn_factor == doctest::Approx(0.4f));
  CHECK(model.params.rope_scaling_yarn_beta_fast == doctest::Approx(0.5f));
  CHECK(model.params.rope_scaling_yarn_beta_slow == doctest::Approx(0.6f));
  CHECK(model.params.ssm_conv_kernel == 7);
  CHECK(model.params.ssm_inner_size == 8);
  CHECK(model.params.ssm_state_size == 9);
  CHECK(model.params.ssm_time_step_rank == 10);
  CHECK(model.params.ssm_group_count == 11);
  CHECK(model.params.ssm_dt_b_c_rms);
  CHECK(model.params.kda_head_dim == 32);
  CHECK(model.params.wkv_head_size == 64);
  CHECK(model.params.posnet_embd == 48);
  CHECK(model.params.posnet_block_count == 2);
  CHECK(model.params.convnext_embd == 96);
  CHECK(model.params.convnext_block_count == 3);
  CHECK(model.params.shortconv_l_cache == 4);
  CHECK(model.params.swiglu_clamp_exp_count == 3);
  CHECK(model.params.swiglu_clamp_exp[2] == doctest::Approx(0.3f));
  CHECK(model.params.swiglu_clamp_shexp_count == 3);
  CHECK(model.params.swiglu_clamp_shexp[1] == doctest::Approx(0.5f));
  CHECK(model.meta.classifier_data.label_count == 2);
  CHECK(meta_view(model.meta, model.meta.classifier_data.labels[1]) == "dog");

  std::remove(path);
}
