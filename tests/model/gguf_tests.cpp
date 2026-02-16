#include <array>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "doctest/doctest.h"
#include "emel/emel.h"
#include "emel/model/gguf/loader.hpp"

namespace {

bool write_u32(std::FILE * file, const uint32_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_i64(std::FILE * file, const int64_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_u64(std::FILE * file, const uint64_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_string(std::FILE * file, const char * value) {
  const uint64_t len = std::strlen(value);
  if (!write_u64(file, len)) {
    return false;
  }
  return std::fwrite(value, 1, len, file) == len;
}

bool write_kv_string(std::FILE * file, const char * key, const char * value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_string);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_string(file, value);
}

bool write_kv_u32(std::FILE * file, const char * key, const uint32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_u32);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_u32(file, value);
}

bool write_kv_array_u32(std::FILE * file, const char * key, const uint32_t * values, const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_array);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::model::gguf::value_type::k_u32);
  if (std::fwrite(&elem_type, 1, sizeof(elem_type), file) != sizeof(elem_type)) {
    return false;
  }
  if (!write_u64(file, count)) {
    return false;
  }
  return std::fwrite(values, sizeof(uint32_t), count, file) == count;
}

bool write_long_key_kv(std::FILE * file, const uint64_t key_len, const uint32_t value) {
  if (!write_u64(file, key_len)) {
    return false;
  }
  std::array<char, 300> key = {};
  std::memset(key.data(), 'a', key.size());
  if (std::fwrite(key.data(), 1, key_len, file) != key_len) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_u32);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_u32(file, value);
}

bool write_header(std::FILE * file, const uint32_t version, const int64_t n_tensors, const int64_t n_kv) {
  const char magic[4] = {'G', 'G', 'U', 'F'};
  if (std::fwrite(magic, 1, sizeof(magic), file) != sizeof(magic)) {
    return false;
  }
  if (!write_u32(file, version)) {
    return false;
  }
  if (!write_i64(file, n_tensors) || !write_i64(file, n_kv)) {
    return false;
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
  if (!write_i64(file, dims[0])) {
    return false;
  }
  if (!write_i64(file, dims[1])) {
    return false;
  }
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_u64(file, offset);
}

bool write_minimal_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 2;
  const uint32_t version = emel::model::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::model::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  const long meta_end = std::ftell(file);
  if (meta_end < 0) {
    std::fclose(file);
    return false;
  }
  const uint64_t alignment = emel::model::gguf::k_default_alignment;
  const uint64_t aligned =
    emel::model::gguf::align_up_u64(static_cast<uint64_t>(meta_end), alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  if (padding > 0) {
    std::array<uint8_t, emel::model::gguf::k_default_alignment> zeros = {};
    if (std::fwrite(zeros.data(), 1, static_cast<size_t>(padding), file) != padding) {
      std::fclose(file);
      return false;
    }
  }
  std::array<uint8_t, 128> weights = {};
  if (std::fwrite(weights.data(), 1, weights.size(), file) != weights.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_gguf_without_arch(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 1;
  const uint32_t version = emel::model::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  const long meta_end = std::ftell(file);
  if (meta_end < 0) {
    std::fclose(file);
    return false;
  }
  const uint64_t alignment = emel::model::gguf::k_default_alignment;
  const uint64_t aligned =
    emel::model::gguf::align_up_u64(static_cast<uint64_t>(meta_end), alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  if (padding > 0) {
    std::array<uint8_t, emel::model::gguf::k_default_alignment> zeros = {};
    if (std::fwrite(zeros.data(), 1, static_cast<size_t>(padding), file) != padding) {
      std::fclose(file);
      return false;
    }
  }
  std::array<uint8_t, 128> weights = {};
  if (std::fwrite(weights.data(), 1, weights.size(), file) != weights.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_bad_alignment_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 0;
  const int64_t n_kv = 1;
  const uint32_t version = emel::model::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::model::gguf::k_key_alignment, 3)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_bad_tensor_type_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 2;
  const uint32_t version = emel::model::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::model::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = 99;
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_bad_tensor_offset_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 2;
  const uint32_t version = emel::model::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::model::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 128)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool make_temp_path(char * out, const size_t capacity) {
#if defined(_WIN32)
  return tmpnam_s(out, capacity) == 0;
#else
  if (capacity < 20) {
    return false;
  }
  std::snprintf(out, capacity, "/tmp/emel_ggufXXXXXX");
  const int fd = mkstemp(out);
  if (fd < 0) {
    return false;
  }
  close(fd);
  return true;
#endif
}

}  // namespace

TEST_CASE("gguf loader parses metadata and tensors") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::parser::event::parse_model parse_request{
    .model = &model,
    .model_path = path,
    .format_ctx = &ctx
  };

  CHECK(emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_OK);
  CHECK(std::strcmp(model.architecture_name.data(), "llama") == 0);

  CHECK(emel::model::gguf::parse_hparams(parse_request, &err));
  CHECK(err == EMEL_OK);
  CHECK(model.n_layers == 2);

  CHECK(emel::model::gguf::map_tensors(parse_request, &err));
  CHECK(err == EMEL_OK);
  CHECK(model.n_tensors == 1);
  CHECK(model.tensors[0].data_size == 128);
  CHECK(model.weights_size == 128);

  std::remove(path);
}

TEST_CASE("gguf loader rejects missing architecture") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_gguf_without_arch(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::parser::event::parse_model parse_request{
    .model = &model,
    .model_path = path,
    .format_ctx = &ctx
  };

  CHECK(!emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader rejects invalid alignment") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_bad_alignment_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader rejects invalid tensor type") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_bad_tensor_type_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader rejects invalid tensor offsets") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_bad_tensor_offset_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader accepts long key and array kv") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));

  std::FILE * file = std::fopen(path, "wb");
  REQUIRE(file != nullptr);
  const int64_t n_tensors = 1;
  const int64_t n_kv = 4;
  const uint32_t version = emel::model::gguf::k_gguf_version;
  REQUIRE(write_header(file, version, n_tensors, n_kv));
  REQUIRE(write_kv_string(file, emel::model::gguf::k_key_architecture, "llama"));
  REQUIRE(write_kv_u32(file, "llama.block_count", 2));
  const uint32_t arr[2] = {1, 2};
  REQUIRE(write_kv_array_u32(file, "dummy.array", arr, 2));
  REQUIRE(write_long_key_kv(file, 260, 7));
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  REQUIRE(write_tensor_info(file, "tensor.weight", type, dims, 0));
  const long meta_end = std::ftell(file);
  REQUIRE(meta_end >= 0);
  const uint64_t aligned =
    emel::model::gguf::align_up_u64(static_cast<uint64_t>(meta_end),
                                    emel::model::gguf::k_default_alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  if (padding > 0) {
    std::array<uint8_t, emel::model::gguf::k_default_alignment> zeros = {};
    REQUIRE(std::fwrite(zeros.data(), 1, static_cast<size_t>(padding), file) == padding);
  }
  std::array<uint8_t, 128> weights = {};
  REQUIRE(std::fwrite(weights.data(), 1, weights.size(), file) == weights.size());
  std::fclose(file);

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::remove(path);
}

TEST_CASE("gguf loader checks architecture list") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;
  const char * archs[1] = {"llama"};
  request.architectures = archs;
  request.n_architectures = 1;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::parser::event::parse_model parse_request{
    .model = &model,
    .model_path = path,
    .architectures = archs,
    .n_architectures = 1,
    .format_ctx = &ctx
  };

  CHECK(emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(emel::model::gguf::map_architecture(parse_request, &err));
  CHECK(err == EMEL_OK);

  std::remove(path);
}

TEST_CASE("gguf loader streams weights into provided buffer") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == 128);
  CHECK(total == 128);
  CHECK(model.tensors[0].data == buffer.data());

  std::remove(path);
}

TEST_CASE("gguf loader rejects too-small weight buffer") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 64> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::remove(path);
}

TEST_CASE("gguf loader validates structure") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(emel::model::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_OK);

  ctx.data_offset = 0;
  CHECK(!emel::model::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}


#if !defined(_WIN32)
TEST_CASE("gguf loader maps weights via mmap") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::model::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == 128);
  CHECK(total == 128);
  CHECK(model.tensors[0].data != nullptr);

  emel::model::gguf::reset_context(ctx);
  std::remove(path);
}
#endif
