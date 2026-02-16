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

TEST_CASE("gguf tensor type helpers cover switch cases") {
  using emel::model::gguf::blck_size_for;
  using emel::model::gguf::tensor_type;
  using emel::model::gguf::type_size_for;

  for (int32_t i = 0; i < static_cast<int32_t>(tensor_type::k_count); ++i) {
    const auto type = static_cast<tensor_type>(i);
    (void)blck_size_for(type);
    (void)type_size_for(type);
  }

  CHECK(blck_size_for(tensor_type::k_f32) > 0);
  CHECK(type_size_for(tensor_type::k_f32) > 0);
  CHECK(blck_size_for(tensor_type::k_q4_0) > 0);
  CHECK(type_size_for(tensor_type::k_q4_0) > 0);
  CHECK(blck_size_for(tensor_type::k_q8_0) > 0);
  CHECK(type_size_for(tensor_type::k_q8_0) > 0);

  const auto invalid = static_cast<tensor_type>(static_cast<int32_t>(tensor_type::k_count));
  CHECK(blck_size_for(invalid) == 0);
  CHECK(type_size_for(invalid) == 0);
}

TEST_CASE("gguf parses u32 values across types") {
  using emel::model::gguf::parse_u32_value;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t value = 123u;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(parse_u32_value(r, value_type::k_u32, out));
    CHECK(out == value);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const int32_t value = 7;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(parse_u32_value(r, value_type::k_i32, out));
    CHECK(out == 7u);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const int32_t value = -1;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(!parse_u32_value(r, value_type::k_i32, out));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint16_t value = 5u;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(parse_u32_value(r, value_type::k_u16, out));
    CHECK(out == 5u);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t value = 9u;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(!parse_u32_value(r, value_type::k_f32, out));
    std::fclose(file);
  }
}

TEST_CASE("gguf compute_tensor_size handles invalid dimensions") {
  using emel::model::gguf::compute_tensor_size;
  using emel::model::gguf::tensor_type;

  uint64_t out = 0;
  std::array<int64_t, emel::model::gguf::k_max_dims> dims = {32, 1, 1, 1};
  CHECK(compute_tensor_size(dims, tensor_type::k_f32, out));
  CHECK(out > 0);

  dims = {-1, 1, 1, 1};
  CHECK(!compute_tensor_size(dims, tensor_type::k_f32, out));

  dims = {31, 1, 1, 1};
  CHECK(!compute_tensor_size(dims, tensor_type::k_q4_0, out));

  dims = {std::numeric_limits<int64_t>::max(), 2, 1, 1};
  CHECK(!compute_tensor_size(dims, tensor_type::k_f32, out));
}

TEST_CASE("gguf read_key and skip_value cover branches") {
  using emel::model::gguf::read_key;
  using emel::model::gguf::reader;
  using emel::model::gguf::skip_value;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 300;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    std::array<char, 300> payload = {};
    std::memset(payload.data(), 'a', payload.size());
    REQUIRE(std::fwrite(payload.data(), 1, payload.size(), file) == payload.size());
    std::rewind(file);
    reader r{file};
    std::array<char, 8> out = {};
    uint64_t out_len = 0;
    CHECK(read_key(r, out.data(), out.size(), out_len));
    CHECK(out_len == 0);
    CHECK(out[0] == '\0');
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char key[] = {'t', 'e', 's', 't'};
    REQUIRE(std::fwrite(key, 1, sizeof(key), file) == sizeof(key));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    CHECK(read_key(r, nullptr, 0, out_len));
    CHECK(out_len == len);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t val = 0x12345678u;
    REQUIRE(std::fwrite(&val, 1, sizeof(val), file) == sizeof(val));
    std::rewind(file);
    reader r{file};
    CHECK(skip_value(r, value_type::k_u32, 1));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 3;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'f', 'o', 'o'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    CHECK(skip_value(r, value_type::k_string, 1));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    CHECK(!skip_value(r, value_type::k_count, 1));
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_header rejects invalid versions") {
  using emel::model::gguf::parse_header;
  using emel::model::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char bad_magic[4] = {'B', 'A', 'D', '!'};
    REQUIRE(std::fwrite(bad_magic, 1, sizeof(bad_magic), file) == sizeof(bad_magic));
    std::rewind(file);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_header(file, 0, 0, 0));
    std::rewind(file);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t bad_version = emel::model::gguf::k_gguf_version + 1;
    REQUIRE(write_header(file, bad_version, 0, 0));
    std::rewind(file);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv binds pending block count") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_kv_u32(file, "llama.block_count", 7));
  REQUIRE(write_kv_string(file, emel::model::gguf::k_key_architecture, "llama"));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = 0;
  CHECK(parse_kv(r, ctx, 2, err));
  CHECK(ctx.block_count == 7);
  std::fclose(file);
}

TEST_CASE("gguf parser reports missing model data") {
  using emel::model::gguf::map_architecture;
  using emel::model::gguf::map_tensors;
  using emel::model::gguf::parse_hparams;

  emel::model::parser::event::parse_model parse_request{};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  CHECK(!map_tensors(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(!parse_hparams(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(!map_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf map_architecture rejects mismatched list") {
  emel::model::data model{};
  std::memcpy(model.architecture_name.data(), "llama", 6);

  const char * list[] = {"gpt", "other"};
  emel::model::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  parse_request.architectures = list;
  parse_request.n_architectures = 2;

  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::map_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("gguf parse_kv handles arrays and alignment") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t values[] = {1};
    REQUIRE(write_kv_array_u32(file, "numbers", values, 1));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(parse_kv(r, ctx, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::model::gguf::k_key_alignment));
    const int32_t type = static_cast<int32_t>(value_type::k_u32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const uint32_t alignment = 3;
    REQUIRE(write_u32(file, alignment));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_MODEL_INVALID);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "bad.array"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t elem_type = static_cast<int32_t>(value_type::k_count);
    REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
    const uint64_t count = 1;
    REQUIRE(write_u64(file, count));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv handles split metadata") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_kv_u32(file, emel::model::gguf::k_key_split_count, 2));
  REQUIRE(write_kv_u32(file, emel::model::gguf::k_key_split_no, 1));
  REQUIRE(write_kv_u32(file, emel::model::gguf::k_key_split_tensors, 4));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, 3, err));
  CHECK(ctx.split_count == 2);
  CHECK(ctx.split_no == 1);
  CHECK(ctx.split_tensors_count == 4);
  std::fclose(file);
}

TEST_CASE("gguf parse_tensors rejects oversize name") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_tensors;
  using emel::model::gguf::reader;
  using emel::model::gguf::tensor_type;

  emel::model::data model{};
  context ctx{};
  int32_t err = EMEL_OK;
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);

  const uint64_t name_len = emel::model::gguf::k_max_key_length;
  REQUIRE(std::fwrite(&name_len, 1, sizeof(name_len), file) == sizeof(name_len));
  std::array<char, emel::model::gguf::k_max_key_length> name = {};
  std::memset(name.data(), 'a', name.size());
  REQUIRE(std::fwrite(name.data(), 1, name.size(), file) == name.size());
  REQUIRE(write_u32(file, 1));
  REQUIRE(write_i64(file, 32));
  const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
  REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
  REQUIRE(write_u64(file, 0));
  std::rewind(file);

  reader r{file};
  CHECK(!parse_tensors(r, ctx, model, 1, err));
  std::fclose(file);
}

TEST_CASE("gguf helper functions cover branch paths") {
  using emel::model::gguf::add_overflow_u64;
  using emel::model::gguf::align_up_u64;
  using emel::model::gguf::key_equals;
  using emel::model::gguf::key_has_suffix;
  using emel::model::gguf::mul_overflow_u64;

  CHECK(align_up_u64(64, 32) == 64);
  CHECK(align_up_u64(65, 32) == 96);
  CHECK(align_up_u64(17, 0) == 17);

  uint64_t out = 0;
  CHECK(!mul_overflow_u64(2, 3, out));
  CHECK(out == 6u);
  CHECK(mul_overflow_u64(std::numeric_limits<uint64_t>::max(), 2, out));

  CHECK(!add_overflow_u64(2, 3, out));
  CHECK(out == 5u);
  CHECK(add_overflow_u64(std::numeric_limits<uint64_t>::max(), 1, out));

  CHECK(key_equals("abc", 3, "abc"));
  CHECK(!key_equals("abc", 2, "abc"));

  uint64_t prefix_len = 0;
  CHECK(key_has_suffix("foo.bar", 7, ".bar", prefix_len));
  CHECK(prefix_len == 3);
  CHECK(!key_has_suffix("foo", 3, ".bar", prefix_len));
  CHECK(!key_has_suffix("foo.baz", 7, ".bar", prefix_len));
}

TEST_CASE("gguf read_string and skip_value cover branches") {
  using emel::model::gguf::reader;
  using emel::model::gguf::skip_value;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    std::array<uint8_t, 128> zeros = {};
    REQUIRE(std::fwrite(zeros.data(), 1, zeros.size(), file) == zeros.size());
    std::rewind(file);
    reader r{file};
    CHECK(skip_value(r, value_type::k_u8, 4));
    CHECK(skip_value(r, value_type::k_i8, 4));
    CHECK(skip_value(r, value_type::k_bool, 4));
    CHECK(skip_value(r, value_type::k_u16, 2));
    CHECK(skip_value(r, value_type::k_i16, 2));
    CHECK(skip_value(r, value_type::k_u32, 2));
    CHECK(skip_value(r, value_type::k_i32, 2));
    CHECK(skip_value(r, value_type::k_f32, 2));
    CHECK(skip_value(r, value_type::k_u64, 1));
    CHECK(skip_value(r, value_type::k_i64, 1));
    CHECK(skip_value(r, value_type::k_f64, 1));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 5;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'h', 'e', 'l', 'l', 'o'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 8> out = {};
    CHECK(r.read_string(out.data(), out.size(), out_len));
    CHECK(out_len == len);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 8;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    std::array<char, 8> str = {'1','2','3','4','5','6','7','8'};
    REQUIRE(std::fwrite(str.data(), 1, str.size(), file) == str.size());
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 4> out = {};
    CHECK(!r.read_string(out.data(), out.size(), out_len));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'t', 'e', 's', 't'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    CHECK(r.read_string(nullptr, 0, out_len));
    CHECK(out_len == len);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_tensors validates counts") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  int32_t err = EMEL_OK;
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  emel::model::gguf::reader r{file};
  CHECK(!emel::model::gguf::parse_tensors(r, ctx, model, -1, err));
  std::fclose(file);
}

TEST_CASE("gguf parse_tensors rejects invalid entries") {
  using emel::model::gguf::parse_tensors;
  using emel::model::gguf::reader;
  using emel::model::gguf::tensor_type;

  emel::model::data model{};
  emel::model::gguf::context ctx{};
  int32_t err = EMEL_OK;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, ""));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 1));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 0));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, emel::model::gguf::k_max_dims + 1));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, -1));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 1));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_count) + 1;
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 0));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 31));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_q4_0);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 0));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 32));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 8));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }
}

TEST_CASE("gguf map_parser handles invalid inputs") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};

  {
    emel::model::loader::event::load request{model};
    int32_t err = EMEL_OK;
    CHECK(!emel::model::gguf::map_parser(request, &err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::model::loader::event::load request{model};
    request.format_ctx = &ctx;
    request.model_path = "";
    int32_t err = EMEL_OK;
    CHECK(!emel::model::gguf::map_parser(request, &err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char bad_magic[4] = {'B', 'A', 'D', '!'};
    REQUIRE(std::fwrite(bad_magic, 1, sizeof(bad_magic), file) == sizeof(bad_magic));
    std::rewind(file);

    emel::model::loader::event::load request{model};
    request.format_ctx = &ctx;
    request.file_handle = file;
    int32_t err = EMEL_OK;
    CHECK(!emel::model::gguf::map_parser(request, &err));
    CHECK(err == EMEL_ERR_FORMAT_UNSUPPORTED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_architecture and hparams handle invalid ctx") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  emel::model::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  parse_request.format_ctx = &ctx;

  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::memcpy(ctx.architecture.data(), "llama", 5);
  ctx.architecture_len = static_cast<uint32_t>(model.architecture_name.size());
  err = EMEL_OK;
  CHECK(!emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.architecture_len = 5;
  err = EMEL_OK;
  CHECK(emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(std::strcmp(model.architecture_name.data(), "llama") == 0);

  err = EMEL_OK;
  CHECK(!emel::model::gguf::parse_hparams(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.block_count = 3;
  err = EMEL_OK;
  CHECK(emel::model::gguf::parse_hparams(parse_request, &err));
  CHECK(model.n_layers == 3);
}

TEST_CASE("gguf map_tensors and map_layers validate counts") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  int32_t err = EMEL_OK;

  CHECK(!emel::model::gguf::map_tensors({.model = nullptr}, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::map_tensors(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  model.n_tensors = 1;
  err = EMEL_OK;
  CHECK(emel::model::gguf::map_tensors(parse_request, &err));

  err = EMEL_OK;
  CHECK(!emel::model::gguf::map_layers(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  model.n_layers = 2;
  err = EMEL_OK;
  CHECK(emel::model::gguf::map_layers(request, &err));
}

TEST_CASE("gguf parse_kv handles i32 alignment and string arrays") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::model::gguf::k_key_alignment));
    const int32_t type = static_cast<int32_t>(value_type::k_i32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t alignment = 64;
    REQUIRE(std::fwrite(&alignment, 1, sizeof(alignment), file) == sizeof(alignment));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(parse_kv(r, ctx, 1, err));
    CHECK(ctx.alignment == 64);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "strings"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t elem_type = static_cast<int32_t>(value_type::k_string);
    REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
    const uint64_t count = 2;
    REQUIRE(write_u64(file, count));
    REQUIRE(write_string(file, "a"));
    REQUIRE(write_string(file, "b"));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(parse_kv(r, ctx, 1, err));
    std::fclose(file);
  }
}

TEST_CASE("gguf map_parser rejects long path") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  std::array<char, emel::model::gguf::k_max_path_length + 1> buf = {};
  std::memset(buf.data(), 'a', buf.size());
  request.model_path = std::string_view{buf.data(), buf.size()};

  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf parse_tensors rejects oversize tensor count") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  int32_t err = EMEL_OK;
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  emel::model::gguf::reader r{file};
  CHECK(!emel::model::gguf::parse_tensors(r, ctx, model,
    emel::model::data::k_max_tensors + 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv handles architecture type errors") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, emel::model::gguf::k_key_architecture));
  const int32_t type = static_cast<int32_t>(value_type::k_u32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  const uint32_t value = 7;
  REQUIRE(write_u32(file, value));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  std::fclose(file);
}

TEST_CASE("gguf parse_header rejects zero low bits") {
  using emel::model::gguf::parse_header;
  using emel::model::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  const char magic[4] = {'G', 'G', 'U', 'F'};
  REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
  const uint32_t version = 0x00010000u;
  REQUIRE(std::fwrite(&version, 1, sizeof(version), file) == sizeof(version));
  const int64_t n_tensors = 0;
  const int64_t n_kv = 0;
  REQUIRE(std::fwrite(&n_tensors, 1, sizeof(n_tensors), file) == sizeof(n_tensors));
  REQUIRE(std::fwrite(&n_kv, 1, sizeof(n_kv), file) == sizeof(n_kv));
  std::rewind(file);

  reader r{file};
  emel::model::gguf::context ctx{};
  int64_t out_tensors = 0;
  int64_t out_kv = 0;
  CHECK(!parse_header(r, ctx, out_tensors, out_kv));
  std::fclose(file);
}

TEST_CASE("gguf parse_kv reports invalid architecture string length") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, emel::model::gguf::k_key_architecture));
  const int32_t type = static_cast<int32_t>(value_type::k_string);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  const uint64_t len = emel::model::gguf::k_max_architecture;
  REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
  std::array<char, emel::model::gguf::k_max_architecture> name = {};
  REQUIRE(std::fwrite(name.data(), 1, name.size(), file) == name.size());
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv handles array string read failure") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, "bad.strings"));
  const int32_t type = static_cast<int32_t>(value_type::k_array);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  const int32_t elem_type = static_cast<int32_t>(value_type::k_string);
  REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
  const uint64_t count = 1;
  REQUIRE(write_u64(file, count));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv ignores oversize block_count prefix") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  constexpr size_t prefix_len = 70;
  const char suffix[] = ".block_count";
  constexpr size_t suffix_len = sizeof(suffix) - 1;
  std::array<char, 96> key = {};
  std::memset(key.data(), 'a', key.size());
  std::memcpy(key.data() + prefix_len, suffix, suffix_len);
  const uint64_t key_len = prefix_len + suffix_len;
  REQUIRE(std::fwrite(&key_len, 1, sizeof(key_len), file) == sizeof(key_len));
  REQUIRE(std::fwrite(key.data(), 1, key_len, file) == key_len);
  const int32_t type = static_cast<int32_t>(value_type::k_u32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  REQUIRE(write_u32(file, 3));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, 1, err));
  CHECK(!ctx.pending_block_count_valid);
  std::fclose(file);
}

#if !defined(_WIN32)
TEST_CASE("gguf map_mmap handles invalid requests") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};

  emel::model::weight_loader::event::load_weights load_req{};
  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::loader::event::load request{model};
  request.format_ctx = nullptr;
  load_req.loader_request = &request;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  request.format_ctx = &ctx;
  request.model_path = "";
  err = EMEL_OK;
  CHECK(!emel::model::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.mapped_data = &ctx;
  ctx.mapped_size = 8;
  ctx.data_offset = 8;
  ctx.data_size = 16;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  ctx.mapped_data = nullptr;
  ctx.mapped_size = 0;
}
#endif

TEST_CASE("gguf load_streamed reports invalid requests") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};

  emel::model::weight_loader::event::load_weights load_req{};
  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::loader::event::load request{model};
  request.format_ctx = nullptr;
  load_req.loader_request = &request;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  request.format_ctx = &ctx;
  request.weights_buffer = nullptr;
  request.weights_buffer_size = 0;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf load_streamed handles fseek and fread failures") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  ctx.file = file;
  ctx.data_offset = 0;
  ctx.data_size = 8;

  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  std::array<uint8_t, 8> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  std::fclose(file);
  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);

  ctx.file = std::tmpfile();
  REQUIRE(ctx.file != nullptr);
  std::array<uint8_t, 4> small_data = {1, 2, 3, 4};
  REQUIRE(std::fwrite(small_data.data(), 1, small_data.size(), ctx.file) == small_data.size());
  std::rewind(ctx.file);
  err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);
  std::fclose(ctx.file);
  ctx.file = nullptr;
}

TEST_CASE("gguf reader rejects null file") {
  using emel::model::gguf::read_key;
  using emel::model::gguf::reader;

  reader r{};
  uint32_t value = 0;
  CHECK_FALSE(r.read(value));
  CHECK_FALSE(r.skip(4));
  uint64_t out_len = 0;
  std::array<char, 4> out = {};
  CHECK_FALSE(r.read_string(out.data(), out.size(), out_len));
  CHECK_FALSE(read_key(r, out.data(), out.size(), out_len));
}

TEST_CASE("gguf reader handles short strings") {
  using emel::model::gguf::read_key;
  using emel::model::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char data[2] = {'a', 'b'};
    REQUIRE(std::fwrite(data, 1, sizeof(data), file) == sizeof(data));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 8> out = {};
    CHECK(!r.read_string(out.data(), out.size(), out_len));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char data[2] = {'a', 'b'};
    REQUIRE(std::fwrite(data, 1, sizeof(data), file) == sizeof(data));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 8> out = {};
    CHECK(!read_key(r, out.data(), out.size(), out_len));
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_u32_value handles read failures") {
  using emel::model::gguf::parse_u32_value;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  reader r{file};
  uint32_t out = 0;
  CHECK(!parse_u32_value(r, value_type::k_i32, out));
  std::rewind(file);
  CHECK(!parse_u32_value(r, value_type::k_u16, out));
  std::fclose(file);
}

TEST_CASE("gguf parse_header handles short reads") {
  using emel::model::gguf::parse_header;
  using emel::model::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    std::rewind(file);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    const uint32_t version = emel::model::gguf::k_gguf_version;
    REQUIRE(std::fwrite(&version, 1, sizeof(version), file) == sizeof(version));
    std::rewind(file);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    const uint32_t version = emel::model::gguf::k_gguf_version;
    REQUIRE(std::fwrite(&version, 1, sizeof(version), file) == sizeof(version));
    const int64_t n_tensors = 1;
    REQUIRE(std::fwrite(&n_tensors, 1, sizeof(n_tensors), file) == sizeof(n_tensors));
    std::rewind(file);
    reader r{file};
    emel::model::gguf::context ctx{};
    int64_t out_tensors = 0;
    int64_t out_kv = 0;
    CHECK(!parse_header(r, ctx, out_tensors, out_kv));
    std::fclose(file);
  }
}

TEST_CASE("gguf store_name rejects overflow conditions") {
  using emel::model::gguf::store_name;

  emel::model::data model{};
  uint32_t offset = 0;
  model.name_bytes_used = static_cast<uint32_t>(model.name_storage.size()) + 1;
  CHECK(!store_name(model, "a", 1, offset));

  model.name_bytes_used = static_cast<uint32_t>(model.name_storage.size() - 1);
  CHECK(!store_name(model, "ab", 2, offset));
}

TEST_CASE("gguf helper functions cover zero multiplication") {
  using emel::model::gguf::mul_overflow_u64;

  uint64_t out = 1;
  CHECK(!mul_overflow_u64(0, 10, out));
  CHECK(out == 0);
  CHECK(!mul_overflow_u64(10, 0, out));
  CHECK(out == 0);
}

TEST_CASE("gguf parse_kv handles read failures") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "key"));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "key"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "key"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t elem_type = static_cast<int32_t>(value_type::k_u32);
    REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv reports split metadata parse failures") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::model::gguf::k_key_split_count));
    const int32_t type = static_cast<int32_t>(value_type::k_f32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::model::gguf::k_key_split_no));
    const int32_t type = static_cast<int32_t>(value_type::k_f32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::model::gguf::k_key_split_tensors));
    const int32_t type = static_cast<int32_t>(value_type::k_f32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv reports block_count parse failures") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, "llama.block_count"));
  const int32_t type = static_cast<int32_t>(value_type::k_f32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  std::rewind(file);
  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv reports alignment parse failures") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, emel::model::gguf::k_key_alignment));
  const int32_t type = static_cast<int32_t>(value_type::k_f32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv reports unknown skip failures") {
  using emel::model::gguf::context;
  using emel::model::gguf::parse_kv;
  using emel::model::gguf::reader;
  using emel::model::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, "unknown"));
  const int32_t type = static_cast<int32_t>(value_type::k_string);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  std::rewind(file);
  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_tensors handles read failures") {
  using emel::model::gguf::parse_tensors;
  using emel::model::gguf::reader;
  using emel::model::gguf::tensor_type;

  emel::model::data model{};
  emel::model::gguf::context ctx{};
  int32_t err = EMEL_OK;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 32));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 32));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, err));
    std::fclose(file);
  }
}

TEST_CASE("gguf compute_tensor_size rejects invalid type") {
  using emel::model::gguf::compute_tensor_size;
  using emel::model::gguf::tensor_type;

  uint64_t out = 0;
  std::array<int64_t, emel::model::gguf::k_max_dims> dims = {1, 1, 1, 1};
  const auto invalid = static_cast<tensor_type>(static_cast<int32_t>(tensor_type::k_count));
  CHECK(!compute_tensor_size(dims, invalid, out));
}

TEST_CASE("gguf map_parser reports file open failure") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::remove(path);
  request.model_path = path;

  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_IO);
}

TEST_CASE("gguf validate_structure handles null context") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf parse_architecture handles null model") {
  emel::model::gguf::context ctx{};
  emel::model::parser::event::parse_model parse_request{
    .model = nullptr,
    .format_ctx = &ctx
  };
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf map_architecture accepts empty whitelist") {
  emel::model::data model{};
  std::memcpy(model.architecture_name.data(), "llama", 6);
  emel::model::parser::event::parse_model parse_request{
    .model = &model,
    .architectures = nullptr,
    .n_architectures = 0
  };
  int32_t err = EMEL_OK;
  CHECK(emel::model::gguf::map_architecture(parse_request, &err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("gguf validate_structure rejects empty weights") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
}

#if !defined(_WIN32)
TEST_CASE("gguf map_mmap reports file open failure") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::remove(path);
  request.model_path = path;

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);
}
#endif

TEST_CASE("gguf load_streamed reports path errors") {
  emel::model::data model{};
  emel::model::gguf::context ctx{};
  ctx.data_size = 0;

  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  std::array<uint8_t, 1> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  request.model_path = "";
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::remove(path);
  request.model_path = path;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);
}

TEST_CASE("gguf load_streamed handles owned files") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  int32_t err = EMEL_OK;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);
  ctx.file = nullptr;

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

  char short_path[1024] = {};
  CHECK(make_temp_path(short_path, sizeof(short_path)));
  std::FILE * short_file = std::fopen(short_path, "wb");
  REQUIRE(short_file != nullptr);
  const uint32_t marker = 0xabcdef01u;
  REQUIRE(std::fwrite(&marker, 1, sizeof(marker), short_file) == sizeof(marker));
  std::fclose(short_file);

  std::array<uint8_t, 16> small_buffer = {};
  request.model_path = short_path;
  request.weights_buffer = small_buffer.data();
  request.weights_buffer_size = small_buffer.size();
  ctx.file = nullptr;
  ctx.data_offset = 0;
  ctx.data_size = 16;
  err = EMEL_OK;
  CHECK(!emel::model::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);

  std::remove(short_path);

  std::remove(path);
}
