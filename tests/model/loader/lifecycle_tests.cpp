#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

void on_done(void * object, const emel::model::loader::events::load_done & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->done = true;
  owner->error = false;
  owner->bytes_total = ev.bytes_total;
  owner->bytes_done = ev.bytes_done;
  owner->used_mmap = ev.used_mmap;
}

void on_error(void * object, const emel::model::loader::events::load_error & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->done = false;
  owner->error = true;
  owner->err = ev.err;
}

emel::error::type parse_ok(void *, const emel::model::loader::event::load & req) noexcept {
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type parse_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::parse_failed);
}

emel::error::type load_weights_ok(void *,
                                  const emel::model::loader::event::load & req,
                                  uint64_t & bytes_total,
                                  uint64_t & bytes_done,
                                  bool & used_mmap) noexcept {
  static_cast<void>(req);
  bytes_total = 4096;
  bytes_done = 4096;
  used_mmap = true;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type load_weights_backend_error(void *,
                                             const emel::model::loader::event::load &,
                                             uint64_t & bytes_total,
                                             uint64_t & bytes_done,
                                             bool & used_mmap) noexcept {
  bytes_total = 0;
  bytes_done = 0;
  used_mmap = false;
  return emel::error::cast(emel::model::loader::error::backend_error);
}

emel::error::type map_layers_ok(void *, const emel::model::loader::event::load & req) noexcept {
  req.model_data.n_layers = 2;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type validate_structure_ok(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type validate_architecture_ok(void *,
                                           const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

void copy_name(std::array<char, emel::model::data::k_max_architecture_name> & dest,
               const std::string_view value) {
  dest.fill('\0');
  const size_t count = std::min(dest.size() - 1u, value.size());
  for (size_t i = 0; i < count; ++i) {
    dest[i] = value[i];
  }
}

void append_tensor_name(emel::model::data & model, emel::model::data::tensor_record & tensor,
                        const std::string_view name) {
  tensor.name_offset = model.name_bytes_used;
  tensor.name_length = static_cast<uint32_t>(name.size());
  for (size_t i = 0; i < name.size(); ++i) {
    model.name_storage[model.name_bytes_used + static_cast<uint32_t>(i)] = name[i];
  }
  model.name_bytes_used += static_cast<uint32_t>(name.size());
  tensor.n_dims = 2;
  tensor.dims[0] = 8;
  tensor.dims[1] = 8;
  tensor.data = &tensor;
  tensor.data_size = 64u;
}

void build_canonical_model(emel::model::data & model, const int32_t block_count) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "llama");
  model.n_layers = block_count;
  model.params.n_embd = 64;
  model.params.n_ctx = 128;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block, const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." + std::string{suffix});
  };

  add("token_embd.weight");
  add("output_norm.weight");
  add("output.weight");
  for (int32_t block = 0; block < block_count; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "attn_q.weight");
    add_block(block, "attn_k.weight");
    add_block(block, "attn_v.weight");
    add_block(block, "attn_output.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");
  }
  model.n_tensors = tensor_index;
}

void build_qwen3_model(emel::model::data & model,
                       const int32_t block_count,
                       const bool include_q_norm,
                       const bool include_k_norm) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "qwen3");
  model.n_layers = block_count;
  model.params.n_embd = 64;
  model.params.n_ctx = 128;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block, const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." + std::string{suffix});
  };

  add("token_embd.weight");
  add("output_norm.weight");
  add("output.weight");
  for (int32_t block = 0; block < block_count; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "attn_q.weight");
    add_block(block, "attn_k.weight");
    add_block(block, "attn_v.weight");
    if (include_q_norm) {
      add_block(block, "attn_q_norm.weight");
    }
    if (include_k_norm) {
      add_block(block, "attn_k_norm.weight");
    }
    add_block(block, "attn_output.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");
  }
  model.n_tensors = tensor_index;
}

}  // namespace

TEST_CASE("model loader lifecycle succeeds on full load path") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_ok};
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 4096);
  CHECK(owner.bytes_done == 4096);
  CHECK(owner.used_mmap);
}

TEST_CASE("model loader rejects missing source payload") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  emel::model::loader::event::load request{*model, parse_model};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader allows vocab-only parse without weight and map callbacks") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.vocab_only = true;
  request.check_tensors = false;
  request.validate_architecture = false;
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 0);
  CHECK(owner.bytes_done == 0);
  CHECK_FALSE(owner.used_mmap);
}

TEST_CASE("model loader propagates parse failure") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_fail};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::parse_failed));
}

TEST_CASE("model loader rejects full load without load_weights callback") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader rejects full load without map_layers callback") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader propagates load_weights backend error") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_backend_error};
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::backend_error));
}

TEST_CASE("model loader unclassified error guard matches only unclassified codes") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  emel::model::loader::event::load request{*model, parse_model};
  emel::model::loader::event::load_ctx load_ctx{};
  emel::model::loader::event::load_runtime runtime{request, load_ctx};
  const auto guard = emel::model::loader::guard::error_unclassified_code{};

  load_ctx.err = emel::error::cast(emel::model::loader::error::none);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::invalid_request);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::parse_failed);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::backend_error);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::model_invalid);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::internal_error);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::untracked);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = static_cast<emel::error::type>(0xFFFFu);
  CHECK(guard(runtime));
}

TEST_CASE("model_llama_detail_builds_execution_view_for_canonical_tensor_set") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 2);

  emel::model::llama::detail::execution_view view = {};
  const auto err = emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::none));
  CHECK(view.model == model.get());
  CHECK(view.block_count == 2);
  CHECK(view.token_embedding.name == "token_embd.weight");
  CHECK(view.output_norm.name == "output_norm.weight");
  CHECK(view.output.name == "output.weight");

  emel::model::llama::detail::block_view block = {};
  CHECK(emel::model::llama::detail::lookup_block_view(view, 1, block) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(block.index == 1);
  CHECK(block.attention_norm.name == "blk.1.attn_norm.weight");
  CHECK(block.feed_forward_up.name == "blk.1.ffn_up.weight");
}

TEST_CASE("model_llama_detail_rejects_missing_required_tensor") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 1);
  model->tensors[3].data = nullptr;
  model->tensors[3].data_size = 0u;

  emel::model::llama::detail::execution_view view = {};
  const auto err = emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(view.model == nullptr);
}

TEST_CASE("model_llama_detail_builds_qwen3_execution_view_for_canonical_tensor_set") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, true, true);

  emel::model::llama::detail::execution_view view = {};
  const auto err = emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::none));
  CHECK(view.model == model.get());
  CHECK(view.block_count == 1);
  CHECK(view.output.name == "output.weight");

  emel::model::llama::detail::block_view block = {};
  CHECK(emel::model::llama::detail::lookup_block_view(view, 0, block) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(block.attention_q_norm.name == "blk.0.attn_q_norm.weight");
  CHECK(block.attention_k_norm.name == "blk.0.attn_k_norm.weight");
}

TEST_CASE("model_llama_detail_rejects_qwen3_execution_view_without_attention_q_norm") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, false, true);

  emel::model::llama::detail::execution_view view = {};
  const auto err = emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(view.model == nullptr);
}

TEST_CASE("model_llama_detail_rejects_qwen3_execution_view_without_attention_k_norm") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, true, false);

  emel::model::llama::detail::execution_view view = {};
  const auto err = emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(view.model == nullptr);
}
