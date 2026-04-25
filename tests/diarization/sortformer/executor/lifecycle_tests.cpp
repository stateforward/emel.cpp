#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/diarization/sortformer/executor/detail.hpp"
#include "emel/diarization/sortformer/executor/sm.hpp"
#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace {

namespace executor = emel::diarization::sortformer::executor;
namespace encoder_detail = emel::diarization::sortformer::encoder::detail;
namespace executor_detail = emel::diarization::sortformer::executor::detail;
namespace modules_detail = emel::diarization::sortformer::modules::detail;
namespace transformer_detail = emel::diarization::sortformer::transformer::detail;

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, modules_detail::k_tensor_count> k_module_specs{{
    {"mods.ep.b", 1, {modules_detail::k_hidden_dim, 0, 0, 0}},
    {"mods.ep.w", 2, {modules_detail::k_encoder_dim, modules_detail::k_hidden_dim, 0, 0}},
    {"mods.fh2h.b", 1, {modules_detail::k_hidden_dim, 0, 0, 0}},
    {"mods.fh2h.w", 2, {modules_detail::k_hidden_dim, modules_detail::k_hidden_dim, 0, 0}},
    {"mods.h2s.b", 1, {modules_detail::k_speaker_count, 0, 0, 0}},
    {"mods.h2s.w", 2, {modules_detail::k_pair_hidden_dim, modules_detail::k_speaker_count, 0, 0}},
    {"mods.sh2s.b", 1, {modules_detail::k_speaker_count, 0, 0, 0}},
    {"mods.sh2s.w", 2, {modules_detail::k_hidden_dim, modules_detail::k_speaker_count, 0, 0}},
}};

constexpr std::array<tensor_spec, encoder_detail::k_pre_tensor_count> k_encoder_pre_specs{{
    {"enc.pre.conv.0.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
    {"enc.pre.conv.0.w", 4, {3, 3, 1, encoder_detail::k_pre_channel_count}},
    {"enc.pre.conv.2.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
    {"enc.pre.conv.2.w", 4, {3, 3, 1, encoder_detail::k_pre_channel_count}},
    {"enc.pre.conv.3.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
    {"enc.pre.conv.3.w", 4, {1, 1, encoder_detail::k_pre_channel_count,
                              encoder_detail::k_pre_channel_count}},
    {"enc.pre.conv.5.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
    {"enc.pre.conv.5.w", 4, {3, 3, 1, encoder_detail::k_pre_channel_count}},
    {"enc.pre.conv.6.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
    {"enc.pre.conv.6.w", 4, {1, 1, encoder_detail::k_pre_channel_count,
                              encoder_detail::k_pre_channel_count}},
    {"enc.pre.out.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"enc.pre.out.w", 2, {encoder_detail::k_pre_expanded_dim,
                           encoder_detail::k_model_dim, 0, 0}},
}};

constexpr std::array<tensor_spec, encoder_detail::k_layer_tensor_count> k_encoder_layer_specs{{
    {"conv.bn.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.rm", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.rv", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.sc", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.sh", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.dw.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.dw.w", 3, {encoder_detail::k_depthwise_kernel, 1, encoder_detail::k_model_dim, 0}},
    {"conv.pw1.b", 1, {1024, 0, 0, 0}},
    {"conv.pw1.w", 3, {1, encoder_detail::k_model_dim, 1024, 0}},
    {"conv.pw2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.pw2.w", 3, {1, encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0}},
    {"ff1.l1.b", 1, {encoder_detail::k_feed_forward_dim, 0, 0, 0}},
    {"ff1.l1.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_feed_forward_dim, 0, 0}},
    {"ff1.l2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"ff1.l2.w", 2, {encoder_detail::k_feed_forward_dim, encoder_detail::k_model_dim, 0, 0}},
    {"ff2.l1.b", 1, {encoder_detail::k_feed_forward_dim, 0, 0, 0}},
    {"ff2.l1.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_feed_forward_dim, 0, 0}},
    {"ff2.l2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"ff2.l2.w", 2, {encoder_detail::k_feed_forward_dim, encoder_detail::k_model_dim, 0, 0}},
    {"nc.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nc.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff1.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff1.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff2.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"no.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"no.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nsa.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nsa.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.k.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.k.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.o.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.o.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.p.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.q.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.q.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.v.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.v.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.pbu", 2, {encoder_detail::k_attention_head_dim,
                     encoder_detail::k_attention_head_count, 0, 0}},
    {"att.pbv", 2, {encoder_detail::k_attention_head_dim,
                     encoder_detail::k_attention_head_count, 0, 0}},
}};

constexpr std::array<tensor_spec, transformer_detail::k_layer_tensor_count> k_transformer_specs{{
    {"sa.k.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.k.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"sa.o.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.o.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"sa.q.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.q.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"sa.v.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.v.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"ln1.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ln1.w", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ln2.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ln2.w", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ff.di.b", 1, {transformer_detail::k_inner_dim, 0, 0, 0}},
    {"ff.di.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_inner_dim, 0, 0}},
    {"ff.do.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ff.do.w", 2, {transformer_detail::k_inner_dim, transformer_detail::k_hidden_dim, 0, 0}},
}};

struct callback_probe {
  bool done_called = false;
  bool error_called = false;
  const executor::event::execute * request = nullptr;
  int32_t frame_count = 0;
  int32_t hidden_dim = 0;
  emel::error::type err = emel::error::cast(executor::error::none);

  void on_done(const executor::events::execute_done & ev) noexcept {
    done_called = true;
    request = ev.request;
    frame_count = ev.frame_count;
    hidden_dim = ev.hidden_dim;
  }

  void on_error(const executor::events::execute_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

struct model_fixture {
  emel::model::data model = {};
  std::vector<float> encoder_channel_bias =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count), 0.0f);
  std::vector<float> encoder_feature_depthwise =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count * 9), 0.0f);
  std::vector<float> encoder_channel_depthwise =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count * 9), 0.0f);
  std::vector<float> encoder_channel_pointwise =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count *
                                             encoder_detail::k_pre_channel_count),
                         0.0f);
  std::vector<float> encoder_output_bias =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim), 0.0f);
  std::vector<float> encoder_output_weight =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim *
                                             encoder_detail::k_pre_expanded_dim),
                         0.0f);
  std::vector<float> encoder_model_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim), 0.0f);
  std::vector<float> encoder_model_one =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim), 1.0f);
  std::vector<float> encoder_feed_forward_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_feed_forward_dim), 0.0f);
  std::vector<float> encoder_pair_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim * 2), 0.0f);
  std::vector<float> encoder_model_model_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim *
                                             encoder_detail::k_model_dim),
                         0.0f);
  std::vector<float> encoder_model_feed_forward_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim *
                                             encoder_detail::k_feed_forward_dim),
                         0.0f);
  std::vector<float> encoder_feed_forward_model_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_feed_forward_dim *
                                             encoder_detail::k_model_dim),
                         0.0f);
  std::vector<float> encoder_pointwise_1_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim * 2 *
                                             encoder_detail::k_model_dim),
                         0.0f);
  std::vector<float> encoder_depthwise_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim *
                                             encoder_detail::k_depthwise_kernel),
                         0.0f);
  std::vector<float> encoder_position_bias_zero =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_attention_head_dim *
                                             encoder_detail::k_attention_head_count),
                         0.0f);
  std::vector<float> hidden_zero =
      std::vector<float>(static_cast<size_t>(transformer_detail::k_hidden_dim), 0.0f);
  std::vector<float> hidden_one =
      std::vector<float>(static_cast<size_t>(transformer_detail::k_hidden_dim), 1.0f);
  std::vector<float> inner_zero =
      std::vector<float>(static_cast<size_t>(transformer_detail::k_inner_dim), 0.0f);
  std::vector<float> hidden_hidden =
      std::vector<float>(static_cast<size_t>(transformer_detail::k_hidden_dim *
                                             transformer_detail::k_hidden_dim),
                         0.0f);
  std::vector<float> inner_hidden =
      std::vector<float>(static_cast<size_t>(transformer_detail::k_inner_dim *
                                             transformer_detail::k_hidden_dim),
                         0.0f);
  std::vector<float> hidden_inner =
      std::vector<float>(static_cast<size_t>(transformer_detail::k_hidden_dim *
                                             transformer_detail::k_inner_dim),
                         0.0f);
  std::vector<float> encoder_projection =
      std::vector<float>(static_cast<size_t>(modules_detail::k_hidden_dim *
                                             modules_detail::k_encoder_dim),
                         0.0f);
  std::vector<float> speaker_bias =
      std::vector<float>(static_cast<size_t>(modules_detail::k_speaker_count), 0.0f);
  std::vector<float> speaker_hidden =
      std::vector<float>(static_cast<size_t>(modules_detail::k_speaker_count *
                                             modules_detail::k_hidden_dim),
                         0.0f);
  std::vector<float> speaker_pair =
      std::vector<float>(static_cast<size_t>(modules_detail::k_speaker_count *
                                             modules_detail::k_pair_hidden_dim),
                         0.0f);

  model_fixture(const bool include_all_tensors = true) {
    for (int32_t channel = 0; channel < encoder_detail::k_pre_channel_count; ++channel) {
      encoder_feature_depthwise[(static_cast<size_t>(channel) * 9u) + 4u] = 1.0f;
      encoder_channel_depthwise[(static_cast<size_t>(channel) * 9u) + 4u] = 1.0f;
      const size_t pointwise_index =
          (static_cast<size_t>(channel) * static_cast<size_t>(encoder_detail::k_pre_channel_count)) +
          static_cast<size_t>(channel);
      encoder_channel_pointwise[pointwise_index] = 1.0f;
    }

    for (int32_t dim = 0; dim < encoder_detail::k_model_dim; ++dim) {
      const int32_t channel = dim % encoder_detail::k_pre_channel_count;
      const size_t weight_index =
          (static_cast<size_t>(dim) * static_cast<size_t>(encoder_detail::k_pre_expanded_dim)) +
          (static_cast<size_t>(channel) * static_cast<size_t>(encoder_detail::k_pre_expand_lanes));
      encoder_output_weight[weight_index] = 1.0f;
    }

    for (size_t index = 0u; index < static_cast<size_t>(transformer_detail::k_hidden_dim);
         ++index) {
      hidden_hidden[(index * static_cast<size_t>(transformer_detail::k_hidden_dim)) + index] =
          1.0f;
      encoder_projection[(index * static_cast<size_t>(modules_detail::k_encoder_dim)) + index] =
          1.0f;
    }

    for (const auto & spec : k_encoder_pre_specs) {
      append_tensor(spec);
    }

    for (int32_t layer = 0; layer < encoder_detail::k_layer_count; ++layer) {
      for (size_t index = 0u; index < k_encoder_layer_specs.size(); ++index) {
        append_encoder_layer_tensor(layer, k_encoder_layer_specs[index]);
      }
    }

    for (const auto & spec : k_module_specs) {
      append_tensor(spec);
    }

    for (int32_t layer = 0; layer < transformer_detail::k_layer_count; ++layer) {
      for (size_t index = 0u; index < k_transformer_specs.size(); ++index) {
        if (!include_all_tensors && layer == 17 && index == k_transformer_specs.size() - 1u) {
          continue;
        }
        append_transformer_tensor(layer, k_transformer_specs[index]);
      }
    }
  }

  void append_name(emel::model::data::tensor_record & tensor, const std::string_view name) {
    const auto offset = model.name_bytes_used;
    std::memcpy(model.name_storage.data() + offset, name.data(), name.size());
    tensor.name_offset = offset;
    tensor.name_length = static_cast<uint32_t>(name.size());
    model.name_bytes_used += static_cast<uint32_t>(name.size());
  }

  std::span<const float> data_for(const std::string_view name) const noexcept {
    if (name == "enc.pre.conv.0.b" || name == "enc.pre.conv.2.b" ||
        name == "enc.pre.conv.3.b" || name == "enc.pre.conv.5.b" ||
        name == "enc.pre.conv.6.b") {
      return encoder_channel_bias;
    }
    if (name == "enc.pre.conv.0.w") {
      return encoder_feature_depthwise;
    }
    if (name == "enc.pre.conv.2.w" || name == "enc.pre.conv.5.w") {
      return encoder_channel_depthwise;
    }
    if (name == "enc.pre.conv.3.w" || name == "enc.pre.conv.6.w") {
      return encoder_channel_pointwise;
    }
    if (name == "enc.pre.out.b") {
      return encoder_output_bias;
    }
    if (name == "enc.pre.out.w") {
      return encoder_output_weight;
    }
    if (name.starts_with("enc.l")) {
      if (name.ends_with(".nff1.w") || name.ends_with(".nff2.w") ||
          name.ends_with(".nsa.w") || name.ends_with(".nc.w") ||
          name.ends_with(".no.w") || name.ends_with(".conv.bn.sc") ||
          name.ends_with(".conv.bn.rv")) {
        return encoder_model_one;
      }
      if (name.ends_with(".ff1.l1.b") || name.ends_with(".ff2.l1.b")) {
        return encoder_feed_forward_zero;
      }
      if (name.ends_with(".ff1.l1.w") || name.ends_with(".ff2.l1.w")) {
        return encoder_model_feed_forward_zero;
      }
      if (name.ends_with(".ff1.l2.w") || name.ends_with(".ff2.l2.w")) {
        return encoder_feed_forward_model_zero;
      }
      if (name.ends_with(".conv.pw1.b")) {
        return encoder_pair_zero;
      }
      if (name.ends_with(".conv.pw1.w")) {
        return encoder_pointwise_1_zero;
      }
      if (name.ends_with(".conv.dw.w")) {
        return encoder_depthwise_zero;
      }
      if (name.ends_with(".att.pbu") || name.ends_with(".att.pbv")) {
        return encoder_position_bias_zero;
      }
      if (name.ends_with(".w")) {
        return encoder_model_model_zero;
      }
      return encoder_model_zero;
    }
    if (name == "mods.ep.w") {
      return encoder_projection;
    }
    if (name == "mods.h2s.b" || name == "mods.sh2s.b") {
      return speaker_bias;
    }
    if (name == "mods.h2s.w") {
      return speaker_pair;
    }
    if (name == "mods.sh2s.w") {
      return speaker_hidden;
    }
    if (name.ends_with(".w") && name.find("ln") != std::string_view::npos) {
      return hidden_one;
    }
    if (name.ends_with(".b") && name.find("ff.di") != std::string_view::npos) {
      return inner_zero;
    }
    if (name.ends_with(".w") && name.find("ff.di") != std::string_view::npos) {
      return inner_hidden;
    }
    if (name.ends_with(".w") && name.find("ff.do") != std::string_view::npos) {
      return hidden_inner;
    }
    if (name.ends_with(".w")) {
      return hidden_hidden;
    }
    return hidden_zero;
  }

  void append_tensor(const tensor_spec & spec) {
    auto & tensor = model.tensors[model.n_tensors];
    append_name(tensor, spec.name);
    tensor.n_dims = spec.n_dims;
    tensor.dims = spec.dims;
    const auto values = data_for(spec.name);
    tensor.data = values.data();
    tensor.data_size = values.size_bytes();
    ++model.n_tensors;
  }

  void append_encoder_layer_tensor(const int32_t layer, const tensor_spec & spec) {
    std::array<char, 64> name = {};
    const int written = std::snprintf(name.data(),
                                      name.size(),
                                      "enc.l%d.%.*s",
                                      layer,
                                      static_cast<int>(spec.name.size()),
                                      spec.name.data());
    REQUIRE(written > 0);
    tensor_spec named_spec = spec;
    named_spec.name = std::string_view{name.data(), static_cast<size_t>(written)};
    append_tensor(named_spec);
  }

  void append_transformer_tensor(const int32_t layer, const tensor_spec & spec) {
    std::array<char, 64> name = {};
    const int written = std::snprintf(name.data(),
                                      name.size(),
                                      "te.l%d.%.*s",
                                      layer,
                                      static_cast<int>(spec.name.size()),
                                      spec.name.data());
    REQUIRE(written > 0);
    tensor_spec named_spec = spec;
    named_spec.name = std::string_view{name.data(), static_cast<size_t>(written)};
    append_tensor(named_spec);
  }
};

emel::model::sortformer::detail::execution_contract make_contract(
    const emel::model::data & model) noexcept {
  emel::model::sortformer::detail::execution_contract contract = {};
  contract.model = &model;
  contract.speaker_count = executor_detail::k_speaker_count;
  contract.chunk_len = executor_detail::k_frame_count;
  contract.encoder.tensor_count = 1u;
  contract.modules.tensor_count = modules_detail::k_tensor_count;
  contract.transformer_encoder.tensor_count = static_cast<uint32_t>(
      transformer_detail::k_layer_count * transformer_detail::k_layer_tensor_count);
  return contract;
}

std::vector<float> make_encoder_frames() {
  std::vector<float> frames(static_cast<size_t>(
      executor_detail::k_required_encoder_value_count));
  for (size_t index = 0u; index < frames.size(); ++index) {
    frames[index] = static_cast<float>(index % 29u) * 0.03125f;
  }
  return frames;
}

std::vector<float> make_features() {
  std::vector<float> features(static_cast<size_t>(
      encoder_detail::k_required_feature_value_count));
  for (size_t index = 0u; index < features.size(); ++index) {
    features[index] = static_cast<float>(index % 31u) * 0.015625f;
  }
  return features;
}

executor::event::execute make_request(
    const emel::model::sortformer::detail::execution_contract & contract,
    std::span<const float> encoder_frames,
    std::span<float> hidden_out,
    int32_t & frame_count,
    int32_t & hidden_dim,
    emel::error::type & err) noexcept {
  executor::event::execute request{contract, encoder_frames, hidden_out, frame_count, hidden_dim};
  request.error_out = &err;
  return request;
}

}  // namespace

TEST_CASE("sortformer executor produces deterministic hidden frames") {
  auto fixture = std::make_unique<model_fixture>();
  const auto contract = make_contract(fixture->model);
  const auto encoder_frames = make_encoder_frames();
  std::vector<float> hidden_out(static_cast<size_t>(
      executor_detail::k_required_hidden_value_count));
  int32_t frame_count = -1;
  int32_t hidden_dim = -1;
  emel::error::type err = emel::error::cast(executor::error::kernel);
  callback_probe probe = {};

  auto request = make_request(contract, encoder_frames, hidden_out, frame_count, hidden_dim, err);
  request.on_done = emel::callback<void(const executor::events::execute_done &)>::from<
      callback_probe,
      &callback_probe::on_done>(&probe);

  executor::sm machine{};
  REQUIRE(machine.process_event(request));
  CHECK(machine.is(boost::sml::state<executor::state_ready>));
  CHECK(err == emel::error::cast(executor::error::none));
  CHECK(frame_count == executor_detail::k_frame_count);
  CHECK(hidden_dim == executor_detail::k_hidden_dim);
  CHECK(probe.done_called);
  CHECK_FALSE(probe.error_called);
  CHECK(probe.request == &request);
  CHECK(probe.frame_count == executor_detail::k_frame_count);
  CHECK(probe.hidden_dim == executor_detail::k_hidden_dim);
  CHECK(std::all_of(hidden_out.begin(), hidden_out.end(), [](const float value) {
    return std::isfinite(value);
  }));
  CHECK(hidden_out.front() != hidden_out.back());

  std::vector<float> second_hidden(hidden_out.size());
  int32_t second_frame_count = -1;
  int32_t second_hidden_dim = -1;
  emel::error::type second_err = emel::error::cast(executor::error::kernel);
  auto second_request = make_request(contract,
                                     encoder_frames,
                                     second_hidden,
                                     second_frame_count,
                                     second_hidden_dim,
                                     second_err);
  REQUIRE(machine.process_event(second_request));
  CHECK(second_err == emel::error::cast(executor::error::none));
  CHECK(second_hidden == hidden_out);
}

TEST_CASE("sortformer executor consumes native feature-to-encoder frames") {
  auto fixture = std::make_unique<model_fixture>();
  encoder_detail::contract encoder_contract = {};
  REQUIRE(encoder_detail::bind_contract(fixture->model, encoder_contract));

  const auto contract = make_contract(fixture->model);
  const auto features = make_features();
  encoder_detail::pre_encoder_workspace encoder_workspace = {};
  std::vector<float> encoder_frames(static_cast<size_t>(
      encoder_detail::k_required_encoder_value_count));
  REQUIRE(encoder_detail::compute_encoder_frames_from_features(features,
                                                               encoder_contract,
                                                               encoder_workspace,
                                                               encoder_frames));

  std::vector<float> hidden_out(static_cast<size_t>(
      executor_detail::k_required_hidden_value_count));
  int32_t frame_count = -1;
  int32_t hidden_dim = -1;
  emel::error::type err = emel::error::cast(executor::error::kernel);

  auto request = make_request(contract, encoder_frames, hidden_out, frame_count, hidden_dim, err);

  executor::sm machine{};
  REQUIRE(machine.process_event(request));
  CHECK(err == emel::error::cast(executor::error::none));
  CHECK(frame_count == executor_detail::k_frame_count);
  CHECK(hidden_dim == executor_detail::k_hidden_dim);
  CHECK(std::all_of(hidden_out.begin(), hidden_out.end(), [](const float value) {
    return std::isfinite(value);
  }));
}

TEST_CASE("sortformer executor rejects invalid model contract") {
  auto fixture = std::make_unique<model_fixture>();
  auto contract = make_contract(fixture->model);
  contract.model = nullptr;
  const auto encoder_frames = make_encoder_frames();
  std::vector<float> hidden_out(static_cast<size_t>(
      executor_detail::k_required_hidden_value_count));
  int32_t frame_count = -1;
  int32_t hidden_dim = -1;
  emel::error::type err = emel::error::cast(executor::error::none);

  auto request = make_request(contract, encoder_frames, hidden_out, frame_count, hidden_dim, err);
  executor::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(executor::error::model_invalid));
  CHECK(frame_count == 0);
  CHECK(hidden_dim == 0);
}

TEST_CASE("sortformer executor rejects missing maintained tensor") {
  auto fixture = std::make_unique<model_fixture>(false);
  const auto contract = make_contract(fixture->model);
  const auto encoder_frames = make_encoder_frames();
  std::vector<float> hidden_out(static_cast<size_t>(
      executor_detail::k_required_hidden_value_count));
  int32_t frame_count = -1;
  int32_t hidden_dim = -1;
  emel::error::type err = emel::error::cast(executor::error::none);

  auto request = make_request(contract, encoder_frames, hidden_out, frame_count, hidden_dim, err);
  executor::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(executor::error::tensor_contract));
  CHECK(frame_count == 0);
  CHECK(hidden_dim == 0);
}

TEST_CASE("sortformer executor rejects invalid encoder frame shape") {
  auto fixture = std::make_unique<model_fixture>();
  const auto contract = make_contract(fixture->model);
  std::vector<float> encoder_frames(128u, 0.25f);
  std::vector<float> hidden_out(static_cast<size_t>(
      executor_detail::k_required_hidden_value_count));
  int32_t frame_count = -1;
  int32_t hidden_dim = -1;
  emel::error::type err = emel::error::cast(executor::error::none);

  auto request = make_request(contract, encoder_frames, hidden_out, frame_count, hidden_dim, err);
  executor::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(executor::error::input_shape));
  CHECK(frame_count == 0);
  CHECK(hidden_dim == 0);
}

TEST_CASE("sortformer executor rejects insufficient hidden output capacity") {
  auto fixture = std::make_unique<model_fixture>();
  const auto contract = make_contract(fixture->model);
  const auto encoder_frames = make_encoder_frames();
  std::vector<float> hidden_out(static_cast<size_t>(
      executor_detail::k_required_hidden_value_count - 1));
  int32_t frame_count = -1;
  int32_t hidden_dim = -1;
  emel::error::type err = emel::error::cast(executor::error::none);
  callback_probe probe = {};

  auto request = make_request(contract, encoder_frames, hidden_out, frame_count, hidden_dim, err);
  request.on_error = emel::callback<void(const executor::events::execute_error &)>::from<
      callback_probe,
      &callback_probe::on_error>(&probe);

  executor::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(executor::error::output_capacity));
  CHECK(probe.error_called);
  CHECK_FALSE(probe.done_called);
  CHECK(probe.err == emel::error::cast(executor::error::output_capacity));
  CHECK(frame_count == 0);
  CHECK(hidden_dim == 0);
}
