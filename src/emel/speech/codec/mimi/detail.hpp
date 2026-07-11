#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/model/data.hpp"

// Shared Mimi codec binding, sizing, and kernel-dispatch compute helpers.
//
// The Mimi streaming codec (Kyutai Moshi / NVIDIA PersonaPlex) maps 24 kHz
// mono PCM to n_q discrete codebook streams at 12.5 Hz and back:
//   encode: SEANet conv frontend (25 Hz) -> downsample conv (12.5 Hz)
//           -> codec transformer -> split RVQ nearest-codebook search
//   decode: split RVQ codebook sums -> codec transformer -> depthwise
//           transposed-conv upsample (25 Hz) -> SEANet conv decoder (24 kHz)
//
// All numeric work is dispatched to the kernel backend machines as op events
// (generator pattern: the runtime struct owns an emel::kernel::sm). Helpers
// here are non-routing data-plane sequences for the already-chosen variant;
// behavior selection lives in the owning actors' guards and transitions.
//
// Streaming semantics (must match the reference bit-for-bit on the encode
// path):
//  - streaming conv: carries the last (kernel - stride) input samples as left
//    context (zero-initialized), convolves concat(state, frame) unpadded.
//  - streaming transposed conv: full output of length (len-1)*stride + kernel;
//    the first (kernel - stride) outputs accumulate the previous call's tail;
//    the tail is withheld (emitted next call); bias applies to emitted range.
namespace emel::speech::codec::mimi::detail {

inline constexpr int32_t k_max_seanet_layers = 15;
inline constexpr int32_t k_max_transformer_layers = 16;
inline constexpr int32_t k_max_quantizer_levels = 32;

enum class seanet_layer_kind : uint8_t {
  none = 0,
  conv = 1,
  resnet = 2,
  elu = 3,
  conv_transpose = 4,
};

// Fixed mimi_v0_1 SEANet topology (module index -> layer kind and stride).
// Kernel sizes and channel counts are read from the bound tensor shapes and
// cross-checked against the contract at prepare time.
struct seanet_layer_spec {
  seanet_layer_kind kind = seanet_layer_kind::none;
  int32_t stride = 1;
};

inline constexpr std::array<seanet_layer_spec, k_max_seanet_layers>
    k_encoder_topology = {{
        {seanet_layer_kind::conv, 1},   // model.0  k7 1->C
        {seanet_layer_kind::resnet, 1}, // model.1
        {seanet_layer_kind::elu, 1},    // model.2
        {seanet_layer_kind::conv, 4},   // model.3  k8
        {seanet_layer_kind::resnet, 1}, // model.4
        {seanet_layer_kind::elu, 1},    // model.5
        {seanet_layer_kind::conv, 5},   // model.6  k10
        {seanet_layer_kind::resnet, 1}, // model.7
        {seanet_layer_kind::elu, 1},    // model.8
        {seanet_layer_kind::conv, 6},   // model.9  k12
        {seanet_layer_kind::resnet, 1}, // model.10
        {seanet_layer_kind::elu, 1},    // model.11
        {seanet_layer_kind::conv, 8},   // model.12 k16
        {seanet_layer_kind::elu, 1},    // model.13
        {seanet_layer_kind::conv, 1},   // model.14 k3 -> dim
    }};

inline constexpr std::array<seanet_layer_spec, k_max_seanet_layers>
    k_decoder_topology = {{
        {seanet_layer_kind::conv, 1},           // model.0  k7 dim->C
        {seanet_layer_kind::elu, 1},            // model.1
        {seanet_layer_kind::conv_transpose, 8}, // model.2  k16
        {seanet_layer_kind::resnet, 1},         // model.3
        {seanet_layer_kind::elu, 1},            // model.4
        {seanet_layer_kind::conv_transpose, 6}, // model.5  k12
        {seanet_layer_kind::resnet, 1},         // model.6
        {seanet_layer_kind::elu, 1},            // model.7
        {seanet_layer_kind::conv_transpose, 5}, // model.8  k10
        {seanet_layer_kind::resnet, 1},         // model.9
        {seanet_layer_kind::elu, 1},            // model.10
        {seanet_layer_kind::conv_transpose, 4}, // model.11 k8
        {seanet_layer_kind::resnet, 1},         // model.12
        {seanet_layer_kind::elu, 1},            // model.13
        {seanet_layer_kind::conv, 1},           // model.14 k3 -> 1
    }};

// One bound conv weight: canonical f32 taps in the prepared arena. The
// state fields locate this layer's streaming slice (left context for convs,
// full-output overlap tail for transposed convs) inside the state arena.
struct conv_weights {
  const float *weight = nullptr; // [taps * in_channels * out_channels]
  // Raw f16 taps (ggml reshape layout [taps*in, out], out-channel rows
  // contiguous) when the model stores f16 conv weights; the owning actor
  // selects the f16 compute variant through guard rows on
  // codec_runtime::conv_f16.
  const uint16_t *weight_f16 = nullptr;
  const float *bias = nullptr; // [out_channels] or nullptr
  int32_t taps = 0;
  int32_t in_channels = 0;
  int32_t out_channels = 0;
  int32_t stride = 1;
  int32_t frame_length = 0; // input time steps per dispatch at this layer
  uint64_t state_offset = 0;
  uint64_t state_floats = 0;
};

struct resnet_weights {
  conv_weights block_1 = {}; // k3 C -> C/2
  conv_weights block_3 = {}; // k1 C/2 -> C
};

struct seanet_layer_weights {
  seanet_layer_kind kind = seanet_layer_kind::none;
  conv_weights conv = {};
  resnet_weights resnet = {};
};

struct transformer_layer_weights {
  const float *norm1_weight = nullptr;
  const float *norm1_bias = nullptr;
  const float *in_proj = nullptr;  // [dim, 3*dim] fused qkv
  const float *out_proj = nullptr; // [dim, dim]
  // raw q8_0 row blocks when the model carries quantized projections
  const uint8_t *in_proj_q8 = nullptr;
  const uint8_t *out_proj_q8 = nullptr;
  const uint8_t *linear1_q8 = nullptr;
  const uint8_t *linear2_q8 = nullptr;
  const float *layer_scale_1 = nullptr;
  const float *norm2_weight = nullptr;
  const float *norm2_bias = nullptr;
  const float *linear1 = nullptr; // [dim, mlp_dim]
  const float *linear2 = nullptr; // [mlp_dim, dim]
  const float *layer_scale_2 = nullptr;
  int32_t mlp_dim = 0;
};

struct transformer_weights {
  std::array<transformer_layer_weights, k_max_transformer_layers> layers = {};
  int32_t layer_count = 0;
  int32_t dim = 0;
  int32_t head_count = 0;
  int32_t context = 0;
  int32_t max_period = 0;
  int32_t frame_tokens = 0;  // tokens per dispatch (2 at 25 Hz)
  uint64_t state_offset = 0; // K/V ring: layers * 2 * context * dim floats
  uint64_t state_floats = 0;
};

// One RVQ split (rvq_first carries the semantic level, rvq_rest the acoustic
// levels). Codebook rows stay in reference layout; encode scans them with the
// reference squared-distance arithmetic so near-tie argmax behavior does not
// drift from ggml.
struct rvq_split_weights {
  const float *input_proj = nullptr;  // [dim, codebook_dim] (conv1x1)
  const float *output_proj = nullptr; // [codebook_dim, dim]
  // Raw f16 projections (ggml layout) when the model stores f16 weights.
  const uint16_t *input_proj_f16 = nullptr;
  const uint16_t *output_proj_f16 = nullptr;
  // raw q8_0 row blocks when the model carries quantized projections
  const uint8_t *input_proj_q8 = nullptr;
  const uint8_t *output_proj_q8 = nullptr;
  // per level: raw codebook rows [codebook_dim, entries]
  std::array<const float *, k_max_quantizer_levels> codebooks = {};
  int32_t level_count = 0;
};

struct quantizer_weights {
  rvq_split_weights semantic = {};
  rvq_split_weights acoustic = {};
  int32_t codebook_entries = 0;
  int32_t codebook_dim = 0;
};

// Prepared, bound codec runtime. Weights point into the prepared arena (f16
// sources canonicalized to f32 once at prepare; fp16 -> fp32 is lossless so
// the effective operand class is unchanged). Owned by the facade context;
// injected into the actors by reference.
struct codec_runtime {
  const emel::model::data *model = nullptr;
  emel::kernel::sm kernel = {};
  // Host backend, resolved at compile time and applied once at bind; the
  // compute helpers never re-select it.
  emel::kernel::kernel_kind kernel_kind = emel::kernel::detect_host_kind();
  // Bound model properties: conv_f16 selects the reference f16 conv operand
  // class; proj_q8 selects the emel-owned q8_0 projection operand class
  // (transformer + RVQ projections quantized by the converter). Guards on
  // the owning actors read these to select the matching compute rows.
  bool conv_f16 = false;
  bool proj_q8 = false;
  bool rvq_q8 = false;

  std::array<seanet_layer_weights, k_max_seanet_layers> encoder_layers = {};
  conv_weights downsample = {};
  transformer_weights encoder_transformer = {};
  quantizer_weights quantizer = {};
  conv_weights upsample = {}; // depthwise transposed conv
  transformer_weights decoder_transformer = {};
  std::array<seanet_layer_weights, k_max_seanet_layers> decoder_layers = {};

  int32_t sample_rate = 0;
  int32_t frame_samples = 0; // 1920 at 24 kHz / 12.5 Hz
  int32_t n_q = 0;
  int32_t dim = 0;

  // Bound sizing contract (copied from the plan at bind) so per-frame
  // capacity guards are O(1) reads instead of model re-plans.
  uint64_t workspace_floats = 0;
  uint64_t frame_floats = 0;

  std::span<float> prepared_arena = {};
  uint64_t prepared_floats_used = 0;
};

// Streaming state layout: one contiguous caller-owned float arena, carved at
// prepare time into per-conv left-context rings, per-convtr overlap tails,
// and per-transformer rolling KV windows.
struct codec_streaming_state {
  std::span<float> arena = {};
  int64_t encoder_positions = 0; // frames seen by the encoder transformer
  int64_t decoder_positions = 0;
};

//------------------------------------------------------------------------------//
// Sizing (callers allocate one-time before any dispatch).
//------------------------------------------------------------------------------//

uint64_t required_prepared_floats(const emel::model::data &model_data) noexcept;

uint64_t required_state_floats(const emel::model::data &model_data) noexcept;

// Scratch for one frame of encode or decode across all stages (actors process
// exactly one 80 ms frame per dispatch).
uint64_t
required_workspace_floats(const emel::model::data &model_data) noexcept;

// Widest time-major stage buffer one frame flows through (the actor-owned
// io buffer must hold at least this many floats).
uint64_t required_frame_floats(const emel::model::data &model_data) noexcept;

//------------------------------------------------------------------------------//
// Binding (one-time, non-hot-path).
//------------------------------------------------------------------------------//

// Pure contract validator: a compile-time dry-run instantiation of the same
// walk bind_codec_runtime executes (identical tensor presence / dtype /
// element checks, no arena writes), so the guard-side predicate can never
// drift from the bind it authorizes. Capacity is validated separately via
// the required_* sizing contract.
bool validate_codec_contract(const emel::model::data &model_data) noexcept;

// Pure per-request validator for decode codes: every code addresses a valid
// codebook entry, and the request carries a valid codebook prefix. Bounded
// scan; guards route on this before the decode action runs.
bool validate_codes_in_range(const codec_runtime &runtime,
                             std::span<const int32_t> codes) noexcept;

// Canonicalizes weights into `prepared`, carves `state`, fills `runtime_out`.
// Non-failing by contract: the owning machine's guards route on
// validate_codec_contract and the required_* capacities before this runs.
void bind_codec_runtime(const emel::model::data &model_data,
                        std::span<float> prepared, std::span<float> state,
                        codec_runtime &runtime_out,
                        codec_streaming_state &state_out) noexcept;

void reset_streaming_state(const codec_runtime &runtime,
                           codec_streaming_state &state) noexcept;

//------------------------------------------------------------------------------//
// Compute (data-plane only; each helper executes one already-chosen,
// already-validated stage as bounded kernel op dispatches). Every entry is
// non-failing by contract - the owning machines' guards validate the bind
// contract, arena capacities, and request shapes before any compute action
// runs - so no helper output routes machine behavior.
//------------------------------------------------------------------------------//

struct frame_buffer {
  float *data = nullptr;
  int32_t channels = 0;
  int32_t length = 0; // time steps
};

// conv_f16 selects the reference f16 operand class (f16 im2col + ggml-exact
// f16 mul_mat over raw f16 weights); instantiated for both values in
// detail.cpp. The owning actors choose the instantiation via guard rows.
// Each stack is unrolled at compile time from its constexpr topology table,
// so no runtime block-kind selection exists in the compute layer.
template <bool conv_f16>
void compute_seanet_encoder(codec_runtime &runtime,
                            codec_streaming_state &state, frame_buffer &io,
                            std::span<float> workspace) noexcept;

template <bool conv_f16>
void compute_seanet_decoder(codec_runtime &runtime,
                            codec_streaming_state &state, frame_buffer &io,
                            std::span<float> workspace) noexcept;

// Downsample conv (25 Hz -> 12.5 Hz), the encode chain's final conv stage.
template <bool conv_f16>
void compute_encoder_downsample(codec_runtime &runtime,
                                codec_streaming_state &state, frame_buffer &io,
                                std::span<float> workspace) noexcept;

// Depthwise transposed-conv upsample (12.5 Hz -> 25 Hz), the decode chain's
// first stage; the owning actor selects it through its own transition, never
// by sniffing weight shapes here.
void compute_decoder_upsample(codec_runtime &runtime,
                              codec_streaming_state &state, frame_buffer &io,
                              std::span<float> workspace) noexcept;

template <bool proj_q8>
void compute_transformer(codec_runtime &runtime,
                         const transformer_weights &weights,
                         codec_streaming_state &state, int64_t &positions,
                         frame_buffer &io, std::span<float> workspace) noexcept;

// latent [dim, frames] -> codes [n_q, frames]
template <bool conv_f16, bool proj_q8 = false>
void compute_rvq_encode(codec_runtime &runtime, const frame_buffer &latent,
                        std::span<int32_t> codes_out,
                        std::span<float> workspace) noexcept;

// codes [active_n_q, frames] -> latent [dim, frames]
template <bool conv_f16, bool proj_q8 = false>
void compute_rvq_decode(codec_runtime &runtime, std::span<const int32_t> codes,
                        int32_t frames, frame_buffer &latent_out,
                        std::span<float> workspace) noexcept;

} // namespace emel::speech::codec::mimi::detail
