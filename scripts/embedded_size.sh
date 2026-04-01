#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${EMEL_EMBEDDED_SIZE_BUILD_ROOT:-$ROOT_DIR/build/embedded_size}"
REFERENCE_REPOSITORY="${EMEL_EMBEDDED_SIZE_REFERENCE_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"

USE_ZIG=true
JSON_OUTPUT=false
SNAPSHOT_UPDATE=false
REFERENCE_REF="${EMEL_EMBEDDED_SIZE_REF:-}"
SNAPSHOT_PATH="${EMEL_EMBEDDED_SIZE_SNAPSHOT_PATH:-$ROOT_DIR/snapshots/embedded_size/summary.txt}"

BUILD_TYPE="MinSizeRel"
WORKLOAD_NAME="kernel_suite_v1"
MEASUREMENT_MODE="linked_executable"
MEASUREMENT_SCOPE="kernel_runtime"

usage() {
  cat <<'USAGE'
usage: scripts/embedded_size.sh [--zig|--system] [--ref=<git-ref>] [--json] [--snapshot-update]

Build emel and llama.cpp/ggml probe executables with embedded-oriented size flags
and compare the final dead-stripped linked binaries.

Output fields:
  raw_bytes       on-disk executable bytes before stripping
  stripped_bytes  on-disk executable bytes after strip in a temporary copy
  section_bytes   live code/data section bytes reported by `size`

Notes:
  - This measures final linked executables for a matched kernel workload, not
    static archives.
  - The probe is narrower than the full llama.cpp feature surface.
  - The final executable still uses the platform-default C/C++ runtime; this is
    not a fully freestanding firmware image.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --zig) USE_ZIG=true ;;
    --system) USE_ZIG=false ;;
    --json) JSON_OUTPUT=true ;;
    --snapshot-update) SNAPSHOT_UPDATE=true ;;
    --ref=*) REFERENCE_REF="${arg#--ref=}" ;;
    --snapshot-path=*) SNAPSHOT_PATH="${arg#--snapshot-path=}" ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      usage
      exit 1
      ;;
  esac
done

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
}

for tool in cmake ninja git size strip; do
  require_tool "$tool"
done

prepare_toolchain() {
  bench_cc="${EMBEDDED_SIZE_CC:-cc}"
  bench_cxx="${EMBEDDED_SIZE_CXX:-c++}"
  bench_cc_arg=""
  bench_cxx_arg=""
  bench_asm_arg=""
  if $USE_ZIG; then
    require_tool zig
    bench_cc="$(command -v zig)"
    bench_cxx="$bench_cc"
    bench_cc_arg="cc"
    bench_cxx_arg="c++"
    bench_asm_arg="cc"
  fi
}

if [[ -z "$REFERENCE_REF" ]]; then
  ref_file="$ROOT_DIR/tools/bench/reference_ref.txt"
  if [[ -f "$ref_file" ]]; then
    REFERENCE_REF="$(head -n 1 "$ref_file" | tr -d '[:space:]')"
  fi
fi

if [[ -z "$REFERENCE_REF" ]]; then
  REFERENCE_REF="master"
fi

platform="$(uname -s)"
host_arch="$(uname -m)"
strip_args=()
linker_gc_flags=()
case "$platform" in
  Darwin)
    strip_args=(-S -x)
    linker_gc_flags=(-Wl,-dead_strip)
    ;;
  *)
    strip_args=(--strip-debug --strip-unneeded)
    linker_gc_flags=(-Wl,--gc-sections)
    ;;
esac

case "$host_arch" in
  arm64|aarch64) probe_backend="aarch64" ;;
  x86_64|amd64) probe_backend="x86_64" ;;
  *)
    echo "error: unsupported host architecture for embedded probe: $host_arch" >&2
    exit 1
    ;;
esac

common_compile_flags="-ffunction-sections -fdata-sections"
common_compile_flags_csv="-ffunction-sections,-fdata-sections"
linker_gc_flags_csv="$(IFS=,; printf '%s' "${linker_gc_flags[*]}")"

prepare_reference_checkout() {
  local ref_dir="$1"
  mkdir -p "$(dirname "$ref_dir")"
  if [[ -d "$ref_dir/.git" ]]; then
    git -C "$ref_dir" fetch --tags origin
  else
    rm -rf "$ref_dir"
    git clone "$REFERENCE_REPOSITORY" "$ref_dir"
  fi
  git -C "$ref_dir" checkout --detach "$REFERENCE_REF"
  REFERENCE_REF="$(git -C "$ref_dir" rev-parse HEAD)"
}

raw_bytes() {
  wc -c < "$1" | awk '{print $1}'
}

stripped_bytes() {
  local input_path="$1"
  local temp_path="$2"
  local raw_size
  local stripped_size
  raw_size="$(raw_bytes "$input_path")"
  cp "$input_path" "$temp_path"
  if ! strip "${strip_args[@]}" "$temp_path" >/dev/null 2>&1; then
    printf '%s\n' "$raw_size"
    return
  fi
  stripped_size="$(raw_bytes "$temp_path")"
  if [[ "$stripped_size" -gt "$raw_size" ]]; then
    printf '%s\n' "$raw_size"
    return
  fi
  printf '%s\n' "$stripped_size"
}

section_bytes() {
  local input_path="$1"
  case "$platform" in
    Darwin)
      size -m "$input_path" | awk '
        /^Segment __PAGEZERO:/ { skip = 1; next }
        /^Segment __LINKEDIT:/ { skip = 1; next }
        /^Segment / { skip = 0; next }
        /^[[:space:]]*total / { if (!skip) sum += $2 }
        END { print sum + 0 }'
      ;;
    *)
      size "$input_path" | awk 'NR > 1 && NF >= 6 { sum += $1 + $2 + $3 } END { print sum + 0 }'
      ;;
  esac
}

write_emel_probe_source() {
  local path="$1"
  cat > "$path" <<'EOF'
#include <cmath>
#include <cstdint>
#include <vector>

#include "emel/kernel/events.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
#include "emel/kernel/aarch64/sm.hpp"
using backend_sm = emel::kernel::aarch64::sm;
#elif defined(__x86_64__) || defined(_M_X64)
#include "emel/kernel/x86_64/sm.hpp"
using backend_sm = emel::kernel::x86_64::sm;
#else
#error "unsupported backend for embedded size probe"
#endif

using emel::kernel::event::dtype;

namespace {

constexpr int64_t k_vec_len = 1024;
constexpr int64_t k_softmax_width = 128;
constexpr int64_t k_softmax_rows = 8;
constexpr int64_t k_mm_k = 64;
constexpr int64_t k_mm_m = 32;
constexpr int64_t k_mm_n = 48;

std::vector<float> make_signed_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const float wave = std::sin(static_cast<float>(i) * 0.013f) * scale;
    const float bucket = static_cast<float>((i % 29) - 14) * 0.03125f;
    out[static_cast<size_t>(i)] = wave + bucket + bias;
  }
  return out;
}

std::vector<float> make_positive_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out = make_signed_data(count, scale, bias);
  for (float & value : out) {
    value = std::fabs(value) + 0.5f;
  }
  return out;
}

template <class tensor_type>
void fill_default_nb(tensor_type & tensor) {
  constexpr uint64_t elem_size = sizeof(float);
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

emel::kernel::event::tensor_view make_src_view(const float * data,
                                               const uint64_t ne0,
                                               const uint64_t ne1 = 1,
                                               const uint64_t ne2 = 1,
                                               const uint64_t ne3 = 1) {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = dtype::f32;
  tensor.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(tensor);
  return tensor;
}

emel::kernel::event::tensor_view_mut make_dst_view(float * data,
                                                   const uint64_t ne0,
                                                   const uint64_t ne1 = 1,
                                                   const uint64_t ne2 = 1,
                                                   const uint64_t ne3 = 1) {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = dtype::f32;
  tensor.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(tensor);
  return tensor;
}

}  // namespace

int main() {
  backend_sm machine{};
  auto exec = [&](const auto & ev) {
    return machine.process_event(ev);
  };

  volatile float sink = 0.0f;

  {
    auto src = make_signed_data(k_vec_len, 1.25f, 0.1f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_dup ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[0];
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_add ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[1];
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sub ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[2];
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.25f, 0.75f);
    auto rhs = make_signed_data(k_vec_len, 0.45f, 0.5f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_mul ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[3];
  }

  {
    auto lhs = make_positive_data(k_vec_len, 0.3f, 0.25f);
    auto rhs = make_positive_data(k_vec_len, 0.2f, 0.75f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_div ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[4];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.5f, 0.125f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sqr ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[5];
  }

  {
    auto src = make_positive_data(k_vec_len, 0.35f, 0.2f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sqrt ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[6];
  }

  {
    auto src = make_positive_data(k_vec_len, 0.4f, 0.125f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_log ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[7];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, 0.1f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sin ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[8];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, -0.2f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_cos ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[9];
  }

  {
    auto src = make_signed_data(k_softmax_width * k_softmax_rows, 0.1f, 0.05f);
    std::vector<float> dst(static_cast<size_t>(k_softmax_width * k_softmax_rows));
    emel::kernel::event::op_soft_max ev{
      .src0 = make_src_view(src.data(),
                            static_cast<uint64_t>(k_softmax_width),
                            static_cast<uint64_t>(k_softmax_rows)),
      .dst = make_dst_view(dst.data(),
                           static_cast<uint64_t>(k_softmax_width),
                           static_cast<uint64_t>(k_softmax_rows)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[static_cast<size_t>(k_softmax_width)];
  }

  {
    auto src0 = make_signed_data(k_mm_k * k_mm_m, 0.12f, 0.25f);
    auto matrix_a = make_signed_data(k_mm_k * k_mm_n, 0.08f, -0.1f);
    std::vector<float> src1(static_cast<size_t>(k_mm_k * k_mm_n));
    for (int64_t p = 0; p < k_mm_k; ++p) {
      for (int64_t j = 0; j < k_mm_n; ++j) {
        src1[static_cast<size_t>(p * k_mm_n + j)] = matrix_a[static_cast<size_t>(j * k_mm_k + p)];
      }
    }
    std::vector<float> dst(static_cast<size_t>(k_mm_n * k_mm_m));
    emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(src0.data(),
                            static_cast<uint64_t>(k_mm_k),
                            static_cast<uint64_t>(k_mm_m)),
      .src1 = make_src_view(src1.data(),
                            static_cast<uint64_t>(k_mm_n),
                            static_cast<uint64_t>(k_mm_k)),
      .dst = make_dst_view(dst.data(),
                           static_cast<uint64_t>(k_mm_n),
                           static_cast<uint64_t>(k_mm_m)),
      .nth = 1,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[0];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::neg,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[10];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::relu,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[11];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.35f, 0.1f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::exp,
    };
    if (!exec(ev)) {
      return 1;
    }
    sink += dst[12];
  }

  return sink == 0.0f ? 1 : 0;
}
EOF
}

write_reference_probe_source() {
  local path="$1"
  cat > "$path" <<'EOF'
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "llama.h"

namespace {

constexpr int64_t k_vec_len = 1024;
constexpr int64_t k_softmax_width = 128;
constexpr int64_t k_softmax_rows = 8;
constexpr int64_t k_mm_k = 64;
constexpr int64_t k_mm_m = 32;
constexpr int64_t k_mm_n = 48;

std::vector<float> make_signed_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const float wave = std::sin(static_cast<float>(i) * 0.013f) * scale;
    const float bucket = static_cast<float>((i % 29) - 14) * 0.03125f;
    out[static_cast<size_t>(i)] = wave + bucket + bias;
  }
  return out;
}

std::vector<float> make_positive_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out = make_signed_data(count, scale, bias);
  for (float & value : out) {
    value = std::fabs(value) + 0.5f;
  }
  return out;
}

struct ggml_graph_case {
  std::vector<uint8_t> arena;
  ggml_context * ctx = nullptr;
  ggml_tensor * out = nullptr;
  ggml_cgraph * graph = nullptr;

  explicit ggml_graph_case(const size_t arena_bytes = 64u * 1024u * 1024u)
      : arena(arena_bytes) {
    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = false;
    ctx = ggml_init(params);
  }

  ~ggml_graph_case() {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }

  bool compute() const {
    if (ctx == nullptr || graph == nullptr) {
      return false;
    }
    return ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
  }
};

void set_tensor_f32(ggml_tensor * tensor, const std::vector<float> & values) {
  std::memcpy(ggml_get_data_f32(tensor), values.data(), values.size() * sizeof(float));
}

bool finalize_graph(ggml_graph_case & c, ggml_tensor * out) {
  c.out = out;
  c.graph = ggml_new_graph(c.ctx);
  if (c.graph == nullptr || c.out == nullptr) {
    return false;
  }
  ggml_build_forward_expand(c.graph, c.out);
  return c.compute();
}

}  // namespace

int main() {
  llama_backend_init();
  volatile float sink = 0.0f;

  {
    auto src = make_signed_data(k_vec_len, 1.25f, 0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_dup(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[0];
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    if (!finalize_graph(c, ggml_add(c.ctx, a, b)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[1];
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    if (!finalize_graph(c, ggml_sub(c.ctx, a, b)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[2];
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.25f, 0.75f);
    auto rhs = make_signed_data(k_vec_len, 0.45f, 0.5f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    if (!finalize_graph(c, ggml_mul(c.ctx, a, b)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[3];
  }

  {
    auto lhs = make_positive_data(k_vec_len, 0.3f, 0.25f);
    auto rhs = make_positive_data(k_vec_len, 0.2f, 0.75f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    if (!finalize_graph(c, ggml_div(c.ctx, a, b)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[4];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.5f, 0.125f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_sqr(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[5];
  }

  {
    auto src = make_positive_data(k_vec_len, 0.35f, 0.2f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_sqrt(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[6];
  }

  {
    auto src = make_positive_data(k_vec_len, 0.4f, 0.125f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_log(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[7];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, 0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_sin(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[8];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, -0.2f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_cos(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[9];
  }

  {
    auto src = make_signed_data(k_softmax_width * k_softmax_rows, 0.1f, 0.05f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_softmax_width, k_softmax_rows);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_soft_max(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[static_cast<size_t>(k_softmax_width)];
  }

  {
    auto matrix_b = make_signed_data(k_mm_k * k_mm_m, 0.12f, 0.25f);
    auto matrix_a = make_signed_data(k_mm_k * k_mm_n, 0.08f, -0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_mm_k, k_mm_n);
    ggml_tensor * b = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_mm_k, k_mm_m);
    set_tensor_f32(a, matrix_a);
    set_tensor_f32(b, matrix_b);
    if (!finalize_graph(c, ggml_mul_mat(c.ctx, a, b)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[0];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_neg(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[10];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_relu(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[11];
  }

  {
    auto src = make_signed_data(k_vec_len, 0.35f, 0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    if (!finalize_graph(c, ggml_exp(c.ctx, a)) || !c.compute()) {
      llama_backend_free();
      return 1;
    }
    sink += ggml_get_data_f32(c.out)[12];
  }

  llama_backend_free();
  return sink == 0.0f ? 1 : 0;
}
EOF
}

write_emel_probe_cmakelists() {
  local path="$1"
  cat > "$path" <<EOF
cmake_minimum_required(VERSION 3.20)
project(emel_embedded_probe_project C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${ROOT_DIR}" emel_root)

add_executable(emel_embedded_probe
  "\${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp"
)

target_link_libraries(emel_embedded_probe
  PRIVATE
    emel_core
    emel
)
EOF
}

write_reference_probe_cmakelists() {
  local path="$1"
  cat > "$path" <<EOF
cmake_minimum_required(VERSION 3.20)
project(reference_embedded_probe_project C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory("${reference_checkout_dir}" reference_impl)

add_executable(reference_embedded_probe
  "\${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp"
)

target_link_libraries(reference_embedded_probe
  PRIVATE
    llama
    ggml
)

target_include_directories(reference_embedded_probe
  PRIVATE
    "${reference_checkout_dir}/include"
    "${reference_checkout_dir}/ggml/include"
)
EOF
}

configure_cmake_project() {
  local source_dir="$1"
  local build_dir="$2"
  local with_asm="$3"
  shift 2
  shift 1

  local cmake_args=(
    -S "$source_dir"
    -B "$build_dir"
    -G Ninja
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
    "-DCMAKE_C_FLAGS=$common_compile_flags"
    "-DCMAKE_CXX_FLAGS=$common_compile_flags"
    "-DCMAKE_EXE_LINKER_FLAGS=${linker_gc_flags[*]}"
    "-DCMAKE_C_COMPILER=$bench_cc"
    "-DCMAKE_CXX_COMPILER=$bench_cxx"
  )
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ "$with_asm" == "with_asm" && -n "$bench_asm_arg" ]]; then
    cmake_args+=("-DCMAKE_ASM_COMPILER=$bench_cc")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_asm_arg")
  fi
  cmake_args+=("$@")

  cmake "${cmake_args[@]}"
}

prepare_toolchain

reference_checkout_dir="$BUILD_ROOT/reference-src"
emel_probe_source_dir="$BUILD_ROOT/emel_probe_src"
emel_probe_build_dir="$BUILD_ROOT/emel_probe_build"
reference_probe_source_dir="$BUILD_ROOT/reference_probe_src"
reference_probe_build_dir="$BUILD_ROOT/reference_probe_build"
temp_strip_dir="$BUILD_ROOT/stripped"

mkdir -p "$BUILD_ROOT" "$temp_strip_dir" "$emel_probe_source_dir" "$reference_probe_source_dir"
prepare_reference_checkout "$reference_checkout_dir"

write_emel_probe_source "$emel_probe_source_dir/probe.cpp"
write_emel_probe_cmakelists "$emel_probe_source_dir/CMakeLists.txt"
write_reference_probe_source "$reference_probe_source_dir/probe.cpp"
write_reference_probe_cmakelists "$reference_probe_source_dir/CMakeLists.txt"

configure_cmake_project "$emel_probe_source_dir" "$emel_probe_build_dir" without_asm
cmake --build "$emel_probe_build_dir" --parallel --target emel_embedded_probe

configure_cmake_project "$reference_probe_source_dir" "$reference_probe_build_dir" with_asm
cmake --build "$reference_probe_build_dir" --parallel --target reference_embedded_probe

emel_binary="$emel_probe_build_dir/emel_embedded_probe"
reference_binary="$reference_probe_build_dir/reference_embedded_probe"

if [[ ! -f "$emel_binary" ]]; then
  echo "error: missing built emel probe executable: $emel_binary" >&2
  exit 1
fi

if [[ ! -f "$reference_binary" ]]; then
  echo "error: missing built reference probe executable: $reference_binary" >&2
  exit 1
fi

emel_raw_total="$(raw_bytes "$emel_binary")"
emel_stripped_total="$(stripped_bytes "$emel_binary" "$temp_strip_dir/emel_embedded_probe")"
emel_section_total="$(section_bytes "$emel_binary")"

reference_raw_total="$(raw_bytes "$reference_binary")"
reference_stripped_total="$(stripped_bytes "$reference_binary" "$temp_strip_dir/reference_embedded_probe")"
reference_section_total="$(section_bytes "$reference_binary")"

raw_ratio="$(awk -v a="$emel_raw_total" -v b="$reference_raw_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"
stripped_ratio="$(awk -v a="$emel_stripped_total" -v b="$reference_stripped_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"
section_ratio="$(awk -v a="$emel_section_total" -v b="$reference_section_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"

if $SNAPSHOT_UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT_PATH")"
  {
    printf '# embedded_size_config: reference_ref=%s toolchain=%s build_type=%s compile_flags=%s\n' \
      "$REFERENCE_REF" "$bench_cxx" "$BUILD_TYPE" "$common_compile_flags_csv"
    printf '# embedded_size_measurement: mode=%s scope=%s workload=%s backend=%s link_flags=%s\n' \
      "$MEASUREMENT_MODE" "$MEASUREMENT_SCOPE" "$WORKLOAD_NAME" "$probe_backend" "$linker_gc_flags_csv"
    printf '# embedded_size_emel: raw_bytes=%s stripped_bytes=%s section_bytes=%s binary=%s\n' \
      "$emel_raw_total" "$emel_stripped_total" "$emel_section_total" "${emel_binary#$ROOT_DIR/}"
    printf '# embedded_size_reference: raw_bytes=%s stripped_bytes=%s section_bytes=%s binary=%s\n' \
      "$reference_raw_total" "$reference_stripped_total" "$reference_section_total" \
      "${reference_binary#$ROOT_DIR/}"
    printf '# embedded_size_ratio: raw=%s stripped=%s section=%s\n' \
      "$raw_ratio" "$stripped_ratio" "$section_ratio"
  } > "$SNAPSHOT_PATH"
fi

if $JSON_OUTPUT; then
  printf '{\n'
  printf '  "mode": "%s",\n' "$MEASUREMENT_MODE"
  printf '  "scope": "%s",\n' "$MEASUREMENT_SCOPE"
  printf '  "workload": "%s",\n' "$WORKLOAD_NAME"
  printf '  "backend": "%s",\n' "$probe_backend"
  printf '  "reference_ref": "%s",\n' "$REFERENCE_REF"
  printf '  "toolchain": "%s",\n' "$bench_cxx"
  printf '  "build_type": "%s",\n' "$BUILD_TYPE"
  printf '  "compile_flags": "%s",\n' "$common_compile_flags_csv"
  printf '  "link_flags": "%s",\n' "$linker_gc_flags_csv"
  printf '  "emel": {\n'
  printf '    "raw_bytes": %s,\n' "$emel_raw_total"
  printf '    "stripped_bytes": %s,\n' "$emel_stripped_total"
  printf '    "section_bytes": %s,\n' "$emel_section_total"
  printf '    "binary": "%s"\n' "${emel_binary#$ROOT_DIR/}"
  printf '  },\n'
  printf '  "reference": {\n'
  printf '    "raw_bytes": %s,\n' "$reference_raw_total"
  printf '    "stripped_bytes": %s,\n' "$reference_stripped_total"
  printf '    "section_bytes": %s,\n' "$reference_section_total"
  printf '    "binary": "%s"\n' "${reference_binary#$ROOT_DIR/}"
  printf '  },\n'
  printf '  "ratio": {\n'
  printf '    "raw": %s,\n' "$raw_ratio"
  printf '    "stripped": %s,\n' "$stripped_ratio"
  printf '    "section": %s\n' "$section_ratio"
  printf '  }\n'
  printf '}\n'
  exit 0
fi

printf 'Embedded Linked Executable Comparison\n'
printf 'reference_ref: %s\n' "$REFERENCE_REF"
printf 'toolchain: %s\n' "$bench_cxx"
printf 'build_type: %s\n' "$BUILD_TYPE"
printf 'compile_flags: %s\n' "$common_compile_flags"
printf 'link_flags: %s\n' "${linker_gc_flags[*]}"
printf 'backend: %s\n' "$probe_backend"
printf 'workload: %s\n' "$WORKLOAD_NAME"
printf '\n'

printf 'emel\n'
printf '  raw_bytes: %s\n' "$emel_raw_total"
printf '  stripped_bytes: %s\n' "$emel_stripped_total"
printf '  section_bytes: %s\n' "$emel_section_total"
printf '  binary: %s\n' "${emel_binary#$ROOT_DIR/}"
printf '\n'
printf 'reference (llama.cpp/ggml)\n'
printf '  raw_bytes: %s\n' "$reference_raw_total"
printf '  stripped_bytes: %s\n' "$reference_stripped_total"
printf '  section_bytes: %s\n' "$reference_section_total"
printf '  binary: %s\n' "${reference_binary#$ROOT_DIR/}"
printf '\n'
printf 'ratios\n'
printf '  raw: %sx\n' "$raw_ratio"
printf '  stripped: %sx\n' "$stripped_ratio"
printf '  section: %sx\n' "$section_ratio"
printf '\n'
printf 'notes\n'
printf '  - this is a final linked executable measurement for a matched kernel workload, not a static archive estimate.\n'
printf '  - emel is measured through direct C++ usage of its kernel state machine; the reference executable uses llama.cpp/ggml runtime entrypoints and ggml kernel ops.\n'
printf '  - the probe does not represent full feature parity with llama.cpp, and the final binary still includes the platform-default runtime selected by the toolchain.\n'
if $SNAPSHOT_UPDATE; then
  printf '\n'
  printf 'snapshot\n'
  printf '  - updated: %s\n' "${SNAPSHOT_PATH#$ROOT_DIR/}"
fi
