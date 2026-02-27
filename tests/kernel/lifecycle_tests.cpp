#include <doctest/doctest.h>

#include <algorithm>
#include <concepts>

#include "test_helpers.hpp"
#include "emel/kernel/any.hpp"
#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/aarch64/events.hpp"
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/cuda/actions.hpp"
#include "emel/kernel/cuda/events.hpp"
#include "emel/kernel/cuda/sm.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/metal/actions.hpp"
#include "emel/kernel/metal/events.hpp"
#include "emel/kernel/metal/sm.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/kernel/vulkan/actions.hpp"
#include "emel/kernel/vulkan/events.hpp"
#include "emel/kernel/vulkan/sm.hpp"
#include "emel/kernel/wasm/actions.hpp"
#include "emel/kernel/wasm/events.hpp"
#include "emel/kernel/wasm/sm.hpp"
#include "emel/kernel/x86_64/actions.hpp"
#include "emel/kernel/x86_64/events.hpp"
#include "emel/kernel/x86_64/sm.hpp"

namespace {

using kernel_sm = emel::kernel::sm;
using x86_64_sm = emel::kernel::x86_64::sm;
using aarch64_sm = emel::kernel::aarch64::sm;
using wasm_sm = emel::kernel::wasm::sm;
using cuda_sm = emel::kernel::cuda::sm;
using metal_sm = emel::kernel::metal::sm;
using vulkan_sm = emel::kernel::vulkan::sm;
using x86_64_dispatch_event = emel::kernel::x86_64::event::dispatch_request;
using aarch64_dispatch_event = emel::kernel::aarch64::event::dispatch_request;
using wasm_dispatch_event = emel::kernel::wasm::event::dispatch_request;
using cuda_dispatch_event = emel::kernel::cuda::event::dispatch_request;
using metal_dispatch_event = emel::kernel::metal::event::dispatch_request;
using vulkan_dispatch_event = emel::kernel::vulkan::event::dispatch_request;
using emel::kernel::test::dtype;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_smoke_op_event;
using emel::kernel::test::make_src;

template <class machine_type, class event_type>
concept has_public_process_event = requires(machine_type & machine, const event_type & ev) {
  { machine.process_event(ev) } -> std::convertible_to<bool>;
};

template <class machine_type>
void check_backend_op_paths(machine_type & machine,
                            const emel::kernel::event::op_dup & op_dup_ok,
                            const emel::kernel::event::op_dup & op_dup_invalid,
                            const emel::kernel::event::op_add & op_add_ok,
                            const emel::kernel::event::op_add & op_add_invalid,
                            const emel::kernel::event::op_mul & op_mul_ok,
                            const emel::kernel::event::op_mul & op_mul_invalid,
                            const emel::kernel::event::op_mul_mat & op_mul_mat_ok,
                            const emel::kernel::event::op_mul_mat & op_mul_mat_invalid,
                            const emel::kernel::event::op_soft_max & op_soft_max_ok,
                            const emel::kernel::event::op_soft_max & op_soft_max_invalid) {
  CHECK(machine.process_event(op_dup_ok));
  CHECK_FALSE(machine.process_event(op_dup_invalid));

  CHECK(machine.process_event(op_add_ok));
  CHECK_FALSE(machine.process_event(op_add_invalid));

  CHECK(machine.process_event(op_mul_ok));
  CHECK_FALSE(machine.process_event(op_mul_invalid));

  CHECK(machine.process_event(op_mul_mat_ok));
  CHECK_FALSE(machine.process_event(op_mul_mat_invalid));

  CHECK(machine.process_event(op_soft_max_ok));
  CHECK_FALSE(machine.process_event(op_soft_max_invalid));
}

}  // namespace

TEST_CASE("kernel_backends_accept_dispatch_event") {
  const emel::kernel::event::dispatch event{};

  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};
  kernel_sm kernel_machine{};
  emel::kernel::any any_machine{};

  CHECK(x86_64_machine.process_event(event));
  CHECK(aarch64_machine.process_event(event));
  CHECK(wasm_machine.process_event(event));
  CHECK(cuda_machine.process_event(event));
  CHECK(metal_machine.process_event(event));
  CHECK(vulkan_machine.process_event(event));
  CHECK(kernel_machine.process_event(event));
  CHECK(any_machine.process_event(event));
}

TEST_CASE("kernel_backends_expose_explicit_op_transitions") {
  float src0[8] = {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  float src1[8] = {3.0f, 2.0f, 1.0f, 0.0f, 3.0f, 2.0f, 1.0f, 0.0f};
  float dst[8] = {};

  const emel::kernel::event::op_dup op_dup_ok{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_add op_add_ok{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src1, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_mul op_mul_ok{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src1, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat op_mul_mat_ok{
      .src0 = make_src(src0, dtype::f32, 2, 2),
      .src1 = make_src(src1, dtype::f32, 2, 2),
      .dst = make_dst(dst, dtype::f32, 2, 2),
      .nth = 1,
  };
  const emel::kernel::event::op_soft_max op_soft_max_ok{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  emel::kernel::event::op_dup op_dup_invalid = op_dup_ok;
  op_dup_invalid.src0.data = nullptr;

  emel::kernel::event::op_add op_add_invalid = op_add_ok;
  op_add_invalid.src1.data = nullptr;

  emel::kernel::event::op_mul op_mul_invalid = op_mul_ok;
  op_mul_invalid.src1.data = nullptr;

  emel::kernel::event::op_mul_mat op_mul_mat_invalid = op_mul_mat_ok;
  op_mul_mat_invalid.src1.data = nullptr;

  emel::kernel::event::op_soft_max op_soft_max_invalid = op_soft_max_ok;
  op_soft_max_invalid.dst.ne[0] = 0;

  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};
  kernel_sm kernel_machine{};
  emel::kernel::any any_machine{};

  check_backend_op_paths(x86_64_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid,
                         op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(aarch64_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid,
                         op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(wasm_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid,
                         op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(cuda_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid,
                         op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(metal_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid,
                         op_soft_max_ok, op_soft_max_invalid);

  check_backend_op_paths(vulkan_machine, op_dup_ok, op_dup_invalid, op_add_ok, op_add_invalid,
                         op_mul_ok, op_mul_invalid, op_mul_mat_ok, op_mul_mat_invalid,
                         op_soft_max_ok, op_soft_max_invalid);

  CHECK(any_machine.process_event(op_add_ok));
  CHECK_FALSE(any_machine.process_event(op_add_invalid));
  CHECK(kernel_machine.process_event(op_add_ok));
  CHECK_FALSE(kernel_machine.process_event(op_add_invalid));
  CHECK(kernel_machine.process_event(emel::kernel::event::dispatch{}));
}

TEST_CASE("kernel_validation_branch_paths") {
  float src0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float src1[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  float dst[4] = {};

  emel::kernel::event::op_add ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src1, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  CHECK(emel::kernel::detail::validate_dispatch_request(ev));

  emel::kernel::event::op_add invalid = ev;
  invalid.src0.data = nullptr;
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));

  invalid = ev;
  invalid.src1.data = nullptr;
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));

  invalid = ev;
  invalid.dst.data = nullptr;
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));

  invalid = ev;
  invalid.nth = 0;
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));

  invalid = ev;
  invalid.ith = 1;
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));

  invalid = ev;
  invalid.op_params_size = static_cast<uint32_t>(invalid.op_params.size() + 1);
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));

  invalid = ev;
  invalid.dst.type = dtype::q4_0;
  CHECK_FALSE(emel::kernel::detail::validate_dispatch_request(invalid));
}

TEST_CASE("kernel_detail_negative_compute_paths") {
  float src0[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  float src1[8] = {4.0f, 3.0f, 2.0f, 1.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  float dst[8] = {};

  emel::kernel::event::op_dup dup_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 3),
      .nth = 1,
  };
  CHECK_FALSE(emel::kernel::detail::run_copy(dup_ev));

  emel::kernel::event::op_add add_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src1, dtype::f32, 3),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };
  CHECK_FALSE(emel::kernel::detail::run_binary(
      add_ev, [](const float lhs, const float rhs) { return lhs + rhs; }));

  emel::kernel::event::op_sqr sqr_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 3),
      .nth = 1,
  };
  CHECK_FALSE(emel::kernel::detail::run_unary(
      sqr_ev, [](const float value) { return value * value; }));

  emel::kernel::event::op_mul_mat mul_mat_ev{
      .src0 = make_src(src0, dtype::f32, 2, 2),
      .src1 = make_src(src1, dtype::f32, 2, 3),
      .dst = make_dst(dst, dtype::f32, 2, 2),
      .nth = 1,
  };
  CHECK_FALSE(emel::kernel::detail::run_mul_mat(mul_mat_ev));
  mul_mat_ev.src0.ne[0] = 0;
  CHECK_FALSE(emel::kernel::detail::run_mul_mat(mul_mat_ev));

  emel::kernel::event::op_soft_max soft_max_ev{
      .src0 = make_src(src0, dtype::f32, 0, 2),
      .dst = make_dst(dst, dtype::f32, 4, 2),
      .nth = 1,
  };
  CHECK_FALSE(emel::kernel::detail::run_soft_max(soft_max_ev));
}

TEST_CASE("kernel_detail_stride_paths_and_scalar_helpers") {
  float src0_storage[16] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f,
                            5.0f, 0.0f, 6.0f, 0.0f, 7.0f, 0.0f, 8.0f, 0.0f};
  float src1_storage[16] = {2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f, 0.0f,
                            6.0f, 0.0f, 7.0f, 0.0f, 8.0f, 0.0f, 9.0f, 0.0f};
  float dst_storage[16] = {};

  auto src0 = make_src(src0_storage, dtype::f32, 4);
  auto src1 = make_src(src1_storage, dtype::f32, 4);
  auto dst = make_dst(dst_storage, dtype::f32, 4);

  src0.nb[0] = sizeof(float) * 2;
  src0.nb[1] = src0.nb[0] * src0.ne[0];
  src1.nb[0] = sizeof(float) * 2;
  src1.nb[1] = src1.nb[0] * src1.ne[0];
  dst.nb[0] = sizeof(float) * 2;
  dst.nb[1] = dst.nb[0] * dst.ne[0];

  CHECK(emel::kernel::detail::has_valid_tensor_layout(src0));

  auto invalid = src0;
  invalid.nb[0] = 2;
  CHECK_FALSE(emel::kernel::detail::has_valid_tensor_layout(invalid));

  invalid = src0;
  invalid.ne[1] = 2;
  invalid.nb[1] = 0;
  CHECK_FALSE(emel::kernel::detail::has_valid_tensor_layout(invalid));

  auto default_stride = src0;
  default_stride.nb = {0, 0, 0, 0};
  CHECK(emel::kernel::detail::tensor_stride_bytes(default_stride, 0) == sizeof(float));
  CHECK(emel::kernel::detail::tensor_stride_bytes(default_stride, 1) == sizeof(float) * 4);

  auto zero_dim = src0;
  zero_dim.ne[2] = 0;
  (void) emel::kernel::detail::tensor_offset_bytes(zero_dim, static_cast<uint64_t>(1));

  const float read_before = emel::kernel::detail::read_f32(src0, 2);
  CHECK(read_before == doctest::Approx(3.0f));
  emel::kernel::detail::write_f32(dst, 2, 42.0f);
  CHECK(dst_storage[4] == doctest::Approx(42.0f));

  emel::kernel::event::op_add add_ev{
      .src0 = src0,
      .src1 = src1,
      .dst = dst,
      .nth = 1,
  };
  CHECK(emel::kernel::detail::run_binary(
      add_ev, [](const float lhs, const float rhs) { return lhs + rhs; }));
  CHECK(dst_storage[0] == doctest::Approx(3.0f));
  CHECK(dst_storage[2] == doctest::Approx(5.0f));
  CHECK(dst_storage[4] == doctest::Approx(7.0f));
  CHECK(dst_storage[6] == doctest::Approx(9.0f));

  emel::kernel::event::op_mul_mat mul_mat_ev{
      .src0 = make_src(src0_storage, dtype::f32, 2, 2),
      .src1 = make_src(src1_storage, dtype::f32, 2, 2),
      .dst = make_dst(dst_storage, dtype::f32, 2, 2),
      .nth = 1,
  };
  mul_mat_ev.src0.nb[0] = sizeof(float) * 2;
  mul_mat_ev.src0.nb[1] = mul_mat_ev.src0.nb[0] * mul_mat_ev.src0.ne[0];
  mul_mat_ev.src1.nb[0] = sizeof(float) * 2;
  mul_mat_ev.src1.nb[1] = mul_mat_ev.src1.nb[0] * mul_mat_ev.src1.ne[0];
  mul_mat_ev.dst.nb[0] = sizeof(float) * 2;
  mul_mat_ev.dst.nb[1] = mul_mat_ev.dst.nb[0] * mul_mat_ev.dst.ne[0];

  CHECK(emel::kernel::detail::run_mul_mat(mul_mat_ev));
  CHECK(dst_storage[0] == doctest::Approx(10.0f));
  CHECK(dst_storage[2] == doctest::Approx(13.0f));
  CHECK(dst_storage[4] == doctest::Approx(22.0f));
  CHECK(dst_storage[6] == doctest::Approx(29.0f));

  emel::kernel::event::op_soft_max soft_max_ev{
      .src0 = make_src(src0_storage, dtype::f32, 2, 2),
      .dst = make_dst(dst_storage, dtype::f32, 2, 2),
      .nth = 1,
  };
  soft_max_ev.src0.nb[0] = sizeof(float) * 2;
  soft_max_ev.src0.nb[1] = soft_max_ev.src0.nb[0] * soft_max_ev.src0.ne[0];
  soft_max_ev.dst.nb[0] = sizeof(float) * 2;
  soft_max_ev.dst.nb[1] = soft_max_ev.dst.nb[0] * soft_max_ev.dst.ne[0];

  CHECK(emel::kernel::detail::run_soft_max(soft_max_ev));
  CHECK((dst_storage[0] + dst_storage[2]) == doctest::Approx(1.0f));
  CHECK((dst_storage[4] + dst_storage[6]) == doctest::Approx(1.0f));

  emel::kernel::event::op_div div_ev{
      .src0 = make_src(src0_storage, dtype::f32, 4),
      .src1 = make_src(src1_storage, dtype::f32, 4),
      .dst = make_dst(dst_storage, dtype::f32, 4),
      .nth = 1,
  };
  emel::kernel::detail::execute_scalar_unchecked(div_ev);
  CHECK(dst_storage[0] == doctest::Approx(0.5f));

  const emel::kernel::event::op_sum unsupported_ev{
      .src0 = make_src(src0_storage, dtype::f32, 4),
      .dst = make_dst(dst_storage, dtype::f32, 4),
      .nth = 1,
  };
  CHECK_FALSE(emel::kernel::detail::execute_scalar(unsupported_ev));
}

TEST_CASE("kernel_backends_reject_quantized_dispatch_dtypes") {
  float src0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float src1[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  float dst[4] = {};

  const emel::kernel::event::op_add quantized{
      .src0 = make_src(src0, dtype::q4_0, 4),
      .src1 = make_src(src1, dtype::q4_0, 4),
      .dst = make_dst(dst, dtype::q4_0, 4),
      .nth = 1,
  };

  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};

  CHECK_FALSE(x86_64_machine.process_event(quantized));
  CHECK_FALSE(aarch64_machine.process_event(quantized));
  CHECK_FALSE(wasm_machine.process_event(quantized));
  CHECK_FALSE(cuda_machine.process_event(quantized));
  CHECK_FALSE(metal_machine.process_event(quantized));
  CHECK_FALSE(vulkan_machine.process_event(quantized));
}

TEST_CASE("kernel_backend_unexpected_actions_mark_backend_error") {
  const emel::kernel::event::dispatch request{};

  emel::kernel::x86_64::action::context x86_64_ctx{};
  emel::kernel::aarch64::action::context aarch64_ctx{};
  emel::kernel::wasm::action::context wasm_ctx{};
  emel::kernel::cuda::action::context cuda_ctx{};
  emel::kernel::metal::action::context metal_ctx{};
  emel::kernel::vulkan::action::context vulkan_ctx{};

  emel::kernel::x86_64::event::dispatch_ctx x86_64_dispatch_ctx{};
  emel::kernel::aarch64::event::dispatch_ctx aarch64_dispatch_ctx{};
  emel::kernel::wasm::event::dispatch_ctx wasm_dispatch_ctx{};
  emel::kernel::cuda::event::dispatch_ctx cuda_dispatch_ctx{};
  emel::kernel::metal::event::dispatch_ctx metal_dispatch_ctx{};
  emel::kernel::vulkan::event::dispatch_ctx vulkan_dispatch_ctx{};

  const emel::kernel::x86_64::event::dispatch_request x86_64_dispatch{request,
                                                                       x86_64_dispatch_ctx};
  const emel::kernel::aarch64::event::dispatch_request aarch64_dispatch{request,
                                                                         aarch64_dispatch_ctx};
  const emel::kernel::wasm::event::dispatch_request wasm_dispatch{request, wasm_dispatch_ctx};
  const emel::kernel::cuda::event::dispatch_request cuda_dispatch{request, cuda_dispatch_ctx};
  const emel::kernel::metal::event::dispatch_request metal_dispatch{request, metal_dispatch_ctx};
  const emel::kernel::vulkan::event::dispatch_request vulkan_dispatch{request,
                                                                       vulkan_dispatch_ctx};

  emel::kernel::x86_64::action::on_unexpected(x86_64_dispatch, x86_64_ctx);
  emel::kernel::aarch64::action::on_unexpected(aarch64_dispatch, aarch64_ctx);
  emel::kernel::wasm::action::on_unexpected(wasm_dispatch, wasm_ctx);
  emel::kernel::cuda::action::on_unexpected(cuda_dispatch, cuda_ctx);
  emel::kernel::metal::action::on_unexpected(metal_dispatch, metal_ctx);
  emel::kernel::vulkan::action::on_unexpected(vulkan_dispatch, vulkan_ctx);

  CHECK(x86_64_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(aarch64_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(wasm_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(cuda_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(metal_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));
  CHECK(vulkan_dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::error::internal_error)));

  CHECK(x86_64_dispatch_ctx.outcome == emel::kernel::x86_64::events::phase_outcome::failed);
  CHECK(aarch64_dispatch_ctx.outcome == emel::kernel::aarch64::events::phase_outcome::failed);
  CHECK(wasm_dispatch_ctx.outcome == emel::kernel::wasm::events::phase_outcome::failed);
  CHECK(cuda_dispatch_ctx.outcome == emel::kernel::cuda::events::phase_outcome::failed);
  CHECK(metal_dispatch_ctx.outcome == emel::kernel::metal::events::phase_outcome::failed);
  CHECK(vulkan_dispatch_ctx.outcome == emel::kernel::vulkan::events::phase_outcome::failed);
}

TEST_CASE("kernel_backends_cover_all_ggml_ops") {
  x86_64_sm x86_64_machine{};
  aarch64_sm aarch64_machine{};
  wasm_sm wasm_machine{};
  cuda_sm cuda_machine{};
  metal_sm metal_machine{};
  vulkan_sm vulkan_machine{};
  kernel_sm kernel_machine{};
  emel::kernel::any any_machine{};

#define EMEL_KERNEL_CHECK_ALL_BACKENDS(op_name)                                             \
  {                                                                                           \
    const auto ev = make_smoke_op_event<emel::kernel::event::op_name>();                     \
    const bool scalar_supported = emel::kernel::detail::can_execute_scalar(ev);              \
    CHECK(x86_64_machine.process_event(ev) == scalar_supported);                              \
    CHECK(aarch64_machine.process_event(ev) == scalar_supported);                             \
    CHECK(wasm_machine.process_event(ev));                                                    \
    CHECK(cuda_machine.process_event(ev));                                                    \
    CHECK(metal_machine.process_event(ev));                                                   \
    CHECK(vulkan_machine.process_event(ev));                                                  \
    CHECK(kernel_machine.process_event(ev) == scalar_supported);                              \
    CHECK(any_machine.process_event(ev) == scalar_supported);                                 \
  }
  EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_CHECK_ALL_BACKENDS)
#undef EMEL_KERNEL_CHECK_ALL_BACKENDS
}

TEST_CASE("kernel_wrapper_exposes_public_dispatch_entrypoints") {
  CHECK((has_public_process_event<kernel_sm, emel::kernel::event::dispatch>));
  CHECK((has_public_process_event<kernel_sm, emel::kernel::event::op_add>));
}

TEST_CASE("kernel_backend_wrappers_hide_internal_dispatch_entrypoints") {
  CHECK_FALSE((has_public_process_event<x86_64_sm, x86_64_dispatch_event>));
  CHECK_FALSE((has_public_process_event<aarch64_sm, aarch64_dispatch_event>));
  CHECK_FALSE((has_public_process_event<wasm_sm, wasm_dispatch_event>));
  CHECK_FALSE((has_public_process_event<cuda_sm, cuda_dispatch_event>));
  CHECK_FALSE((has_public_process_event<metal_sm, metal_dispatch_event>));
  CHECK_FALSE((has_public_process_event<vulkan_sm, vulkan_dispatch_event>));
}
