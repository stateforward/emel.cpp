#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/kernel/detail.hpp"
#include "emel/speech/predictor/moshi/executor/detail.hpp"

namespace {

namespace detail = emel::speech::predictor::moshi::executor::detail;

detail::dtype dtype(const uint8_t code) {
  return static_cast<detail::dtype>(code);
}

} // namespace

TEST_CASE("speech Moshi executor binds supported dense and quantized views") {
  std::array<float, 16> dense_data = {};
  emel::model::data::tensor_record tensor{};
  tensor.data = dense_data.data();
  tensor.n_dims = 2;
  tensor.dims[0] = 4;
  tensor.dims[1] = 4;
  tensor.type = emel::kernel::detail::dtype_f32;
  detail::tensor_view view{};

  REQUIRE(detail::bind_tensor_view(tensor, view));
  CHECK(view.data == dense_data.data());
  CHECK(view.ne[0] == 4u);
  CHECK(view.ne[1] == 4u);
  CHECK(view.nb[0] == sizeof(float));

  std::array<uint8_t, 256> quantized_data = {};
  tensor.data = quantized_data.data();
  tensor.dims[0] = 32;
  tensor.dims[1] = 2;
  tensor.type = emel::kernel::detail::dtype_q4_0;
  REQUIRE(detail::bind_tensor_view(tensor, view));
  CHECK(view.nb[0] == 1u);
  CHECK(view.nb[1] > 0u);

  tensor.type = emel::kernel::detail::dtype_q4_k_x8_bl4;
  tensor.dims[0] = 256;
  tensor.dims[1] = 8;
  REQUIRE(detail::bind_tensor_view(tensor, view));
  CHECK(view.nb[1] > 0u);
  CHECK(view.nb[2] > 0u);

  tensor.dims[0] = 255;
  CHECK_FALSE(detail::bind_tensor_view(tensor, view));

  tensor.type = emel::kernel::detail::dtype_q4_0;
  tensor.dims[0] = 1;
  tensor.dims[1] = 2;
  CHECK_FALSE(detail::bind_tensor_view(tensor, view));

  tensor.data = nullptr;
  CHECK_FALSE(detail::bind_tensor_view(tensor, view));
}

TEST_CASE("speech Moshi executor classifies supported operand dtypes") {
  CHECK(
      detail::supported_get_rows_dtype(dtype(emel::kernel::detail::dtype_f32)));
  CHECK(
      detail::supported_get_rows_dtype(dtype(emel::kernel::detail::dtype_f16)));
  CHECK(detail::supported_get_rows_dtype(
      dtype(emel::kernel::detail::dtype_bf16)));
  CHECK(detail::supported_get_rows_dtype(
      dtype(emel::kernel::detail::dtype_q4_0)));
  CHECK(detail::supported_get_rows_dtype(
      dtype(emel::kernel::detail::dtype_q8_0)));
  CHECK(detail::supported_get_rows_dtype(
      dtype(emel::kernel::detail::dtype_q4_k)));
  CHECK_FALSE(
      detail::supported_get_rows_dtype(dtype(emel::kernel::detail::dtype_i32)));

  CHECK(
      detail::supported_argmax_dtype(dtype(emel::kernel::detail::dtype_q4_k)));
  CHECK(detail::supported_argmax_dtype(
      dtype(emel::kernel::detail::dtype_q4_k_x8_bl4)));
  CHECK_FALSE(
      detail::supported_argmax_dtype(dtype(emel::kernel::detail::dtype_i32)));

  CHECK(
      detail::supported_mul_mat_dtype(dtype(emel::kernel::detail::dtype_f16)));
  CHECK(
      detail::supported_mul_mat_dtype(dtype(emel::kernel::detail::dtype_q8_0)));
  CHECK(detail::supported_mul_mat_dtype(
      dtype(emel::kernel::detail::dtype_q4_k_x8_bl8)));
  CHECK_FALSE(
      detail::supported_mul_mat_dtype(dtype(emel::kernel::detail::dtype_i32)));
}
