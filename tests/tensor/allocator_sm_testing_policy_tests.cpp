#include <array>

#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/sm.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

}  // namespace

TEST_CASE("tensor_allocator_sm_reports_idle_after_success") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .error_out = &err,
    .detail_out = &detail,
  }));
  CHECK(err == EMEL_OK);

  int32_t state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("tensor_allocator_sm_testing_policy_handles_release_from_mid_state") {
  namespace sml = boost::sml;
  emel::tensor::allocator::action::context ctx{};
  sml::sm<emel::tensor::allocator::model, sml::testing> machine{ctx};
  machine.set_current_states(sml::state<emel::tensor::allocator::Validating>);

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  CHECK(machine.process_event(emel::tensor::allocator::event::release{
    .error_out = &err,
    .detail_out = &detail,
  }));
  CHECK(err == EMEL_OK);
  CHECK(detail.status == EMEL_OK);
}
