#include "doctest/doctest.h"

#include <array>
#include <span>

#include "emel/error/error.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace {

struct owner_state {
  bool bind_done = false;
  bool bind_error = false;
  bool plan_done = false;
  bool plan_error = false;
  bool apply_done = false;
  bool apply_error = false;
  emel::error::type err = emel::error::cast(emel::model::weight_loader::error::none);
  uint32_t effect_count = 0u;
};

void on_bind_done(void * object, const emel::model::weight_loader::events::bind_done &) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->bind_done = true;
  owner->bind_error = false;
}

void on_bind_error(void * object, const emel::model::weight_loader::events::bind_error & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->bind_done = false;
  owner->bind_error = true;
  owner->err = ev.err;
}

void on_plan_done(void * object, const emel::model::weight_loader::events::plan_done & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->plan_done = true;
  owner->plan_error = false;
  owner->effect_count = ev.effect_count;
}

void on_plan_error(void * object, const emel::model::weight_loader::events::plan_error & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->plan_done = false;
  owner->plan_error = true;
  owner->err = ev.err;
}

void on_apply_done(void * object, const emel::model::weight_loader::events::apply_done &) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->apply_done = true;
  owner->apply_error = false;
}

void on_apply_error(void * object,
                    const emel::model::weight_loader::events::apply_error & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->apply_done = false;
  owner->apply_error = true;
  owner->err = ev.err;
}

}  // namespace

TEST_CASE("weight_loader bind plan apply lifecycle") {
  emel::model::weight_loader::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 2> tensors{};

  emel::model::weight_loader::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_done};
  bind.on_error = {&owner, on_bind_error};
  CHECK(machine.process_event(bind));
  CHECK(owner.bind_done);
  CHECK_FALSE(owner.bind_error);

  std::array<emel::model::weight_loader::effect_request, 2> effects{};
  emel::model::weight_loader::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_done};
  plan.on_error = {&owner, on_plan_error};
  CHECK(machine.process_event(plan));
  CHECK(owner.plan_done);
  CHECK_FALSE(owner.plan_error);
  CHECK(owner.effect_count == 2u);

  std::array<emel::model::weight_loader::effect_result, 2> results{
    emel::model::weight_loader::effect_result{
      .kind = emel::model::weight_loader::effect_kind::k_none,
      .handle = reinterpret_cast<void *>(0x1),
      .err = emel::error::cast(emel::model::weight_loader::error::none),
    },
    emel::model::weight_loader::effect_result{
      .kind = emel::model::weight_loader::effect_kind::k_none,
      .handle = reinterpret_cast<void *>(0x2),
      .err = emel::error::cast(emel::model::weight_loader::error::none),
    },
  };
  emel::model::weight_loader::event::apply_effect_results apply{
    std::span<const emel::model::weight_loader::effect_result>{results}};
  apply.on_done = {&owner, on_apply_done};
  apply.on_error = {&owner, on_apply_error};
  CHECK(machine.process_event(apply));
  CHECK(owner.apply_done);
  CHECK_FALSE(owner.apply_error);
  CHECK(tensors[0].data == reinterpret_cast<const void *>(0x1));
  CHECK(tensors[1].data == reinterpret_cast<const void *>(0x2));
}

TEST_CASE("weight_loader bind rejects invalid inputs") {
  emel::model::weight_loader::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};

  emel::model::weight_loader::event::bind_storage bind{std::span{tensors}.subspan(0u, 0u)};
  bind.on_done = {&owner, on_bind_done};
  bind.on_error = {&owner, on_bind_error};
  CHECK_FALSE(machine.process_event(bind));
  CHECK_FALSE(owner.bind_done);
  CHECK(owner.bind_error);
  CHECK(owner.err == emel::error::cast(emel::model::weight_loader::error::invalid_request));
}

TEST_CASE("weight_loader plan rejects insufficient effect capacity") {
  emel::model::weight_loader::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 2> tensors{};

  emel::model::weight_loader::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_done};
  bind.on_error = {&owner, on_bind_error};
  CHECK(machine.process_event(bind));

  std::array<emel::model::weight_loader::effect_request, 1> effects{};
  emel::model::weight_loader::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_done};
  plan.on_error = {&owner, on_plan_error};
  CHECK_FALSE(machine.process_event(plan));
  CHECK_FALSE(owner.plan_done);
  CHECK(owner.plan_error);
  CHECK(owner.err == emel::error::cast(emel::model::weight_loader::error::capacity));
}

TEST_CASE("weight_loader apply rejects count mismatch") {
  emel::model::weight_loader::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};

  emel::model::weight_loader::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_done};
  bind.on_error = {&owner, on_bind_error};
  CHECK(machine.process_event(bind));

  std::array<emel::model::weight_loader::effect_request, 1> effects{};
  emel::model::weight_loader::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_done};
  plan.on_error = {&owner, on_plan_error};
  CHECK(machine.process_event(plan));

  std::array<emel::model::weight_loader::effect_result, 0> results{};
  emel::model::weight_loader::event::apply_effect_results apply{
    std::span<const emel::model::weight_loader::effect_result>{results}};
  apply.on_done = {&owner, on_apply_done};
  apply.on_error = {&owner, on_apply_error};
  CHECK_FALSE(machine.process_event(apply));
  CHECK_FALSE(owner.apply_done);
  CHECK(owner.apply_error);
  CHECK(owner.err == emel::error::cast(emel::model::weight_loader::error::invalid_request));
}

TEST_CASE("weight_loader apply maps external effect errors to backend_error") {
  emel::model::weight_loader::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};

  emel::model::weight_loader::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_done};
  bind.on_error = {&owner, on_bind_error};
  CHECK(machine.process_event(bind));

  std::array<emel::model::weight_loader::effect_request, 1> effects{};
  emel::model::weight_loader::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_done};
  plan.on_error = {&owner, on_plan_error};
  CHECK(machine.process_event(plan));

  std::array<emel::model::weight_loader::effect_result, 1> results{
    emel::model::weight_loader::effect_result{
      .kind = emel::model::weight_loader::effect_kind::k_none,
      .handle = nullptr,
      .err = emel::error::cast(emel::model::weight_loader::error::out_of_memory),
    },
  };
  emel::model::weight_loader::event::apply_effect_results apply{
    std::span<const emel::model::weight_loader::effect_result>{results}};
  apply.on_done = {&owner, on_apply_done};
  apply.on_error = {&owner, on_apply_error};
  CHECK_FALSE(machine.process_event(apply));
  CHECK_FALSE(owner.apply_done);
  CHECK(owner.apply_error);
  CHECK(owner.err == emel::error::cast(emel::model::weight_loader::error::backend_error));
}
