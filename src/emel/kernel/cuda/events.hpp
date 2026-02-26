#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/detail.hpp"

namespace emel::kernel::cuda::events {

enum class phase_outcome : uint8_t {
  unknown = 0,
  done = 1,
  failed = 2,
};

struct dispatch_done {};

struct dispatch_error {
  int32_t err = static_cast<int32_t>(emel::error::cast(error::internal_error));
};

}  // namespace emel::kernel::cuda::events

namespace emel::kernel::cuda::event {

// Internal context object carried per dispatch call.
struct dispatch_ctx {
  events::phase_outcome outcome = events::phase_outcome::unknown;
  int32_t err = static_cast<int32_t>(emel::error::cast(error::none));
};

struct dispatch_scaffold {
  const ::emel::kernel::event::scaffold & request;
  dispatch_ctx & ctx;
};

#define EMEL_KERNEL_DECLARE_DISPATCH_EVENT(op_name) \
  struct dispatch_##op_name {                       \
    const ::emel::kernel::event::op_name & request; \
    dispatch_ctx & ctx;                              \
  };
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_DISPATCH_EVENT)
#undef EMEL_KERNEL_DECLARE_DISPATCH_EVENT

template <class event_type>
struct dispatch_event_for;

#define EMEL_KERNEL_DECLARE_DISPATCH_EVENT_TRAIT(op_name)          \
  template <>                                                       \
  struct dispatch_event_for<::emel::kernel::event::op_name> {      \
    using type = dispatch_##op_name;                                \
  };
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_DISPATCH_EVENT_TRAIT)
#undef EMEL_KERNEL_DECLARE_DISPATCH_EVENT_TRAIT

template <class event_type>
using dispatch_event_for_t = typename dispatch_event_for<event_type>::type;

}  // namespace emel::kernel::cuda::event
