#pragma once

#include <type_traits>

#include "emel/kernel/events.hpp"
#include "emel/kernel/op_list.hpp"

namespace emel::kernel {

template <class event_type>
struct is_op_event : std::false_type {};

#define EMEL_KERNEL_MARK_OP_EVENT(op_name)                        \
  template <>                                                     \
  struct is_op_event<event::op_name> : std::true_type {};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_MARK_OP_EVENT)
#undef EMEL_KERNEL_MARK_OP_EVENT

template <class event_type>
inline constexpr bool is_op_event_v = is_op_event<event_type>::value;

}  // namespace emel::kernel
