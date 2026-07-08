#pragma once

#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"

namespace emel::memory::hybrid {

using kv_reserve_dispatch_fn = bool(void *, const emel::memory::event::reserve &);
using kv_allocate_sequence_dispatch_fn =
    bool(void *, const emel::memory::event::allocate_sequence &);
using kv_allocate_slots_dispatch_fn = bool(void *, const emel::memory::event::allocate_slots &);
using kv_branch_sequence_dispatch_fn =
    bool(void *, const emel::memory::event::branch_sequence &);
using kv_free_sequence_dispatch_fn = bool(void *, const emel::memory::event::free_sequence &);
using kv_rollback_slots_dispatch_fn = bool(void *, const emel::memory::event::rollback_slots &);
using kv_capture_view_dispatch_fn = bool(void *, const emel::memory::event::capture_view &);

struct kv_binding {
  void * actor = nullptr;
  kv_reserve_dispatch_fn * dispatch_reserve = nullptr;
  kv_allocate_sequence_dispatch_fn * dispatch_allocate_sequence = nullptr;
  kv_allocate_slots_dispatch_fn * dispatch_allocate_slots = nullptr;
  kv_branch_sequence_dispatch_fn * dispatch_branch_sequence = nullptr;
  kv_free_sequence_dispatch_fn * dispatch_free_sequence = nullptr;
  kv_rollback_slots_dispatch_fn * dispatch_rollback_slots = nullptr;
  kv_capture_view_dispatch_fn * dispatch_capture_view = nullptr;
};

template <class actor_type>
bool dispatch_kv_reserve(void * actor, const emel::memory::event::reserve & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
bool dispatch_kv_allocate_sequence(void * actor,
                                   const emel::memory::event::allocate_sequence & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
bool dispatch_kv_allocate_slots(void * actor,
                                const emel::memory::event::allocate_slots & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
bool dispatch_kv_branch_sequence(void * actor,
                                 const emel::memory::event::branch_sequence & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
bool dispatch_kv_free_sequence(void * actor,
                               const emel::memory::event::free_sequence & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
bool dispatch_kv_rollback_slots(void * actor,
                                const emel::memory::event::rollback_slots & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
bool dispatch_kv_capture_view(void * actor, const emel::memory::event::capture_view & ev) {
  return static_cast<actor_type *>(actor)->process_event(ev);
}

template <class actor_type>
kv_binding bind_kv_actor(actor_type & actor) noexcept {
  return kv_binding{
      .actor = &actor,
      .dispatch_reserve = &dispatch_kv_reserve<actor_type>,
      .dispatch_allocate_sequence = &dispatch_kv_allocate_sequence<actor_type>,
      .dispatch_allocate_slots = &dispatch_kv_allocate_slots<actor_type>,
      .dispatch_branch_sequence = &dispatch_kv_branch_sequence<actor_type>,
      .dispatch_free_sequence = &dispatch_kv_free_sequence<actor_type>,
      .dispatch_rollback_slots = &dispatch_kv_rollback_slots<actor_type>,
      .dispatch_capture_view = &dispatch_kv_capture_view<actor_type>,
  };
}

inline bool kv_binding_complete(const kv_binding & binding) noexcept {
  return binding.actor != nullptr && binding.dispatch_reserve != nullptr &&
         binding.dispatch_allocate_sequence != nullptr &&
         binding.dispatch_allocate_slots != nullptr &&
         binding.dispatch_branch_sequence != nullptr &&
         binding.dispatch_free_sequence != nullptr &&
         binding.dispatch_rollback_slots != nullptr &&
         binding.dispatch_capture_view != nullptr;
}

inline kv_binding bind_or_default_kv_actor(const kv_binding & binding,
                                           emel::memory::kv::sm & fallback) noexcept {
  return kv_binding_complete(binding) ? binding : bind_kv_actor(fallback);
}

}  // namespace emel::memory::hybrid

namespace emel::memory::hybrid::action {

struct context {
  context() : kv(), recurrent(), kv_actor(emel::memory::hybrid::bind_kv_actor(kv)) {}

  emel::memory::kv::sm kv = {};
  emel::memory::recurrent::sm recurrent = {};
  emel::memory::hybrid::kv_binding kv_actor = {};
};

}  // namespace emel::memory::hybrid::action
