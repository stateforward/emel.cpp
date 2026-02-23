#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/hybrid/sm.hpp"
#include "emel/memory/coordinator/kv/sm.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"
#include "emel/sm.hpp"

namespace emel::memory::coordinator {

enum class coordinator_kind : uint8_t {
  recurrent = 0,
  kv = 1,
  hybrid = 2,
};

class any {
 public:
  any() = default;
  explicit any(const coordinator_kind kind) : core_(kind), selected_kind_(kind) {}

  any(const any &) = delete;
  any & operator=(const any &) = delete;
  any(any &&) = delete;
  any & operator=(any &&) = delete;

  ~any() = default;

  void set_kind(const coordinator_kind kind) {
    core_.set_kind(kind);
    selected_kind_ = kind;
  }

  coordinator_kind kind() const noexcept { return selected_kind_; }

  bool process_event(const event::prepare_update & ev) {
    last_error_from_lifecycle_ = false;
    return core_.process_event(ev);
  }

  bool process_event(const event::prepare_batch & ev) {
    last_error_from_lifecycle_ = false;
    return core_.process_event(ev);
  }

  bool process_event(const event::prepare_full & ev) {
    last_error_from_lifecycle_ = false;
    return core_.process_event(ev);
  }

  bool process_event(const event::reserve & ev) {
    last_error_from_lifecycle_ = true;
    switch (selected_kind_) {
      case coordinator_kind::recurrent: {
        const bool accepted = recurrent_memory_.process_event(
            emel::memory::recurrent::event::reserve{
                .slot_capacity = ev.recurrent_slot_capacity,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = recurrent_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::kv: {
        const bool accepted = kv_memory_.process_event(
            emel::memory::kv::event::reserve{
                .kv_size = ev.kv_size,
                .n_stream = ev.n_stream,
                .n_pad = ev.n_pad,
                .n_swa = ev.n_swa,
                .swa_type = ev.swa_type,
                .seq_to_stream = ev.seq_to_stream,
                .seq_to_stream_count = ev.seq_to_stream_count,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = kv_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::hybrid: {
        const bool accepted = hybrid_memory_.process_event(
            emel::memory::hybrid::event::reserve{
                .kv_size = ev.kv_size,
                .recurrent_slot_capacity = ev.recurrent_slot_capacity,
                .n_stream = ev.n_stream,
                .n_pad = ev.n_pad,
                .n_swa = ev.n_swa,
                .swa_type = ev.swa_type,
                .seq_to_stream = ev.seq_to_stream,
                .seq_to_stream_count = ev.seq_to_stream_count,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = hybrid_memory_.last_error();
        return accepted;
      }
    }
    lifecycle_last_error_ = EMEL_ERR_BACKEND;
    if (ev.error_out != nullptr) {
      *ev.error_out = lifecycle_last_error_;
    }
    return false;
  }

  bool process_event(const event::allocate_sequence & ev) {
    last_error_from_lifecycle_ = true;
    switch (selected_kind_) {
      case coordinator_kind::recurrent: {
        const bool accepted = recurrent_memory_.process_event(
            emel::memory::recurrent::event::allocate_sequence{
                .seq_id = ev.seq_id,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = recurrent_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::kv: {
        const bool accepted = kv_memory_.process_event(
            emel::memory::kv::event::allocate_sequence{
                .seq_id = ev.seq_id,
                .slot_count = ev.slot_count,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = kv_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::hybrid: {
        const bool accepted = hybrid_memory_.process_event(
            emel::memory::hybrid::event::allocate_sequence{
                .seq_id = ev.seq_id,
                .slot_count = ev.slot_count,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = hybrid_memory_.last_error();
        return accepted;
      }
    }
    lifecycle_last_error_ = EMEL_ERR_BACKEND;
    if (ev.error_out != nullptr) {
      *ev.error_out = lifecycle_last_error_;
    }
    return false;
  }

  bool process_event(const event::branch_sequence & ev) {
    last_error_from_lifecycle_ = true;
    switch (selected_kind_) {
      case coordinator_kind::recurrent: {
        const bool accepted = recurrent_memory_.process_event(
            emel::memory::recurrent::event::branch_sequence{
                .seq_id_src = ev.seq_id_src,
                .seq_id_dst = ev.seq_id_dst,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = recurrent_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::kv: {
        const bool accepted = kv_memory_.process_event(
            emel::memory::kv::event::branch_sequence{
                .seq_id_src = ev.seq_id_src,
                .seq_id_dst = ev.seq_id_dst,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = kv_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::hybrid: {
        const bool accepted = hybrid_memory_.process_event(
            emel::memory::hybrid::event::branch_sequence{
                .seq_id_src = ev.seq_id_src,
                .seq_id_dst = ev.seq_id_dst,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = hybrid_memory_.last_error();
        return accepted;
      }
    }
    lifecycle_last_error_ = EMEL_ERR_BACKEND;
    if (ev.error_out != nullptr) {
      *ev.error_out = lifecycle_last_error_;
    }
    return false;
  }

  bool process_event(const event::free_sequence & ev) {
    last_error_from_lifecycle_ = true;
    switch (selected_kind_) {
      case coordinator_kind::recurrent: {
        const bool accepted = recurrent_memory_.process_event(
            emel::memory::recurrent::event::free_sequence{
                .seq_id = ev.seq_id,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = recurrent_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::kv: {
        const bool accepted = kv_memory_.process_event(
            emel::memory::kv::event::free_sequence{
                .seq_id = ev.seq_id,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = kv_memory_.last_error();
        return accepted;
      }
      case coordinator_kind::hybrid: {
        const bool accepted = hybrid_memory_.process_event(
            emel::memory::hybrid::event::free_sequence{
                .seq_id = ev.seq_id,
                .error_out = ev.error_out,
            });
        lifecycle_last_error_ = hybrid_memory_.last_error();
        return accepted;
      }
    }
    lifecycle_last_error_ = EMEL_ERR_BACKEND;
    if (ev.error_out != nullptr) {
      *ev.error_out = lifecycle_last_error_;
    }
    return false;
  }

  int32_t last_error() const noexcept {
    if (last_error_from_lifecycle_) {
      return lifecycle_last_error_;
    }
    int32_t err = EMEL_ERR_BACKEND;
    core_.visit([&](const auto & sm) { err = sm.last_error(); });
    return err;
  }

  event::memory_status last_status() const noexcept {
    event::memory_status status = {};
    core_.visit([&](const auto & sm) { status = sm.last_status(); });
    return status;
  }

  bool has_sequence(int32_t seq_id) const noexcept {
    switch (selected_kind_) {
      case coordinator_kind::recurrent:
        return recurrent_memory_.has_sequence(seq_id);
      case coordinator_kind::kv:
        return kv_memory_.has_sequence(seq_id);
      case coordinator_kind::hybrid:
        return hybrid_memory_.has_sequence(seq_id);
    }
    return false;
  }

 private:
  using sm_list = boost::sml::aux::type_list<recurrent::sm, kv::sm, hybrid::sm>;
  using event_list = boost::sml::aux::type_list<event::prepare_update,
                                                event::prepare_batch,
                                                event::prepare_full>;

  emel::sm_any<coordinator_kind, sm_list, event_list> core_{};
  coordinator_kind selected_kind_ = coordinator_kind::recurrent;
  emel::memory::kv::sm kv_memory_{};
  emel::memory::recurrent::sm recurrent_memory_{};
  emel::memory::hybrid::sm hybrid_memory_{};
  int32_t lifecycle_last_error_ = EMEL_OK;
  bool last_error_from_lifecycle_ = false;
};

}  // namespace emel::memory::coordinator
