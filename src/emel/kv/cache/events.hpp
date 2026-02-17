#pragma once

#include <cstdint>

namespace emel::kv::cache::event {

struct prepare {
  const int32_t * ubatch_sizes = nullptr;
  int32_t ubatch_count = 0;
  int32_t requested_capacity = 0;

  int32_t * slot_offsets_out = nullptr;
  int32_t slot_offsets_capacity = 0;
  int32_t * ubatch_count_out = nullptr;
  int32_t * error_out = nullptr;

  int32_t n_stream = 1;
  const int32_t * seq_to_stream = nullptr;
  int32_t seq_to_stream_count = 0;
  const int32_t * ubatch_stream_ids = nullptr;
  int32_t ubatch_stream_ids_count = 0;
  const int32_t * ubatch_seq_ids = nullptr;
  int32_t ubatch_seq_ids_count = 0;
};

struct apply_ubatch {
  int32_t ubatch_index = 0;
  int32_t * kv_tokens_out = nullptr;
  int32_t * error_out = nullptr;
  const int32_t * positions = nullptr;
  int32_t positions_count = 0;
};

struct rollback {
  int32_t from_ubatch_index = 0;
  int32_t * error_out = nullptr;
};

struct seq_remove {
  int32_t seq_id = 0;
  int32_t pos_start = -1;
  int32_t pos_end = -1;
  int32_t * error_out = nullptr;
};

struct seq_copy {
  int32_t seq_id_src = 0;
  int32_t seq_id_dst = 0;
  int32_t pos_start = -1;
  int32_t pos_end = -1;
  int32_t * error_out = nullptr;
};

struct seq_keep {
  int32_t seq_id = 0;
  int32_t * error_out = nullptr;
};

struct seq_add {
  int32_t seq_id = 0;
  int32_t pos_start = -1;
  int32_t pos_end = -1;
  int32_t shift = 0;
  int32_t * error_out = nullptr;
};

struct seq_div {
  int32_t seq_id = 0;
  int32_t pos_start = -1;
  int32_t pos_end = -1;
  int32_t divisor = 1;
  int32_t * error_out = nullptr;
};

struct apply_updates {
  using stream_copy_fn =
      bool (*)(int32_t src_stream, int32_t dst_stream, void * user_data, int32_t * error_out);
  using apply_shift_fn =
      bool (*)(int32_t stream_id, const int32_t * shifts, int32_t shift_count, void * user_data,
               int32_t * error_out);

  stream_copy_fn stream_copy = nullptr;
  apply_shift_fn apply_shift = nullptr;
  void * user_data = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_prepare {
  const prepare * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_apply {
  const apply_ubatch * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_rollback {
  const rollback * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_seq_remove {
  const seq_remove * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_seq_copy {
  const seq_copy * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_seq_keep {
  const seq_keep * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_seq_add {
  const seq_add * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_seq_div {
  const seq_div * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_updates {
  const apply_updates * request = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_slots {
  int32_t * error_out = nullptr;
};

struct apply_step {
  const apply_ubatch * request = nullptr;
  int32_t * error_out = nullptr;
};

struct rollback_step {
  const rollback * request = nullptr;
  int32_t * error_out = nullptr;
};

struct seq_remove_step {
  const seq_remove * request = nullptr;
  int32_t * error_out = nullptr;
};

struct seq_copy_step {
  const seq_copy * request = nullptr;
  int32_t * error_out = nullptr;
};

struct seq_keep_step {
  const seq_keep * request = nullptr;
  int32_t * error_out = nullptr;
};

struct seq_add_step {
  const seq_add * request = nullptr;
  int32_t * error_out = nullptr;
};

struct seq_div_step {
  const seq_div * request = nullptr;
  int32_t * error_out = nullptr;
};

struct apply_updates_step {
  const apply_updates * request = nullptr;
  int32_t * error_out = nullptr;
};

struct publish {
  int32_t * error_out = nullptr;
};

}  // namespace emel::kv::cache::event

namespace emel::kv::cache::events {

struct request_ref {
  const event::prepare * prepare = nullptr;
  const event::apply_ubatch * apply = nullptr;
  const event::rollback * rollback = nullptr;
};

struct validate_done {
  request_ref request = {};
};
struct validate_error {
  int32_t err = 0;
  request_ref request = {};
};

struct prepare_slots_done {
  request_ref request = {};
};
struct prepare_slots_error {
  int32_t err = 0;
  request_ref request = {};
};

struct apply_done {
  request_ref request = {};
};
struct apply_error {
  int32_t err = 0;
  request_ref request = {};
};

struct rollback_done {
  request_ref request = {};
};
struct rollback_error {
  int32_t err = 0;
  request_ref request = {};
};

struct publish_done {
  request_ref request = {};
};
struct publish_error {
  int32_t err = 0;
  request_ref request = {};
};

struct seq_remove_done {
  const event::seq_remove * request = nullptr;
};
struct seq_remove_error {
  int32_t err = 0;
  const event::seq_remove * request = nullptr;
};

struct seq_copy_done {
  const event::seq_copy * request = nullptr;
};
struct seq_copy_error {
  int32_t err = 0;
  const event::seq_copy * request = nullptr;
};

struct seq_keep_done {
  const event::seq_keep * request = nullptr;
};
struct seq_keep_error {
  int32_t err = 0;
  const event::seq_keep * request = nullptr;
};

struct seq_add_done {
  const event::seq_add * request = nullptr;
};
struct seq_add_error {
  int32_t err = 0;
  const event::seq_add * request = nullptr;
};

struct seq_div_done {
  const event::seq_div * request = nullptr;
};
struct seq_div_error {
  int32_t err = 0;
  const event::seq_div * request = nullptr;
};

struct updates_done {
  const event::apply_updates * request = nullptr;
};
struct updates_error {
  int32_t err = 0;
  const event::apply_updates * request = nullptr;
};

struct kv_done {};
struct kv_error {
  int32_t err = 0;
};

}  // namespace emel::kv::cache::events
