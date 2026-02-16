#ifndef EMEL_EMEL_H
#define EMEL_EMEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum emel_status {
  EMEL_OK = 0,
  EMEL_ERR_INVALID_ARGUMENT = 1,
  EMEL_ERR_FORMAT_UNSUPPORTED = 2,
  EMEL_ERR_PARSE_FAILED = 3,
  EMEL_ERR_IO = 4,
  EMEL_ERR_MODEL_INVALID = 5,
  EMEL_ERR_BACKEND = 6
} emel_status;

// Error detail domains for machine-level diagnostics.
#define EMEL_ERROR_DOMAIN_NONE 0u
#define EMEL_ERROR_DOMAIN_TENSOR_ALLOCATOR 1u
#define EMEL_BUFFER_MAX_SOURCES 4u

typedef struct emel_error_detail {
  int32_t status;
  uint32_t domain;
  uint32_t phase;
  uint32_t reason;
  int32_t index;
  int32_t aux;
} emel_error_detail;

typedef struct emel_buffer_tensor_desc {
  int32_t tensor_id;
  int32_t alloc_size;
  int32_t src_ids[EMEL_BUFFER_MAX_SOURCES];
  uint8_t is_view;
  uint8_t _pad0[3];
  int32_t view_src_id;
  uint8_t is_input;
  uint8_t is_output;
  uint8_t can_inplace;
  uint8_t has_external_data;
} emel_buffer_tensor_desc;

typedef struct emel_buffer_graph_view {
  const emel_buffer_tensor_desc * nodes;
  int32_t n_nodes;
  const emel_buffer_tensor_desc * leafs;
  int32_t n_leafs;
} emel_buffer_graph_view;

typedef struct emel_buffer_allocator emel_buffer_allocator;

size_t emel_buffer_allocator_storage_size(void);
size_t emel_buffer_allocator_storage_alignment(void);
emel_buffer_allocator * emel_buffer_allocator_init(void * storage, size_t storage_size);
void emel_buffer_allocator_destroy(emel_buffer_allocator * allocator);

emel_status emel_buffer_allocator_initialize(
  emel_buffer_allocator * allocator,
  int32_t buffer_count,
  const int32_t * buffer_alignments,
  const int32_t * buffer_max_sizes);

emel_status emel_buffer_allocator_reserve_n_size(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph,
  const int32_t * node_buffer_ids,
  const int32_t * leaf_buffer_ids,
  int32_t * sizes_out,
  int32_t sizes_out_count);

emel_status emel_buffer_allocator_reserve_n(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph,
  const int32_t * node_buffer_ids,
  const int32_t * leaf_buffer_ids);

emel_status emel_buffer_allocator_reserve(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph);

emel_status emel_buffer_allocator_alloc_graph(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph);

emel_status emel_buffer_allocator_release(emel_buffer_allocator * allocator);

emel_status emel_buffer_allocator_alloc_ctx_tensors_from_buft_size(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph,
  const int32_t * node_buffer_ids,
  const int32_t * leaf_buffer_ids,
  int32_t * sizes_out,
  int32_t sizes_out_count);

emel_status emel_buffer_allocator_alloc_ctx_tensors_from_buft(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph,
  const int32_t * node_buffer_ids,
  const int32_t * leaf_buffer_ids);

emel_status emel_buffer_allocator_alloc_ctx_tensors(
  emel_buffer_allocator * allocator,
  const emel_buffer_graph_view * graph);

int32_t emel_buffer_allocator_buffer_size(const emel_buffer_allocator * allocator, int32_t buffer_id);
int32_t emel_buffer_allocator_buffer_chunk_id(
  const emel_buffer_allocator * allocator, int32_t buffer_id);
uint64_t emel_buffer_allocator_buffer_chunk_offset(
  const emel_buffer_allocator * allocator, int32_t buffer_id);
uint64_t emel_buffer_allocator_buffer_alloc_size(
  const emel_buffer_allocator * allocator, int32_t buffer_id);

#ifdef __cplusplus
}
#endif

#endif  // EMEL_EMEL_H
