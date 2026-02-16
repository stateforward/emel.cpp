#ifndef EMEL_EMEL_H
#define EMEL_EMEL_H

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

typedef struct emel_error_detail {
  int32_t status;
  uint32_t domain;
  uint32_t phase;
  uint32_t reason;
  int32_t index;
  int32_t aux;
} emel_error_detail;

#ifdef __cplusplus
}
#endif

#endif  // EMEL_EMEL_H
