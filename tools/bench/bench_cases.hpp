#pragma once

#include "bench_common.hpp"

#include <vector>

namespace emel::bench {

void append_emel_buffer_allocator_cases(std::vector<result> & results, const config & cfg);
void append_reference_buffer_allocator_cases(std::vector<result> & results, const config & cfg);
void append_emel_batch_splitter_cases(std::vector<result> & results, const config & cfg);
void append_reference_batch_splitter_cases(std::vector<result> & results, const config & cfg);
void append_emel_batch_sanitizer_cases(std::vector<result> & results, const config & cfg);
void append_reference_batch_sanitizer_cases(std::vector<result> & results, const config & cfg);

}  // namespace emel::bench
