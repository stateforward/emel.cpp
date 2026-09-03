#pragma once

#include <cmath>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace emel::kernel::swa::detail {

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_SWA_DETAIL_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define EMEL_KERNEL_SWA_DETAIL_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_SWA_DETAIL_AVX2_TARGET
#endif

#if defined(__x86_64__) || defined(_M_X64)
// Source-owned AVX2 port of the ARM optimized-routines expf polynomial used by
// ggml_v_expf. This is an approximation, not a vector libm call. Stable
// softmax only supplies finite max-shifted inputs <= 0; callers retain scalar
// std::exp for tails. The bit construction is intrinsic-only and does not
// alias float and integer objects.
EMEL_KERNEL_SWA_DETAIL_AVX2_TARGET inline __m256
expf8_approx_avx2(const __m256 x) noexcept {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b =
      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                       _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
                    _mm256_set1_ps(126.0f), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(
      _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                      _mm256_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                      _mm256_set1_ps(0x1.fffdb6p-2f))),
      u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (_mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0)
    return _mm256_fmadd_ps(j, k, k);

  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(static_cast<int32_t>(0x82000000u)));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
                    _mm256_set1_ps(192.0f), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c),
                               _mm256_fmadd_ps(k, j, k)))));
}
#endif

// Replaces each finite score with exp(score - max_score) and returns a float
// sum. Eight weights are computed together, then accumulated in logical lane
// order (0..7) so the reduction order remains the scalar row order. The tail
// uses std::exp. `scores` and `count` must name a valid writable row.
EMEL_KERNEL_SWA_DETAIL_AVX2_TARGET inline float
exp_sum_avx2(float *scores, const uint32_t count,
             const float max_score) noexcept {
  float sum = 0.0f;
  size_t offset = 0u;
#if defined(__x86_64__) || defined(_M_X64)
  const __m256 max_v = _mm256_set1_ps(max_score);
  const size_t vector_end = static_cast<size_t>(count) & ~size_t{7u};
  for (; offset < vector_end; offset += 8u) {
    const __m256 weights = expf8_approx_avx2(
        _mm256_sub_ps(_mm256_loadu_ps(scores + offset), max_v));
    _mm256_storeu_ps(scores + offset, weights);
    sum += scores[offset + 0u];
    sum += scores[offset + 1u];
    sum += scores[offset + 2u];
    sum += scores[offset + 3u];
    sum += scores[offset + 4u];
    sum += scores[offset + 5u];
    sum += scores[offset + 6u];
    sum += scores[offset + 7u];
  }
#endif
  for (; offset < static_cast<size_t>(count); ++offset) {
    const float weight = std::exp(scores[offset] - max_score);
    scores[offset] = weight;
    sum += weight;
  }
  return sum;
}

} // namespace emel::kernel::swa::detail
