#ifndef BITNET_KERNELS_H
#define BITNET_KERNELS_H

#include <stdint.h>

#define BITNET_LM_HEAD_TOPK_MAX 64u

float bitnet_rmsnorm_i8_dynamic(const int8_t *src, float src_factor,
                                int8_t *dst, const uint16_t *weight,
                                uint32_t count, float *scratch);
float bitnet_ffn_norm_quantize(const int32_t *gate, float gate_factor,
                               const int32_t *up, float up_factor,
                               int8_t *dst, const uint16_t *weight,
                               uint32_t count, float *scratch);
int32_t bitnet_dot_i8_i8_neon(const int8_t *a, const int8_t *b, uint32_t count);
int bitnet_lm_head_argmax_i8(const int8_t *activation,
                             const int8_t *weights,
                             const float *scales,
                             uint32_t row_count,
                             uint32_t row_size,
                             uint32_t *best_row,
                             float *best_score);
int bitnet_lm_head_topk_i8(const int8_t *activation,
                           const int8_t *weights,
                           const float *scales,
                           uint32_t row_count,
                           uint32_t row_size,
                           uint32_t top_k,
                           uint32_t *top_rows,
                           float *top_scores);

#endif
