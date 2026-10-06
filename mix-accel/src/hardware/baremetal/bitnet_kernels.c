#include <math.h>
#include <stdint.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(BITNET_LINUX)
#include <pthread.h>
#include <unistd.h>
#endif

#include "bitnet_kernels.h"

#define BITNET_LM_HEAD_MAX_WORKERS 8u

static float bf16_to_float(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16;
    return bits.f;
}

static int8_t float_to_i8_scaled(float value, float scale) {
    float scaled_f = value * scale;
    int32_t scaled = (scaled_f >= 0.0f) ?
        (int32_t)(scaled_f + 0.5f) : (int32_t)(scaled_f - 0.5f);

    if (scaled > 127) {
        scaled = 127;
    } else if (scaled < -128) {
        scaled = -128;
    }
    return (int8_t)scaled;
}

float bitnet_rmsnorm_i8_dynamic(const int8_t *src, float src_factor,
                                int8_t *dst, const uint16_t *weight,
                                uint32_t count, float *scratch) {
    double sumsq = 0.0;
    float max_abs = 0.0f;
    float inv_rms;
    float quant_scale;

    if (!scratch) {
        return 1.0f;
    }
    for (uint32_t i = 0u; i < count; i++) {
        float value = (float)src[i] * src_factor;

        scratch[i] = value;
        sumsq += (double)value * (double)value;
    }
    inv_rms = 1.0f / sqrtf((float)(sumsq / (double)count) + 1.0e-5f);
    for (uint32_t i = 0u; i < count; i++) {
        float norm = scratch[i] * inv_rms;
        float value = (weight == 0) ? norm : norm * bf16_to_float(weight[i]);
        float magnitude = (value < 0.0f) ? -value : value;

        scratch[i] = value;
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    if (max_abs <= 1.0e-20f) {
        for (uint32_t i = 0u; i < count; i++) {
            dst[i] = 0;
        }
        return 1.0f;
    }

    quant_scale = 127.0f / max_abs;
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = float_to_i8_scaled(scratch[i], quant_scale);
    }
    return 1.0f / quant_scale;
}

float bitnet_ffn_norm_quantize(const int32_t *gate, float gate_factor,
                               const int32_t *up, float up_factor,
                               int8_t *dst, const uint16_t *weight,
                               uint32_t count, float *scratch) {
    double sumsq = 0.0;
    float max_abs = 0.0f;
    float inv_rms;
    float quant_scale;

    if (!scratch) {
        return 1.0f;
    }
    for (uint32_t i = 0u; i < count; i++) {
        float g = (float)gate[i] * gate_factor;
        float u = (float)up[i] * up_factor;
        float mixed = (g > 0.0f) ? (g * g * u) : 0.0f;

        scratch[i] = mixed;
        sumsq += (double)mixed * (double)mixed;
    }
    inv_rms = 1.0f / sqrtf((float)(sumsq / (double)count) + 1.0e-5f);
    for (uint32_t i = 0u; i < count; i++) {
        float value = scratch[i] * inv_rms;
        float magnitude;

        if (weight != 0) {
            value *= bf16_to_float(weight[i]);
        }
        scratch[i] = value;
        magnitude = (value < 0.0f) ? -value : value;
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    if (max_abs <= 1.0e-20f) {
        for (uint32_t i = 0u; i < count; i++) {
            dst[i] = 0;
        }
        return 1.0f;
    }

    quant_scale = 127.0f / max_abs;
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = float_to_i8_scaled(scratch[i], quant_scale);
    }
    return 1.0f / quant_scale;
}

int32_t bitnet_dot_i8_i8_neon(const int8_t *a, const int8_t *b, uint32_t count) {
#if defined(__aarch64__)
    const int8_t *a_cursor = a;
    const int8_t *b_cursor = b;
    uint32_t blocks = count >> 5;
    uint32_t i = blocks << 5;
    int32_t score = 0;

    if (blocks != 0u) {
        __asm__ volatile(
            "movi v16.4s, #0\n"
            "movi v17.4s, #0\n"
            "movi v18.4s, #0\n"
            "movi v19.4s, #0\n"
            "1:\n"
            "ld1 {v0.16b}, [%[ap]], #16\n"
            "ld1 {v1.16b}, [%[bp]], #16\n"
            "smull v2.8h, v0.8b, v1.8b\n"
            "smull2 v3.8h, v0.16b, v1.16b\n"
            "sadalp v16.4s, v2.8h\n"
            "sadalp v17.4s, v3.8h\n"
            "ld1 {v0.16b}, [%[ap]], #16\n"
            "ld1 {v1.16b}, [%[bp]], #16\n"
            "smull v2.8h, v0.8b, v1.8b\n"
            "smull2 v3.8h, v0.16b, v1.16b\n"
            "sadalp v18.4s, v2.8h\n"
            "sadalp v19.4s, v3.8h\n"
            "subs %w[blocks], %w[blocks], #1\n"
            "b.ne 1b\n"
            "add v16.4s, v16.4s, v17.4s\n"
            "add v18.4s, v18.4s, v19.4s\n"
            "add v16.4s, v16.4s, v18.4s\n"
            "addv s16, v16.4s\n"
            "fmov %w[score], s16\n"
            : [ap] "+r" (a_cursor), [bp] "+r" (b_cursor),
              [blocks] "+r" (blocks), [score] "=r" (score)
            :
            : "cc", "memory", "v0", "v1", "v2", "v3",
              "v16", "v17", "v18", "v19");
    }
    for (; i < count; i++) {
        score += ((int32_t)a[i]) * ((int32_t)b[i]);
    }
    return score;
#else
    int32_t score = 0;

    for (uint32_t i = 0u; i < count; i++) {
        score += ((int32_t)a[i]) * ((int32_t)b[i]);
    }
    return score;
#endif
}

typedef struct {
    const int8_t *activation;
    const int8_t *weights;
    const float *scales;
    uint32_t row_begin;
    uint32_t row_end;
    uint32_t row_size;
    uint32_t top_k;
    uint32_t top_count;
    uint32_t top_rows[BITNET_LM_HEAD_TOPK_MAX];
    float top_scores[BITNET_LM_HEAD_TOPK_MAX];
} LmHeadWorker;

static void lm_head_topk_reset(uint32_t top_k, uint32_t *rows, float *scores) {
    for (uint32_t i = 0u; i < top_k; i++) {
        rows[i] = UINT32_MAX;
        scores[i] = -3.402823466e+38f;
    }
}

static void lm_head_topk_insert(uint32_t row, float score, uint32_t top_k,
                                uint32_t *rows, float *scores) {
    uint32_t insert_at = top_k;

    for (uint32_t i = 0u; i < top_k; i++) {
        if ((score > scores[i]) ||
            ((score == scores[i]) && (row < rows[i]))) {
            insert_at = i;
            break;
        }
    }
    if (insert_at == top_k) {
        return;
    }
    for (uint32_t i = top_k - 1u; i > insert_at; i--) {
        rows[i] = rows[i - 1u];
        scores[i] = scores[i - 1u];
    }
    rows[insert_at] = row;
    scores[insert_at] = score;
}

static int lm_head_candidate_is_worse(uint32_t row_a, float score_a,
                                      uint32_t row_b, float score_b) {
    return (score_a < score_b) ||
           ((score_a == score_b) && (row_a > row_b));
}

/*
 * Keep the least desirable selected candidate at the root.  Each vocabulary
 * row then needs one comparison in the common case instead of scanning and
 * shifting all top_k entries.  Ties retain the lower token id, matching the
 * original increasing-row stable insertion exactly.
 */
static void lm_head_topk_heap_push(uint32_t row, float score, uint32_t top_k,
                                   uint32_t *count,
                                   uint32_t *rows, float *scores) {
    uint32_t index;

    if (*count < top_k) {
        index = (*count)++;
        rows[index] = row;
        scores[index] = score;
        while (index > 0u) {
            uint32_t parent = (index - 1u) >> 1;

            if (!lm_head_candidate_is_worse(rows[index], scores[index],
                                            rows[parent], scores[parent])) {
                break;
            }
            {
                uint32_t swap_row = rows[parent];
                float swap_score = scores[parent];

                rows[parent] = rows[index];
                scores[parent] = scores[index];
                rows[index] = swap_row;
                scores[index] = swap_score;
            }
            index = parent;
        }
        return;
    }

    if (!lm_head_candidate_is_worse(rows[0], scores[0], row, score)) {
        return;
    }

    rows[0] = row;
    scores[0] = score;
    index = 0u;
    for (;;) {
        uint32_t left = (index << 1) + 1u;
        uint32_t right = left + 1u;
        uint32_t worse_child;

        if (left >= top_k) {
            break;
        }
        worse_child = left;
        if ((right < top_k) &&
            lm_head_candidate_is_worse(rows[right], scores[right],
                                       rows[left], scores[left])) {
            worse_child = right;
        }
        if (!lm_head_candidate_is_worse(rows[worse_child], scores[worse_child],
                                        rows[index], scores[index])) {
            break;
        }
        {
            uint32_t swap_row = rows[index];
            float swap_score = scores[index];

            rows[index] = rows[worse_child];
            scores[index] = scores[worse_child];
            rows[worse_child] = swap_row;
            scores[worse_child] = swap_score;
        }
        index = worse_child;
    }
}

static void lm_head_scan(LmHeadWorker *worker) {
    uint32_t row = worker->row_begin;

    lm_head_topk_reset(worker->top_k, worker->top_rows, worker->top_scores);
    worker->top_count = 0u;
    for (; row < worker->row_end; row++) {
        const int8_t *row_weights =
            worker->weights + ((uintptr_t)row * worker->row_size);
        if ((row + 2u) < worker->row_end) {
            __builtin_prefetch(worker->weights +
                               ((uintptr_t)(row + 2u) * worker->row_size),
                               0, 1);
        }
        float score = (float)bitnet_dot_i8_i8_neon(
            worker->activation,
            row_weights,
            worker->row_size) * worker->scales[row];

        lm_head_topk_heap_push(row, score, worker->top_k,
                               &worker->top_count,
                               worker->top_rows, worker->top_scores);
    }
}

/* Decode uses argmax (top_k == 1) by default.  Keep this path branch-light:
 * the generic bounded heap is useful for sampling/top-k, but it needlessly
 * performs heap bookkeeping for every vocabulary row in the hot decode loop. */
static void lm_head_scan_argmax(LmHeadWorker *worker) {
    uint32_t row;
    uint32_t best_row = UINT32_MAX;
    float best_score = -3.402823466e+38f;

    for (row = worker->row_begin; row < worker->row_end; row++) {
        const int8_t *row_weights =
            worker->weights + ((uintptr_t)row * worker->row_size);
        float score;

        if ((row + 2u) < worker->row_end) {
            __builtin_prefetch(worker->weights +
                               ((uintptr_t)(row + 2u) * worker->row_size),
                               0, 1);
        }
        score = (float)bitnet_dot_i8_i8_neon(
            worker->activation, row_weights, worker->row_size) *
            worker->scales[row];
        if ((best_row == UINT32_MAX) || (score > best_score) ||
            ((score == best_score) && (row < best_row))) {
            best_row = row;
            best_score = score;
        }
    }
    worker->top_count = (best_row == UINT32_MAX) ? 0u : 1u;
    worker->top_rows[0] = best_row;
    worker->top_scores[0] = best_score;
}

#if defined(BITNET_LINUX)
static void *lm_head_worker_main(void *argument) {
    LmHeadWorker *worker = (LmHeadWorker *)argument;

    if (worker->top_k == 1u) {
        lm_head_scan_argmax(worker);
    } else {
        lm_head_scan(worker);
    }
    return 0;
}
#endif

int bitnet_lm_head_topk_i8(const int8_t *activation,
                           const int8_t *weights,
                           const float *scales,
                           uint32_t row_count,
                           uint32_t row_size,
                           uint32_t top_k,
                           uint32_t *top_rows,
                           float *top_scores) {
    uint32_t worker_count = 1u;
    LmHeadWorker workers[BITNET_LM_HEAD_MAX_WORKERS];

    if (!activation || !weights || !scales || !row_count || !row_size ||
        !top_rows || !top_scores || !top_k ||
        top_k > BITNET_LM_HEAD_TOPK_MAX || top_k > row_count) {
        return -1;
    }

#if defined(BITNET_LINUX)
    {
        long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        pthread_t threads[BITNET_LM_HEAD_MAX_WORKERS];
        uint32_t started = 0u;

        if (online_cpus > 1) {
            worker_count = (online_cpus > BITNET_LM_HEAD_MAX_WORKERS) ?
                           BITNET_LM_HEAD_MAX_WORKERS : (uint32_t)online_cpus;
        }
        for (uint32_t i = 0u; i < worker_count; i++) {
            workers[i].activation = activation;
            workers[i].weights = weights;
            workers[i].scales = scales;
            workers[i].row_begin = (row_count * i) / worker_count;
            workers[i].row_end = (row_count * (i + 1u)) / worker_count;
            workers[i].row_size = row_size;
            workers[i].top_k = top_k;
            if (i == 0u) {
                continue;
            }
            if (pthread_create(&threads[i], 0, lm_head_worker_main, &workers[i]) == 0) {
                started |= 1u << i;
            } else {
                if (top_k == 1u) {
                    lm_head_scan_argmax(&workers[i]);
                } else {
                    lm_head_scan(&workers[i]);
                }
            }
        }
        if (top_k == 1u) {
            lm_head_scan_argmax(&workers[0]);
        } else {
            lm_head_scan(&workers[0]);
        }
        for (uint32_t i = 1u; i < worker_count; i++) {
            if ((started & (1u << i)) != 0u) {
                (void)pthread_join(threads[i], 0);
            }
        }
    }
#else
    workers[0].activation = activation;
    workers[0].weights = weights;
    workers[0].scales = scales;
    workers[0].row_begin = 0u;
    workers[0].row_end = row_count;
    workers[0].row_size = row_size;
    workers[0].top_k = top_k;
    if (top_k == 1u) {
        lm_head_scan_argmax(&workers[0]);
    } else {
        lm_head_scan(&workers[0]);
    }
#endif

    lm_head_topk_reset(top_k, top_rows, top_scores);
    for (uint32_t worker = 0u; worker < worker_count; worker++) {
        for (uint32_t i = 0u; i < top_k; i++) {
            if (workers[worker].top_rows[i] != UINT32_MAX) {
                lm_head_topk_insert(workers[worker].top_rows[i],
                                    workers[worker].top_scores[i],
                                    top_k, top_rows, top_scores);
            }
        }
    }
    return 0;
}

int bitnet_lm_head_argmax_i8(const int8_t *activation,
                             const int8_t *weights,
                             const float *scales,
                             uint32_t row_count,
                             uint32_t row_size,
                             uint32_t *best_row,
                             float *best_score) {
    return bitnet_lm_head_topk_i8(activation, weights, scales,
                                  row_count, row_size, 1u,
                                  best_row, best_score);
}
