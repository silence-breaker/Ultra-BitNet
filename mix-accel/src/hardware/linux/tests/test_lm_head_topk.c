#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bitnet_kernels.h"

#define TEST_ROWS 131u
#define TEST_COLS 67u

static void reference_insert(uint32_t row, float score, uint32_t top_k,
                             uint32_t *rows, float *scores) {
    uint32_t insert_at = top_k;

    for (uint32_t i = 0u; i < top_k; i++) {
        if (score > scores[i]) {
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

static void reference_topk(const int8_t *activation, const int8_t *weights,
                           const float *scales, uint32_t row_count,
                           uint32_t row_size, uint32_t top_k,
                           uint32_t *rows, float *scores) {
    for (uint32_t i = 0u; i < top_k; i++) {
        rows[i] = UINT32_MAX;
        scores[i] = -3.402823466e+38f;
    }
    for (uint32_t row = 0u; row < row_count; row++) {
        int32_t dot = 0;

        for (uint32_t col = 0u; col < row_size; col++) {
            dot += (int32_t)activation[col] *
                   (int32_t)weights[((uintptr_t)row * row_size) + col];
        }
        reference_insert(row, (float)dot * scales[row], top_k, rows, scores);
    }
}

static void run_random_case(uint32_t top_k) {
    int8_t activation[TEST_COLS];
    int8_t weights[TEST_ROWS * TEST_COLS];
    float scales[TEST_ROWS];
    uint32_t actual_rows[BITNET_LM_HEAD_TOPK_MAX];
    uint32_t expected_rows[BITNET_LM_HEAD_TOPK_MAX];
    float actual_scores[BITNET_LM_HEAD_TOPK_MAX];
    float expected_scores[BITNET_LM_HEAD_TOPK_MAX];

    for (uint32_t i = 0u; i < TEST_COLS; i++) {
        activation[i] = (int8_t)((int32_t)((i * 37u + 11u) % 255u) - 127);
    }
    for (uint32_t i = 0u; i < TEST_ROWS * TEST_COLS; i++) {
        weights[i] = (int8_t)((int32_t)((i * 73u + 19u) % 255u) - 127);
    }
    for (uint32_t row = 0u; row < TEST_ROWS; row++) {
        scales[row] = 0.0005f * (float)((row % 17u) + 1u);
    }

    reference_topk(activation, weights, scales, TEST_ROWS, TEST_COLS,
                   top_k, expected_rows, expected_scores);
    assert(bitnet_lm_head_topk_i8(
        activation, weights, scales, TEST_ROWS, TEST_COLS, top_k,
        actual_rows, actual_scores) == 0);
    for (uint32_t i = 0u; i < top_k; i++) {
        assert(actual_rows[i] == expected_rows[i]);
        assert(actual_scores[i] == expected_scores[i]);
    }
}

static void run_tie_case(void) {
    int8_t activation[TEST_COLS] = {0};
    int8_t weights[TEST_ROWS * TEST_COLS] = {0};
    float scales[TEST_ROWS];
    uint32_t rows[BITNET_LM_HEAD_TOPK_MAX];
    float scores[BITNET_LM_HEAD_TOPK_MAX];

    for (uint32_t row = 0u; row < TEST_ROWS; row++) {
        scales[row] = 1.0f;
    }
    assert(bitnet_lm_head_topk_i8(
        activation, weights, scales, TEST_ROWS, TEST_COLS, 40u,
        rows, scores) == 0);
    for (uint32_t i = 0u; i < 40u; i++) {
        assert(rows[i] == i);
        assert(scores[i] == 0.0f);
    }
}

int main(void) {
    int8_t activation[1] = {1};
    int8_t weights[1] = {1};
    float scales[1] = {1.0f};
    uint32_t row;
    float score;

    run_random_case(1u);
    run_random_case(7u);
    run_random_case(40u);
    run_random_case(64u);
    run_tie_case();

    assert(bitnet_lm_head_topk_i8(
        activation, weights, scales, 1u, 1u, 0u, &row, &score) == -1);
    assert(bitnet_lm_head_topk_i8(
        activation, weights, scales, 1u, 1u, 2u, &row, &score) == -1);
    puts("LM_HEAD_TOPK_TEST_PASS");
    return 0;
}
