#include "bitnet_accel.h"
#include "xil_cache.h"
#include "xaxidma_hw.h"
#include "xil_types.h"

#ifdef BITNET_LINUX
#include <stdio.h>
#define BITNET_ACCEL_DMA_TIMEOUT 50000000u
#define BITNET_DMA_DIAG(...) fprintf(stderr, __VA_ARGS__)
#else
#include "xil_printf.h"
#define BITNET_ACCEL_DMA_TIMEOUT 1000000000u
#define BITNET_DMA_DIAG(...) xil_printf(__VA_ARGS__)
#endif

static int wait_dma_idle_checked(
    XAxiDma *dma,
    int direction,
    const char *stage
) {
    uintptr_t channel = dma->RegBase +
        ((direction == XAXIDMA_DMA_TO_DEVICE) ?
             XAXIDMA_TX_OFFSET : XAXIDMA_RX_OFFSET);
    uint32_t timeout = BITNET_ACCEL_DMA_TIMEOUT;

    for (;;) {
        uint32_t status = XAxiDma_ReadReg(channel, XAXIDMA_SR_OFFSET);

        if ((status & XAXIDMA_ERR_ALL_MASK) != 0u) {
            BITNET_DMA_DIAG(
                "DMA_FAILURE stage=%s base=0x%lx direction=%s status=0x%08lx reason=error\n",
                stage, (unsigned long)dma->RegBase,
                (direction == XAXIDMA_DMA_TO_DEVICE) ? "MM2S" : "S2MM",
                (unsigned long)status);
            return XST_FAILURE;
        }
        if ((status & XAXIDMA_IDLE_MASK) != 0u) {
            /* Preserve the Linux compatibility layer's pending S2MM bounce
             * copy, which is completed by XAxiDma_Busy() on the first idle
             * observation. */
            (void)XAxiDma_Busy(dma, direction);
            return XST_SUCCESS;
        }
        if (timeout-- == 0u) {
            BITNET_DMA_DIAG(
                "DMA_FAILURE stage=%s base=0x%lx direction=%s status=0x%08lx reason=timeout\n",
                stage, (unsigned long)dma->RegBase,
                (direction == XAXIDMA_DMA_TO_DEVICE) ? "MM2S" : "S2MM",
                (unsigned long)status);
            return XST_FAILURE;
        }
    }
}

static uint32_t align_input_beat(uint32_t bytes) {
    return (bytes + BITNET_ACCEL_INPUT_BYTES_PER_BEAT - 1u) &
           ~(BITNET_ACCEL_INPUT_BYTES_PER_BEAT - 1u);
}

static uint8_t encode_i8(int8_t v) {
    return (uint8_t)v;
}

static int32_t decode_contrib(int8_t act, uint8_t code, uint32_t flags) {
    if ((flags & 0x3u) == BITNET_ACCEL_MAP_HF_PACKED) {
        if ((code & 0x3u) == 0x0u) {
            return -(int32_t)act;
        }
        if ((code & 0x3u) == 0x2u) {
            return (int32_t)act;
        }
        return 0;
    }
    switch (code & 0x3u) {
    case 0x1u:
        return -(int32_t)act;
    case 0x3u:
        return (int32_t)act;
    default:
        return 0;
    }
}

uint32_t bitnet_accel_stream_words(uint32_t out_features, uint32_t in_features) {
    uint32_t act_bytes = align_input_beat(in_features);
    uint32_t act_words = act_bytes / sizeof(uint32_t);
    uint32_t weight_bytes = (out_features / 4u) * align_input_beat(in_features);
    uint32_t weight_words = weight_bytes / sizeof(uint32_t);
    return BITNET_ACCEL_HEADER_WORDS + act_words + weight_words;
}

static uint32_t normalize_batch_rows(uint32_t batch_rows) {
    if ((batch_rows == 0u) || (batch_rows > BITNET_ACCEL_GEMM_MAX_BATCH_ROWS)) {
        return 0u;
    }
    return batch_rows;
}

static uint32_t encode_gemm_flags(uint32_t flags, uint32_t batch_rows) {
    uint32_t clean_flags = flags &
        ~(BITNET_ACCEL_GEMM_BATCH_MASK << BITNET_ACCEL_GEMM_BATCH_SHIFT);

    return clean_flags |
           ((batch_rows & BITNET_ACCEL_GEMM_BATCH_MASK) << BITNET_ACCEL_GEMM_BATCH_SHIFT);
}

uint32_t bitnet_accel_gemm_stream_words(
    uint32_t out_features,
    uint32_t in_features,
    uint32_t batch_rows
) {
    uint32_t batch = normalize_batch_rows(batch_rows);
    uint32_t act_bytes = align_input_beat(in_features);
    uint32_t act_words = (act_bytes / sizeof(uint32_t)) * batch;
    uint32_t weight_bytes = (out_features / 4u) * align_input_beat(in_features);
    uint32_t weight_words = weight_bytes / sizeof(uint32_t);

    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (batch == 0u)) {
        return 0u;
    }

    return BITNET_ACCEL_HEADER_WORDS + act_words + weight_words;
}

static int build_weight_stream_words(
    uint32_t *stream_words,
    uint32_t stream_word_capacity,
    const uint8_t *packed_weights,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t *weight_word_count
) {
    uint32_t padded_group_bytes = align_input_beat(in_features);
    uint32_t groups = out_features / 4u;
    uint32_t required = (groups * padded_group_bytes) / sizeof(uint32_t);
    uint32_t word_index = 0u;

    if ((stream_words == 0) || (packed_weights == 0) || (weight_word_count == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u)) {
        return XST_INVALID_PARAM;
    }
    if (stream_word_capacity < required) {
        return XST_BUFFER_TOO_SMALL;
    }

    for (uint32_t group = 0u; group < groups; group++) {
        uint32_t group_base = group * in_features;

        for (uint32_t byte = 0u; byte < padded_group_bytes; byte += 4u) {
            uint32_t word = 0u;
            for (uint32_t b = 0u; b < 4u; b++) {
                uint32_t k = byte + b;
                if (k < in_features) {
                    word |= ((uint32_t)packed_weights[group_base + k]) << (8u * b);
                }
            }
            stream_words[word_index++] = word;
        }
    }

    *weight_word_count = word_index;
    return XST_SUCCESS;
}

static uint32_t weights_are_input_beat_aligned(uint32_t in_features) {
    return align_input_beat(in_features) == in_features;
}

static uint32_t normalize_chunk_out(uint32_t out_features, uint32_t chunk_out_features) {
    uint32_t chunk = chunk_out_features;

    if ((chunk == 0u) || (chunk > out_features)) {
        chunk = out_features;
    }
    chunk &= ~0x3u;
    return chunk;
}

uint32_t bitnet_accel_chunk_count(uint32_t out_features, uint32_t chunk_out_features) {
    uint32_t chunk = normalize_chunk_out(out_features, chunk_out_features);

    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) || (chunk == 0u)) {
        return 0u;
    }

    return (out_features + chunk - 1u) / chunk;
}

uint32_t bitnet_accel_chunked_stream_words(
    uint32_t out_features,
    uint32_t in_features,
    uint32_t chunk_out_features
) {
    uint32_t chunk = normalize_chunk_out(out_features, chunk_out_features);
    uint32_t total = 0u;

    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (chunk == 0u)) {
        return 0u;
    }

    for (uint32_t out_base = 0u; out_base < out_features; out_base += chunk) {
        uint32_t remaining = out_features - out_base;
        uint32_t this_out = (remaining > chunk) ? chunk : remaining;
        total += bitnet_accel_stream_words(this_out, in_features);
    }

    return total;
}

int bitnet_accel_build_stream(
    uint32_t *stream_words,
    uint32_t stream_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags
) {
    uint32_t required = bitnet_accel_stream_words(out_features, in_features);
    uint32_t word_index = 0u;
    uint32_t weight_word_count = 0u;
    int status;

    if ((stream_words == 0) || (activations == 0) || (packed_weights == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) || (in_features == 0u)) {
        return XST_INVALID_PARAM;
    }
    if (stream_word_capacity < required) {
        return XST_BUFFER_TOO_SMALL;
    }

    stream_words[word_index++] = BITNET_ACCEL_MAGIC;
    stream_words[word_index++] = out_features;
    stream_words[word_index++] = in_features;
    stream_words[word_index++] = flags;
    while (word_index < BITNET_ACCEL_HEADER_WORDS) {
        stream_words[word_index++] = 0u;
    }

    for (uint32_t k = 0u; k < in_features; k += 4u) {
        uint32_t word = 0u;
        for (uint32_t b = 0u; b < 4u; b++) {
            uint32_t idx = k + b;
            if (idx < in_features) {
                word |= ((uint32_t)encode_i8(activations[idx])) << (8u * b);
            }
        }
        stream_words[word_index++] = word;
    }
    while (word_index < (BITNET_ACCEL_HEADER_WORDS +
                         (align_input_beat(in_features) / sizeof(uint32_t)))) {
        stream_words[word_index++] = 0u;
    }

    status = build_weight_stream_words(stream_words + word_index,
                                       stream_word_capacity - word_index,
                                       packed_weights, out_features, in_features,
                                       &weight_word_count);
    if (status != XST_SUCCESS) {
        return status;
    }
    word_index += weight_word_count;

    return XST_SUCCESS;
}

int bitnet_accel_build_gemm_stream(
    uint32_t *stream_words,
    uint32_t stream_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t batch_rows,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags
) {
    uint32_t batch = normalize_batch_rows(batch_rows);
    uint32_t required = bitnet_accel_gemm_stream_words(out_features, in_features, batch);
    uint32_t word_index = 0u;
    uint32_t weight_word_count = 0u;
    int status;

    if ((stream_words == 0) || (activations == 0) || (packed_weights == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (batch == 0u) || (required == 0u)) {
        return XST_INVALID_PARAM;
    }
    if (stream_word_capacity < required) {
        return XST_BUFFER_TOO_SMALL;
    }

    stream_words[word_index++] = BITNET_ACCEL_MAGIC;
    stream_words[word_index++] = out_features;
    stream_words[word_index++] = in_features;
    stream_words[word_index++] = encode_gemm_flags(flags, batch);
    while (word_index < BITNET_ACCEL_HEADER_WORDS) {
        stream_words[word_index++] = 0u;
    }

    for (uint32_t row = 0u; row < batch; row++) {
        uint32_t row_start = row * in_features;
        uint32_t row_word_limit = word_index +
            (align_input_beat(in_features) / sizeof(uint32_t));

        for (uint32_t k = 0u; k < in_features; k += 4u) {
            uint32_t word = 0u;
            for (uint32_t b = 0u; b < 4u; b++) {
                uint32_t idx = k + b;
                if (idx < in_features) {
                    word |= ((uint32_t)encode_i8(activations[row_start + idx])) << (8u * b);
                }
            }
            stream_words[word_index++] = word;
        }
        while (word_index < row_word_limit) {
            stream_words[word_index++] = 0u;
        }
    }

    status = build_weight_stream_words(stream_words + word_index,
                                       stream_word_capacity - word_index,
                                       packed_weights, out_features, in_features,
                                       &weight_word_count);
    if (status != XST_SUCCESS) {
        return status;
    }
    word_index += weight_word_count;

    return XST_SUCCESS;
}

int bitnet_accel_run(
    XAxiDma *dma,
    uint32_t *tx_stream,
    uint32_t tx_words,
    int32_t *rx_outputs,
    uint32_t out_features
) {
    int status;
    UINTPTR tx_addr = (UINTPTR)tx_stream;
    UINTPTR rx_addr = (UINTPTR)rx_outputs;
    uint32_t tx_bytes = tx_words * sizeof(uint32_t);
    uint32_t rx_bytes = out_features * sizeof(int32_t);

    Xil_DCacheFlushRange(tx_addr, tx_bytes);
    Xil_DCacheFlushRange(rx_addr, rx_bytes);

    status = XAxiDma_SimpleTransfer(dma, rx_addr, rx_bytes, XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = XAxiDma_SimpleTransfer(dma, tx_addr, tx_bytes, XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = wait_dma_idle_checked(dma, XAXIDMA_DMA_TO_DEVICE,
                                   "single-input");
    if (status != XST_SUCCESS) {
        return status;
    }
    status = wait_dma_idle_checked(dma, XAXIDMA_DEVICE_TO_DMA,
                                   "single-output");
    if (status != XST_SUCCESS) {
        return status;
    }

    Xil_DCacheInvalidateRange(rx_addr, rx_bytes);
    return XST_SUCCESS;
}

int bitnet_accel_run_physical(
    XAxiDma *dma,
    uint64_t packet_addr,
    uint32_t packet_words,
    int32_t *rx_outputs,
    uint32_t output_words
) {
    UINTPTR rx_addr = (UINTPTR)rx_outputs;
    uint32_t tx_bytes;
    uint32_t rx_bytes;
    int status;

    if (!dma || !packet_addr || !packet_words || !rx_outputs || !output_words ||
        packet_words > (UINT32_MAX / sizeof(uint32_t)) ||
        output_words > (UINT32_MAX / sizeof(uint32_t))) {
        return XST_INVALID_PARAM;
    }
    tx_bytes = packet_words * sizeof(uint32_t);
    rx_bytes = output_words * sizeof(uint32_t);

    Xil_DCacheFlushRange(rx_addr, rx_bytes);
    status = XAxiDma_SimpleTransfer(
        dma, rx_addr, rx_bytes, XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) {
        return status;
    }
#if defined(BITNET_LINUX)
    status = XAxiDma_SimpleTransferPhysical(
        dma, packet_addr, tx_bytes, XAXIDMA_DMA_TO_DEVICE);
#else
    status = XAxiDma_SimpleTransfer(
        dma, (UINTPTR)packet_addr, tx_bytes, XAXIDMA_DMA_TO_DEVICE);
#endif
    if (status != XST_SUCCESS) {
        return status;
    }

    status = wait_dma_idle_checked(dma, XAXIDMA_DMA_TO_DEVICE,
                                   "single-plddr-input");
    if (status != XST_SUCCESS) {
        return status;
    }
    status = wait_dma_idle_checked(dma, XAXIDMA_DEVICE_TO_DMA,
                                   "single-plddr-output");
    if (status != XST_SUCCESS) {
        return status;
    }

    Xil_DCacheInvalidateRange(rx_addr, rx_bytes);
    return XST_SUCCESS;
}

static int submit_physical_mm2s(
    XAxiDma *dma,
    uint64_t packet_addr,
    uint32_t packet_bytes
) {
#ifdef BITNET_LINUX
    return XAxiDma_SimpleTransferPhysical(
        dma, packet_addr, packet_bytes, XAXIDMA_DMA_TO_DEVICE);
#else
    return XAxiDma_SimpleTransfer(
        dma, (UINTPTR)packet_addr, packet_bytes, XAXIDMA_DMA_TO_DEVICE);
#endif
}

int bitnet_accel_run_dual_physical(
    XAxiDma *dma0,
    XAxiDma *dma1,
    uint64_t lane0_packet_addr,
    uint32_t lane0_packet_words,
    uint64_t lane1_packet_addr,
    uint32_t lane1_packet_words,
    int32_t *rx_outputs,
    uint32_t output_words
) {
    UINTPTR rx_addr = (UINTPTR)rx_outputs;
    uint32_t lane0_bytes;
    uint32_t lane1_bytes;
    uint32_t rx_bytes;
    int status;

    if (!dma0 || !dma1 || !lane0_packet_addr || !lane1_packet_addr ||
        !lane0_packet_words || !lane1_packet_words || !rx_outputs ||
        !output_words || lane0_packet_words > (UINT32_MAX / sizeof(uint32_t)) ||
        lane1_packet_words > (UINT32_MAX / sizeof(uint32_t)) ||
        output_words > (UINT32_MAX / sizeof(uint32_t))) {
        return XST_INVALID_PARAM;
    }
    lane0_bytes = lane0_packet_words * sizeof(uint32_t);
    lane1_bytes = lane1_packet_words * sizeof(uint32_t);
    rx_bytes = output_words * sizeof(uint32_t);

    Xil_DCacheFlushRange(rx_addr, rx_bytes);
    status = XAxiDma_SimpleTransfer(
        dma0, rx_addr, rx_bytes, XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) {
        return status;
    }

    /* Arm both MM2S channels after S2MM.  The dual RTL holds lane 1's output
     * behind lane-0 TLAST, but accepts lane-1 input concurrently so the two
     * PL compute paths overlap safely. */
    status = submit_physical_mm2s(dma0, lane0_packet_addr, lane0_bytes);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = submit_physical_mm2s(dma1, lane1_packet_addr, lane1_bytes);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = wait_dma_idle_checked(dma0, XAXIDMA_DMA_TO_DEVICE,
                                   "dual-lane0-input");
    if (status != XST_SUCCESS) {
        return status;
    }

    status = wait_dma_idle_checked(dma1, XAXIDMA_DMA_TO_DEVICE,
                                   "dual-lane1-input");
    if (status != XST_SUCCESS) {
        return status;
    }
    status = wait_dma_idle_checked(dma0, XAXIDMA_DEVICE_TO_DMA,
                                   "dual-output");
    if (status != XST_SUCCESS) {
        return status;
    }

    Xil_DCacheInvalidateRange(rx_addr, rx_bytes);
    return XST_SUCCESS;
}

int bitnet_accel_run_gemv_chunked(
    XAxiDma *dma,
    uint32_t *stream_words,
    uint32_t stream_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t chunk_out_features,
    uint32_t flags,
    int32_t *rx_outputs
) {
    uint32_t chunk = normalize_chunk_out(out_features, chunk_out_features);

    if ((dma == 0) || (stream_words == 0) || (activations == 0) ||
        (packed_weights == 0) || (rx_outputs == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (chunk == 0u)) {
        return XST_INVALID_PARAM;
    }
    if (bitnet_accel_stream_words(chunk, in_features) > stream_word_capacity) {
        return XST_BUFFER_TOO_SMALL;
    }

    for (uint32_t out_base = 0u; out_base < out_features; out_base += chunk) {
        uint32_t remaining = out_features - out_base;
        uint32_t this_out = (remaining > chunk) ? chunk : remaining;
        uint32_t weight_byte_offset = (out_base / 4u) * in_features;
        uint32_t tx_words = bitnet_accel_stream_words(this_out, in_features);
        int status;

        status = bitnet_accel_build_stream(stream_words,
                                           stream_word_capacity,
                                           activations,
                                           packed_weights + weight_byte_offset,
                                           this_out,
                                           in_features,
                                           flags);
        if (status != XST_SUCCESS) {
            return status;
        }

        status = bitnet_accel_run(dma,
                                  stream_words,
                                  tx_words,
                                  rx_outputs + out_base,
                                  this_out);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    return XST_SUCCESS;
}

static int wait_for_dma_idle(XAxiDma *dma, int direction) {
    return wait_dma_idle_checked(dma, direction, "streamed-transfer");
}

int bitnet_accel_build_prefix(
    uint32_t *prefix_words,
    uint32_t prefix_word_capacity,
    const int8_t *activations,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags,
    uint32_t *prefix_word_count
) {
    uint32_t act_bytes = (in_features + BITNET_ACCEL_INPUT_BYTES_PER_BEAT - 1u) &
                         ~(BITNET_ACCEL_INPUT_BYTES_PER_BEAT - 1u);
    uint32_t act_words = act_bytes / sizeof(uint32_t);
    uint32_t required = BITNET_ACCEL_HEADER_WORDS + act_words;
    uint32_t word_index = 0u;

    if ((prefix_words == 0) || (activations == 0) || (prefix_word_count == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (prefix_word_capacity < required)) {
        return XST_INVALID_PARAM;
    }

    prefix_words[word_index++] = BITNET_ACCEL_MAGIC;
    prefix_words[word_index++] = out_features;
    prefix_words[word_index++] = in_features;
    prefix_words[word_index++] = flags;
    while (word_index < BITNET_ACCEL_HEADER_WORDS) {
        prefix_words[word_index++] = 0u;
    }

    for (uint32_t k = 0u; k < in_features; k += 4u) {
        uint32_t word = 0u;
        for (uint32_t b = 0u; b < 4u; b++) {
            uint32_t idx = k + b;
            if (idx < in_features) {
                word |= ((uint32_t)encode_i8(activations[idx])) << (8u * b);
            }
        }
        prefix_words[word_index++] = word;
    }
    while (word_index < required) {
        prefix_words[word_index++] = 0u;
    }

    *prefix_word_count = word_index;
    return XST_SUCCESS;
}

int bitnet_accel_build_gemm_prefix(
    uint32_t *prefix_words,
    uint32_t prefix_word_capacity,
    const int8_t *activations,
    uint32_t batch_rows,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags,
    uint32_t *prefix_word_count
) {
    uint32_t batch = normalize_batch_rows(batch_rows);
    uint32_t act_bytes = (in_features + BITNET_ACCEL_INPUT_BYTES_PER_BEAT - 1u) &
                         ~(BITNET_ACCEL_INPUT_BYTES_PER_BEAT - 1u);
    uint32_t act_words = act_bytes / sizeof(uint32_t);
    uint32_t required = BITNET_ACCEL_HEADER_WORDS + (act_words * batch);
    uint32_t word_index = 0u;

    if ((prefix_words == 0) || (activations == 0) || (prefix_word_count == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (batch == 0u) ||
        (prefix_word_capacity < required)) {
        return XST_INVALID_PARAM;
    }

    prefix_words[word_index++] = BITNET_ACCEL_MAGIC;
    prefix_words[word_index++] = out_features;
    prefix_words[word_index++] = in_features;
    prefix_words[word_index++] = encode_gemm_flags(flags, batch);
    while (word_index < BITNET_ACCEL_HEADER_WORDS) {
        prefix_words[word_index++] = 0u;
    }

    for (uint32_t row = 0u; row < batch; row++) {
        uint32_t row_start = row * in_features;
        uint32_t row_word_limit = word_index + act_words;

        for (uint32_t k = 0u; k < in_features; k += 4u) {
            uint32_t word = 0u;
            for (uint32_t b = 0u; b < 4u; b++) {
                uint32_t idx = k + b;
                if (idx < in_features) {
                    word |= ((uint32_t)encode_i8(activations[row_start + idx])) << (8u * b);
                }
            }
            prefix_words[word_index++] = word;
        }
        while (word_index < row_word_limit) {
            prefix_words[word_index++] = 0u;
        }
    }

    *prefix_word_count = word_index;
    return XST_SUCCESS;
}

int bitnet_accel_run_gemv_chunked_direct(
    XAxiDma *dma,
    uint32_t *prefix_words,
    uint32_t prefix_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t chunk_out_features,
    uint32_t flags,
    int32_t *rx_outputs,
    uint32_t flush_packed_weights
) {
    uint32_t chunk = normalize_chunk_out(out_features, chunk_out_features);

    if ((dma == 0) || (prefix_words == 0) || (activations == 0) ||
        (packed_weights == 0) || (rx_outputs == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (chunk == 0u)) {
        return XST_INVALID_PARAM;
    }

    for (uint32_t out_base = 0u; out_base < out_features; out_base += chunk) {
        uint32_t remaining = out_features - out_base;
        uint32_t this_out = (remaining > chunk) ? chunk : remaining;
        uint32_t weight_byte_offset = (out_base / 4u) * in_features;
        uint32_t weight_bytes = (this_out / 4u) * in_features;
        uint32_t prefix_word_count;
        uint32_t weight_word_count;
        uint32_t direct_weight_stream = weights_are_input_beat_aligned(in_features);
        UINTPTR prefix_addr = (UINTPTR)prefix_words;
        UINTPTR weight_addr = (UINTPTR)(packed_weights + weight_byte_offset);
        UINTPTR rx_addr = (UINTPTR)(rx_outputs + out_base);
        uint32_t rx_bytes = this_out * sizeof(int32_t);
        int status;

        status = bitnet_accel_build_prefix(prefix_words, prefix_word_capacity,
                                   activations, this_out, in_features, flags,
                                   &prefix_word_count);
        if (status != XST_SUCCESS) {
            return status;
        }

        Xil_DCacheFlushRange(prefix_addr, prefix_word_count * sizeof(uint32_t));
        if (flush_packed_weights != 0u) {
            Xil_DCacheFlushRange(weight_addr, weight_bytes);
        }
        Xil_DCacheFlushRange(rx_addr, rx_bytes);

        status = XAxiDma_SimpleTransfer(dma, rx_addr, rx_bytes, XAXIDMA_DEVICE_TO_DMA);
        if (status != XST_SUCCESS) {
            return status;
        }

        if (direct_weight_stream != 0u) {
            status = XAxiDma_SimpleTransfer(dma, prefix_addr,
                                            prefix_word_count * sizeof(uint32_t),
                                            XAXIDMA_DMA_TO_DEVICE);
            if (status != XST_SUCCESS) {
                return status;
            }
            status = wait_for_dma_idle(dma, XAXIDMA_DMA_TO_DEVICE);
            if (status != XST_SUCCESS) {
                return status;
            }
            status = XAxiDma_SimpleTransfer(dma, weight_addr, weight_bytes,
                                            XAXIDMA_DMA_TO_DEVICE);
        } else {
            status = XAxiDma_SimpleTransfer(dma, prefix_addr,
                                            prefix_word_count * sizeof(uint32_t),
                                            XAXIDMA_DMA_TO_DEVICE);
            if (status != XST_SUCCESS) {
                return status;
            }
            status = wait_for_dma_idle(dma, XAXIDMA_DMA_TO_DEVICE);
            if (status != XST_SUCCESS) {
                return status;
            }
            status = build_weight_stream_words(prefix_words, prefix_word_capacity,
                                               packed_weights + weight_byte_offset,
                                               this_out, in_features,
                                               &weight_word_count);
            if (status != XST_SUCCESS) {
                return status;
            }
            Xil_DCacheFlushRange(prefix_addr, weight_word_count * sizeof(uint32_t));

            status = XAxiDma_SimpleTransfer(dma, prefix_addr,
                                            weight_word_count * sizeof(uint32_t),
                                            XAXIDMA_DMA_TO_DEVICE);
        }
        if (status != XST_SUCCESS) {
            return status;
        }
        status = wait_for_dma_idle(dma, XAXIDMA_DMA_TO_DEVICE);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = wait_for_dma_idle(dma, XAXIDMA_DEVICE_TO_DMA);
        if (status != XST_SUCCESS) {
            return status;
        }

        Xil_DCacheInvalidateRange(rx_addr, rx_bytes);
    }

    return XST_SUCCESS;
}

int bitnet_accel_run_gemm_chunked_direct(
    XAxiDma *dma,
    uint32_t *prefix_words,
    uint32_t prefix_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t batch_rows,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t chunk_out_features,
    uint32_t flags,
    int32_t *rx_outputs,
    uint32_t flush_packed_weights
) {
    uint32_t batch = normalize_batch_rows(batch_rows);
    uint32_t chunk = normalize_chunk_out(out_features, chunk_out_features);

    if ((dma == 0) || (prefix_words == 0) || (activations == 0) ||
        (packed_weights == 0) || (rx_outputs == 0)) {
        return XST_INVALID_PARAM;
    }
    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (batch == 0u) || (chunk == 0u)) {
        return XST_INVALID_PARAM;
    }

    for (uint32_t out_base = 0u; out_base < out_features; out_base += chunk) {
        uint32_t remaining = out_features - out_base;
        uint32_t this_out = (remaining > chunk) ? chunk : remaining;
        uint32_t weight_byte_offset = (out_base / 4u) * in_features;
        uint32_t weight_bytes = (this_out / 4u) * in_features;
        uint32_t prefix_word_count;
        uint32_t weight_word_count;
        uint32_t direct_weight_stream = weights_are_input_beat_aligned(in_features);
        UINTPTR prefix_addr = (UINTPTR)prefix_words;
        UINTPTR weight_addr = (UINTPTR)(packed_weights + weight_byte_offset);
        UINTPTR rx_addr = (UINTPTR)(rx_outputs + ((out_base / 4u) * batch * 4u));
        uint32_t rx_bytes = batch * this_out * sizeof(int32_t);
        int status;

        status = bitnet_accel_build_gemm_prefix(prefix_words, prefix_word_capacity,
                                   activations, batch, this_out, in_features, flags,
                                   &prefix_word_count);
        if (status != XST_SUCCESS) {
            return status;
        }

        Xil_DCacheFlushRange(prefix_addr, prefix_word_count * sizeof(uint32_t));
        if (flush_packed_weights != 0u) {
            Xil_DCacheFlushRange(weight_addr, weight_bytes);
        }
        Xil_DCacheFlushRange(rx_addr, rx_bytes);

        status = XAxiDma_SimpleTransfer(dma, rx_addr, rx_bytes, XAXIDMA_DEVICE_TO_DMA);
        if (status != XST_SUCCESS) {
            return status;
        }

        status = XAxiDma_SimpleTransfer(dma, prefix_addr,
                                        prefix_word_count * sizeof(uint32_t),
                                        XAXIDMA_DMA_TO_DEVICE);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = wait_for_dma_idle(dma, XAXIDMA_DMA_TO_DEVICE);
        if (status != XST_SUCCESS) {
            return status;
        }

        if (direct_weight_stream != 0u) {
            status = XAxiDma_SimpleTransfer(dma, weight_addr, weight_bytes,
                                            XAXIDMA_DMA_TO_DEVICE);
        } else {
            status = build_weight_stream_words(prefix_words, prefix_word_capacity,
                                               packed_weights + weight_byte_offset,
                                               this_out, in_features,
                                               &weight_word_count);
            if (status != XST_SUCCESS) {
                return status;
            }
            Xil_DCacheFlushRange(prefix_addr, weight_word_count * sizeof(uint32_t));

            status = XAxiDma_SimpleTransfer(dma, prefix_addr,
                                            weight_word_count * sizeof(uint32_t),
                                            XAXIDMA_DMA_TO_DEVICE);
        }
        if (status != XST_SUCCESS) {
            return status;
        }
        status = wait_for_dma_idle(dma, XAXIDMA_DMA_TO_DEVICE);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = wait_for_dma_idle(dma, XAXIDMA_DEVICE_TO_DMA);
        if (status != XST_SUCCESS) {
            return status;
        }

        Xil_DCacheInvalidateRange(rx_addr, rx_bytes);
    }

    return XST_SUCCESS;
}

int32_t bitnet_accel_reference_dot4_layout(
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t out_channel,
    uint32_t in_features,
    uint32_t flags
) {
    uint32_t group = out_channel / 4u;
    uint32_t lane = out_channel & 0x3u;
    int32_t acc = 0;

    for (uint32_t k = 0u; k < in_features; k++) {
        uint8_t packed = packed_weights[group * in_features + k];
        uint8_t code = (packed >> (2u * lane)) & 0x3u;
        acc += decode_contrib(activations[k], code, flags);
    }

    return acc;
}

int32_t bitnet_accel_reference_gemm_dot4_layout(
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t row,
    uint32_t out_channel,
    uint32_t in_features,
    uint32_t flags
) {
    return bitnet_accel_reference_dot4_layout(
        activations + (row * in_features),
        packed_weights,
        out_channel,
        in_features,
        flags);
}
