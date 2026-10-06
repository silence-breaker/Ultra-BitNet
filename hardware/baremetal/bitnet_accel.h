#ifndef BITNET_ACCEL_H
#define BITNET_ACCEL_H

#include <stdint.h>
#include "xaxidma.h"

#define BITNET_ACCEL_MAGIC 0x42544e31u
/* GEMV is widened to 512 bits.  Keep the header in its own 512-bit beat so
 * the AXIS width converter cannot merge header fields with the first row. */
#define BITNET_ACCEL_HEADER_WORDS 16u
#define BITNET_ACCEL_INPUT_BYTES_PER_BEAT 64u
#define BITNET_ACCEL_MAP_GPU_PLUS2 0u
#define BITNET_ACCEL_MAP_HF_PACKED 1u
#define BITNET_ACCEL_GEMM_MAX_BATCH_ROWS 16u
#define BITNET_ACCEL_GEMM_BATCH_SHIFT 8u
#define BITNET_ACCEL_GEMM_BATCH_MASK 0xFFu

uint32_t bitnet_accel_stream_words(uint32_t out_features, uint32_t in_features);

uint32_t bitnet_accel_gemm_stream_words(
    uint32_t out_features,
    uint32_t in_features,
    uint32_t batch_rows
);

uint32_t bitnet_accel_chunk_count(
    uint32_t out_features,
    uint32_t chunk_out_features
);

uint32_t bitnet_accel_chunked_stream_words(
    uint32_t out_features,
    uint32_t in_features,
    uint32_t chunk_out_features
);

int bitnet_accel_build_stream(
    uint32_t *stream_words,
    uint32_t stream_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags
);

int bitnet_accel_build_gemm_stream(
    uint32_t *stream_words,
    uint32_t stream_word_capacity,
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t batch_rows,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags
);

/* Build only the packet header and activation prefix.  The weight payload is
 * intentionally omitted so callers can keep a pre-packed copy in PL DDR and
 * update only the per-token activation data. */
int bitnet_accel_build_prefix(
    uint32_t *prefix_words,
    uint32_t prefix_word_capacity,
    const int8_t *activations,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags,
    uint32_t *prefix_word_count
);

int bitnet_accel_build_gemm_prefix(
    uint32_t *prefix_words,
    uint32_t prefix_word_capacity,
    const int8_t *activations,
    uint32_t batch_rows,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags,
    uint32_t *prefix_word_count
);

int bitnet_accel_run(
    XAxiDma *dma,
    uint32_t *tx_stream,
    uint32_t tx_words,
    int32_t *rx_outputs,
    uint32_t out_features
);

/* Submit one prebuilt packet from a physical PL-DDR buffer.  This is the
 * single-lane fallback for shapes that cannot be split evenly across both
 * lanes; the PLDDR design routes DMA0 MM2S exclusively to PL DDR. */
int bitnet_accel_run_physical(
    XAxiDma *dma,
    uint64_t packet_addr,
    uint32_t packet_words,
    int32_t *rx_outputs,
    uint32_t output_words
);

/* Submit two prebuilt BTN1 packets from independent physical buffers.  The
 * dual-lane RTL concatenates lane0 and lane1 output into one DMA0 S2MM frame. */
int bitnet_accel_run_dual_physical(
    XAxiDma *dma0,
    XAxiDma *dma1,
    uint64_t lane0_packet_addr,
    uint32_t lane0_packet_words,
    uint64_t lane1_packet_addr,
    uint32_t lane1_packet_words,
    int32_t *rx_outputs,
    uint32_t output_words
);

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
);

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
);

/*
 * GEMM output is stream-major for hardware efficiency:
 *   rx_outputs[((out_channel / 4) * batch_rows * 4) + (row * 4) + lane]
 * where lane = out_channel & 3.  For batch_rows == 1 this is identical to the
 * original GEMV channel order.  flags[15:8] carries batch_rows on the
 * AXI-Stream header; a zero batch field remains backward-compatible with the
 * original GEMV protocol and means one row.
 */
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
);

int32_t bitnet_accel_reference_dot4_layout(
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t out_channel,
    uint32_t in_features,
    uint32_t flags
);

int32_t bitnet_accel_reference_gemm_dot4_layout(
    const int8_t *activations,
    const uint8_t *packed_weights,
    uint32_t row,
    uint32_t out_channel,
    uint32_t in_features,
    uint32_t flags
);

#endif
