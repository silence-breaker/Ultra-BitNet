#ifndef BITNET_PROTOCOL_H
#define BITNET_PROTOCOL_H

#include <stdint.h>

#define BITNET_INFER_CONTROL_BASE ((uintptr_t)0x4B000000u)
#define BITNET_INFER_PROMPT_BASE ((uintptr_t)0x4B010000u)
#define BITNET_INFER_OUTPUT_BASE ((uintptr_t)0x64630000u)

#define BITNET_INFER_CONTROL_MAGIC 0x42494631u
#define BITNET_INFER_CONTROL_VERSION 1u
#define BITNET_INFER_MODE_UART_TEXT 1u
#define BITNET_INFER_MODE_DDR_TEXT 2u
#define BITNET_INFER_MODE_DDR_TOKEN_IDS 3u
#define BITNET_INFER_PREFILL_BATCH_DEFAULT 16u
#define BITNET_INFER_PREFILL_BATCH_MAX 16u

#define BITNET_INFER_PROMPT_MAX_BYTES 2048u
#define BITNET_INFER_OUTPUT_MAGIC 0x314F4942u
#define BITNET_INFER_OUTPUT_VERSION 1u
#define BITNET_INFER_OUTPUT_MAX_BYTES 8192u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t mode;
    uint32_t prompt_addr;
    uint32_t prompt_length;
    uint32_t first_layer;
    uint32_t num_layers;
    uint32_t activation_seed;
    uint32_t flags;
    uint32_t max_new_tokens;
    uint32_t temperature_milli;
    uint32_t top_k;
    uint32_t prefill_batch_rows;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
} BitnetInferControl;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t byte_count;
    uint32_t token_count;
    uint8_t bytes[BITNET_INFER_OUTPUT_MAX_BYTES];
} BitnetInferOutput;

typedef char BitnetInferControlSizeMustBe64[
    (sizeof(BitnetInferControl) == 64u) ? 1 : -1];
typedef char BitnetInferOutputHeaderSizeMustBe16[
    (sizeof(BitnetInferOutput) == (16u + BITNET_INFER_OUTPUT_MAX_BYTES)) ? 1 : -1];

#endif
