#ifndef BITNET_CLIENT_H
#define BITNET_CLIENT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t max_new_tokens;
    uint32_t num_layers;
    uint32_t temperature_milli;
    uint32_t top_k;
    uint32_t prefill_batch_rows;
} BitnetGenerationOptions;

typedef struct {
    const uint8_t *bytes;
    uint32_t byte_count;
    uint32_t token_count;
} BitnetResponse;

int bitnet_client_infer(const char *prompt, size_t prompt_length,
                        const BitnetGenerationOptions *options,
                        BitnetResponse *response);

#endif
