#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bitnet_client.h"
#include "bitnet_protocol.h"

int bitnet_core_main(void);

int bitnet_client_infer(const char *prompt, size_t prompt_length,
                        const BitnetGenerationOptions *options,
                        BitnetResponse *response) {
    BitnetInferControl *control =
        (BitnetInferControl *)(uintptr_t)BITNET_INFER_CONTROL_BASE;
    BitnetInferOutput *output =
        (BitnetInferOutput *)(uintptr_t)BITNET_INFER_OUTPUT_BASE;
    int result;

    if (!prompt || !options || !response || prompt_length == 0u ||
        prompt_length > BITNET_INFER_PROMPT_MAX_BYTES) {
        return -1;
    }

    memset(control, 0, sizeof(*control));
    memset(output, 0, sizeof(*output));
    memcpy((void *)(uintptr_t)BITNET_INFER_PROMPT_BASE, prompt, prompt_length);

    control->magic = BITNET_INFER_CONTROL_MAGIC;
    control->version = BITNET_INFER_CONTROL_VERSION;
    control->mode = BITNET_INFER_MODE_DDR_TEXT;
    control->prompt_addr = (uint32_t)BITNET_INFER_PROMPT_BASE;
    control->prompt_length = (uint32_t)prompt_length;
    control->first_layer = 0u;
    control->num_layers = options->num_layers;
    control->activation_seed = 23u;
    control->flags = 1u;
    control->max_new_tokens = options->max_new_tokens;
    control->temperature_milli = options->temperature_milli;
    control->top_k = options->top_k;
    control->prefill_batch_rows = options->prefill_batch_rows;

    result = bitnet_core_main();
    if (result != 0 || output->magic != BITNET_INFER_OUTPUT_MAGIC ||
        output->version != BITNET_INFER_OUTPUT_VERSION ||
        output->byte_count > BITNET_INFER_OUTPUT_MAX_BYTES) {
        fprintf(stderr, "推理失败: result=%d output_magic=0x%08x\n",
                result, output->magic);
        return -1;
    }

    response->bytes = output->bytes;
    response->byte_count = output->byte_count;
    response->token_count = output->token_count;
    return 0;
}
