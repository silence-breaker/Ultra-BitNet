#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitnet_client.h"
#include "bitnet_kernels.h"
#include "bitnet_protocol.h"
#include "prompt_source.h"
#include "xaxidma.h"

typedef enum {
    PROMPT_NONE,
    PROMPT_ARGUMENT,
    PROMPT_FILE,
    PROMPT_STDIO
} PromptMode;

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: sudo %s --model /opt/bitnet <prompt source> [options]\n"
        "Prompt source (select exactly one):\n"
        "  --prompt TEXT       Run one inference from the command line\n"
        "  --prompt-file PATH  Read one UTF-8 prompt file\n"
        "  --stdio             Process one prompt per line from SSH or UART\n"
        "Generation options:\n"
        "  --tokens N          Number of generated tokens, default 8\n"
        "  --layers N          Number of layers, default 30\n"
        "  --temperature N     Temperature in milli-units, default 0\n"
        "  --top-k N           Top-k value, default 1\n"
        "  --prefill-batch N   Prefill GEMM rows, 1..16, default 16\n",
        program);
}

static uint32_t parse_u32(const char *name, const char *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);

    if (!value[0] || !end || *end || parsed > UINT32_MAX) {
        fprintf(stderr, "%s has an invalid value: %s\n", name, value);
        exit(2);
    }
    return (uint32_t)parsed;
}

static int set_prompt_mode(PromptMode *mode, PromptMode requested) {
    if (*mode != PROMPT_NONE) {
        fprintf(stderr, "Select exactly one prompt source\n");
        return -1;
    }
    *mode = requested;
    return 0;
}

static int emit_response(const char *prompt, size_t prompt_length,
                         const BitnetGenerationOptions *options) {
    BitnetResponse response;

    if (bitnet_client_infer(prompt, prompt_length, options, &response) != 0) {
        return -1;
    }
    printf("BITNET_RESPONSE_BEGIN tokens=%u bytes=%u\n",
           response.token_count, response.byte_count);
    fwrite(response.bytes, 1u, response.byte_count, stdout);
    printf("\nBITNET_RESPONSE_END\n");
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_root = "/opt/bitnet";
    const char *prompt_argument = NULL;
    const char *prompt_file = NULL;
    PromptMode prompt_mode = PROMPT_NONE;
    BitnetGenerationOptions options = {
        .max_new_tokens = 8u,
        .num_layers = 30u,
        .temperature_milli = 0u,
        .top_k = 1u,
        .prefill_batch_rows = BITNET_INFER_PREFILL_BATCH_DEFAULT
    };
    char prompt[BITNET_INFER_PROMPT_MAX_BYTES + 1u];
    size_t prompt_length = 0u;
    int exit_code = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_root = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            if (set_prompt_mode(&prompt_mode, PROMPT_ARGUMENT) != 0) return 2;
            prompt_argument = argv[++i];
        } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            if (set_prompt_mode(&prompt_mode, PROMPT_FILE) != 0) return 2;
            prompt_file = argv[++i];
        } else if (strcmp(argv[i], "--stdio") == 0) {
            if (set_prompt_mode(&prompt_mode, PROMPT_STDIO) != 0) return 2;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            options.max_new_tokens = parse_u32("--tokens", argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            options.num_layers = parse_u32("--layers", argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            options.temperature_milli = parse_u32("--temperature", argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            options.top_k = parse_u32("--top-k", argv[++i]);
        } else if (strcmp(argv[i], "--prefill-batch") == 0 && i + 1 < argc) {
            options.prefill_batch_rows = parse_u32("--prefill-batch", argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (prompt_mode == PROMPT_NONE || options.num_layers == 0u ||
        options.num_layers > 30u || options.max_new_tokens == 0u ||
        options.top_k == 0u || options.top_k > BITNET_LM_HEAD_TOPK_MAX ||
        options.prefill_batch_rows == 0u ||
        options.prefill_batch_rows > BITNET_INFER_PREFILL_BATCH_MAX) {
        usage(argv[0]);
        return 2;
    }
    if (bitnet_linux_runtime_init(model_root) != 0) {
        fprintf(stderr, "Linux FPGA runtime initialization failed\n");
        return 1;
    }

    if (prompt_mode == PROMPT_ARGUMENT) {
        prompt_length = strlen(prompt_argument);
        if (prompt_length == 0u || prompt_length > BITNET_INFER_PROMPT_MAX_BYTES ||
            emit_response(prompt_argument, prompt_length, &options) != 0) {
            exit_code = 1;
        }
    } else if (prompt_mode == PROMPT_FILE) {
        if (bitnet_prompt_read_file(prompt_file, prompt,
                                    BITNET_INFER_PROMPT_MAX_BYTES,
                                    &prompt_length) != 0 ||
            emit_response(prompt, prompt_length, &options) != 0) {
            exit_code = 1;
        }
    } else {
        for (;;) {
            int read_status = bitnet_prompt_read_line(
                stdin, prompt, BITNET_INFER_PROMPT_MAX_BYTES, &prompt_length);
            if (read_status == 0) break;
            if (read_status < 0) {
                exit_code = 1;
                continue;
            }
            if (prompt_length == 0u) continue;
            if (emit_response(prompt, prompt_length, &options) != 0) {
                exit_code = 1;
            }
        }
    }

    bitnet_linux_runtime_close();
    return exit_code;
}
