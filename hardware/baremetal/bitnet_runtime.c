#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#if defined(BITNET_LINUX)
#include <pthread.h>
#include <stdlib.h>
#endif
#include "bitnet_accel.h"
#include "bitnet_kernels.h"
#include "bitnet_protocol.h"
#include "ff.h"
#include "xaxidma.h"
#include "xaxidma_hw.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xtime_l.h"

#define TEST_K 8u
#define TEST_N 8u
#define TEST_CHUNK_OUT 4u
#define TEST_GEMM_BATCH 3u
#define TX_WORD_CAP 128u

#define INSTALL_CONTROL_BASE ((uintptr_t)0x4C000000u)
#define INSTALL_CONTROL_MAGIC 0x42494E31u
#define INSTALL_CONTROL_VERSION 1u
#define INSTALL_OP_WRITE_FILE 1u
#define INSTALL_OP_FORMAT_FAT 2u
#define INSTALL_FORMAT_KEY 0x454D4D43u
#define INSTALL_DATA_BASE ((uintptr_t)0x52000000u)
#define INSTALL_DATA_LIMIT ((uintptr_t)0x56000000u)
#define INSTALL_PATH_MAX 128u
#define INSTALL_MKFS_WORK_BYTES (64u * 1024u)
#define INFER_CONTROL_BASE BITNET_INFER_CONTROL_BASE
#define INFER_PROMPT_BASE BITNET_INFER_PROMPT_BASE
#define INFER_CONTROL_MAGIC BITNET_INFER_CONTROL_MAGIC
#define INFER_CONTROL_VERSION BITNET_INFER_CONTROL_VERSION
#define INFER_MODE_UART_TEXT BITNET_INFER_MODE_UART_TEXT
#define INFER_MODE_DDR_TEXT BITNET_INFER_MODE_DDR_TEXT
#define INFER_MODE_DDR_TOKEN_IDS BITNET_INFER_MODE_DDR_TOKEN_IDS
#define INFER_PROMPT_MAX_BYTES BITNET_INFER_PROMPT_MAX_BYTES
#define INFER_CONTEXT_MAX_TOKENS 4096u
#define INFER_PROMPT_MAX_TOKENS INFER_CONTEXT_MAX_TOKENS
#define INFER_MAX_NEW_TOKENS 128u
#define INFER_TOTAL_MAX_TOKENS INFER_CONTEXT_MAX_TOKENS
#define MODEL_VOCAB_SIZE 128256u
#define MODEL_NUM_Q_HEADS 20u
#define MODEL_NUM_KV_HEADS 5u
#define MODEL_HEAD_DIM 128u
#define MODEL_KV_GROUP_SIZE (MODEL_NUM_Q_HEADS / MODEL_NUM_KV_HEADS)
#define ATTENTION_SCORE_DIVISOR 11
#define ATTENTION_EXP_STEP 64
#define ATTENTION_EXP_MAX_DECAY 15u
#define TOKENIZER_ASSET_BASE ((uintptr_t)0x62000000u)
#define TOKENIZER_ASSET_MAX_BYTES (32u * 1024u * 1024u)
#define TOKENIZER_ASSET_MAGIC 0x314B5442u
#define TOKENIZER_ASSET_VERSION 1u
#define TOKENIZER_DECODE_BASE ((uintptr_t)0x62800000u)
#define TOKENIZER_DECODE_MAX_BYTES (8u * 1024u * 1024u)
#define TOKENIZER_DECODE_MAGIC 0x31445442u
#define TOKENIZER_DECODE_VERSION 1u
#define TOKENIZER_DECODE_PATH "0:/BITNET/TOK/DECODE.BIN"
#define TOKENIZER_PRETOKEN_BASE ((uintptr_t)0x63000000u)
#define TOKENIZER_PRETOKEN_MAX_BYTES (2u * 1024u * 1024u)
#define TOKENIZER_PRETOKEN_MAGIC 0x31505442u
#define TOKENIZER_PRETOKEN_VERSION 1u
#define TOKENIZER_PRETOKEN_PATH "0:/BITNET/TOK/PRET.BIN"
#define TOKENIZER_CLASS_LETTER 1u
#define TOKENIZER_CLASS_NUMBER 2u
#define TOKENIZER_CLASS_SPACE 4u
#define TOKENIZER_TOKEN_ID_NONE 0xFFFFFFFFu
#define TOKENIZER_BOS_ID 128000u
#define TOKENIZER_EOS_ID 128001u
#define TOKENIZER_EOT_ID 128009u
#define TOKENIZER_PATH "0:/BITNET/TOK/TRIE.BIN"
#define MODEL_AUX_BASE ((uintptr_t)0x64000000u)
#define MODEL_AUX_MAX_BYTES (2u * 1024u * 1024u)
#define MODEL_AUX_MAGIC 0x42584131u
#define MODEL_AUX_VERSION 1u
#define MODEL_AUX_PATH "0:/BITNET/AUX/AUX.BIN"
#define MODEL_EMBED_ROW_BASE ((uintptr_t)0x64200000u)
#define MODEL_EMBED_ROW_BYTES (LAYER_HIDDEN_SIZE * 2u)
#define MODEL_EMBED_ROWS_PER_SHARD 8192u
#define MODEL_EMBED_SHARD_COUNT ((MODEL_VOCAB_SIZE + MODEL_EMBED_ROWS_PER_SHARD - 1u) / MODEL_EMBED_ROWS_PER_SHARD)
#define MODEL_EMBED_SHARD_MAX_BYTES (MODEL_EMBED_ROWS_PER_SHARD * MODEL_EMBED_ROW_BYTES)
#define MODEL_EMBED_SCALE 4096.0f
#define MODEL_NORM_SCALE 32.0f
#define ROPE_ASSET_BASE ((uintptr_t)0x64400000u)
#define ROPE_ASSET_MAX_BYTES (2u * 1024u * 1024u)
#define ROPE_ASSET_MAGIC 0x42505231u
#define ROPE_ASSET_VERSION 1u
#define ROPE_ASSET_PATH "0:/BITNET/AUX/ROPE.BIN"
#define INFER_TOKEN_IDS_BASE ((uintptr_t)0x64600000u)
#define ATTN_SCORE_BASE ((uintptr_t)0x64610000u)
#define ATTN_WEIGHT_BASE ((uintptr_t)0x64620000u)
#define INFER_OUTPUT_TEXT_BASE BITNET_INFER_OUTPUT_BASE
#define INFER_OUTPUT_TEXT_MAGIC BITNET_INFER_OUTPUT_MAGIC
#define INFER_OUTPUT_TEXT_VERSION BITNET_INFER_OUTPUT_VERSION
#define INFER_OUTPUT_TEXT_MAX_BYTES BITNET_INFER_OUTPUT_MAX_BYTES
#define LM_HEAD_SHARD_BASE ((uintptr_t)0x52000000u)
#define LM_HEAD_SHARD_MAX_BYTES (64u * 1024u * 1024u)
#define LM_HEAD_I8_RESIDENT_BASE ((uintptr_t)0x28000000u)
#define LM_HEAD_SCALE_RESIDENT_BASE ((uintptr_t)0x4A000000u)
#define LM_HEAD_I8_ROW_BYTES LAYER_HIDDEN_SIZE
#define LM_HEAD_I8_SHARD_MAX_BYTES (MODEL_EMBED_ROWS_PER_SHARD * LM_HEAD_I8_ROW_BYTES)
#define LM_HEAD_SCALE_SHARD_MAX_BYTES (MODEL_EMBED_ROWS_PER_SHARD * sizeof(float))
#define LM_HEAD_PACKED_SHARD_MAX_BYTES (LM_HEAD_I8_SHARD_MAX_BYTES + LM_HEAD_SCALE_SHARD_MAX_BYTES)
#define LM_HEAD_I8_TOTAL_BYTES (MODEL_VOCAB_SIZE * LM_HEAD_I8_ROW_BYTES)
#define LM_HEAD_SCALE_TOTAL_BYTES (MODEL_VOCAB_SIZE * sizeof(float))
#define INFER_TOPK_MAX 64u
#define EMMC_CONTROL_BASE ((uintptr_t)0x4D000000u)
#define EMMC_CONTROL_MAGIC 0x42454D31u
#define LAYER_CONTROL_BASE ((uintptr_t)0x4E000000u)
#define LAYER_CONTROL_MAGIC 0x424C4331u
#define MODEL_U8_RESIDENT_BASE ((uintptr_t)0x08000000u)
#define LAYER_WEIGHT_BASE ((uintptr_t)0x56000000u)
#define LAYER_WEIGHT_STRIDE 0x02000000u
#define LAYER_WEIGHT_Q_OFFSET 0x000000u
#define LAYER_WEIGHT_K_OFFSET 0x200000u
#define LAYER_WEIGHT_V_OFFSET 0x300000u
#define LAYER_WEIGHT_O_OFFSET 0x400000u
#define LAYER_WEIGHT_GATE_OFFSET 0x600000u
#define LAYER_WEIGHT_UP_OFFSET 0xB00000u
#define LAYER_WEIGHT_DOWN_OFFSET 0x1000000u
#define LAYER_HIDDEN_ACT_BASE ((uintptr_t)0x60000000u)
#define LAYER_ATT_ACT_BASE ((uintptr_t)0x60010000u)
#define LAYER_FFN_ACT_BASE ((uintptr_t)0x60020000u)
#define LAYER_Q_OUT_BASE ((uintptr_t)0x60030000u)
#define LAYER_K_OUT_BASE ((uintptr_t)0x60040000u)
#define LAYER_V_OUT_BASE ((uintptr_t)0x60050000u)
#define LAYER_O_OUT_BASE ((uintptr_t)0x60060000u)
#define LAYER_GATE_OUT_BASE ((uintptr_t)0x60070000u)
#define LAYER_UP_OUT_BASE ((uintptr_t)0x60080000u)
#define LAYER_DOWN_OUT_BASE ((uintptr_t)0x60090000u)
#define LAYER_Q8_BASE ((uintptr_t)0x600A0000u)
#define LAYER_K8_BASE ((uintptr_t)0x600B0000u)
#define LAYER_V8_BASE ((uintptr_t)0x600C0000u)
#define LAYER_ATT_FLOAT_BASE ((uintptr_t)0x600D0000u)
#define KV_CACHE_BASE ((uintptr_t)0x66000000u)
#define KV_CACHE_LAYER_BYTES (INFER_TOTAL_MAX_TOKENS * LAYER_KV_SIZE * 2u)
#define KV_CACHE_TOTAL_BYTES (MODEL_TOTAL_LAYERS * KV_CACHE_LAYER_BYTES)
#define KV_SCALE_BASE ((uintptr_t)0x64800000u)
#define KV_SCALE_LAYER_BYTES (INFER_TOTAL_MAX_TOKENS * 2u * sizeof(float))
#define KV_SCALE_TOTAL_BYTES (MODEL_TOTAL_LAYERS * KV_SCALE_LAYER_BYTES)
#define LAYER_HIDDEN_SIZE 2560u
#define LAYER_KV_SIZE 640u
#define LAYER_FFN_SIZE 6912u
#define MODEL_TOTAL_LAYERS 30u
#define PREFILL_BATCH_MAX BITNET_INFER_PREFILL_BATCH_MAX
#define PREFILL_WORKSPACE_BASE ((uintptr_t)0x64A00000u)
#define PREFILL_ALIGN64_BYTES(bytes) (((bytes) + 63u) & ~63u)
#define PREFILL_HIDDEN_BYTES (PREFILL_BATCH_MAX * LAYER_HIDDEN_SIZE)
#define PREFILL_NORM_BYTES PREFILL_HIDDEN_BYTES
#define PREFILL_ATT_BYTES PREFILL_HIDDEN_BYTES
#define PREFILL_FFN_BYTES (PREFILL_BATCH_MAX * LAYER_FFN_SIZE)
#define PREFILL_FACTOR_BYTES (PREFILL_BATCH_MAX * sizeof(float))
#define PREFILL_SCRATCH_BYTES (LAYER_FFN_SIZE * sizeof(float))
#define PREFILL_PROJECTION_MAX_OUT (2u * LAYER_FFN_SIZE)
#define PREFILL_PROJECTION_BYTES \
    (PREFILL_BATCH_MAX * PREFILL_PROJECTION_MAX_OUT * sizeof(int32_t))
#define PREFILL_HIDDEN_BASE PREFILL_WORKSPACE_BASE
#define PREFILL_NORM_BASE \
    (PREFILL_HIDDEN_BASE + PREFILL_ALIGN64_BYTES(PREFILL_HIDDEN_BYTES))
#define PREFILL_ATT_BASE \
    (PREFILL_NORM_BASE + PREFILL_ALIGN64_BYTES(PREFILL_NORM_BYTES))
#define PREFILL_FFN_BASE \
    (PREFILL_ATT_BASE + PREFILL_ALIGN64_BYTES(PREFILL_ATT_BYTES))
#define PREFILL_HIDDEN_FACTOR_BASE \
    (PREFILL_FFN_BASE + PREFILL_ALIGN64_BYTES(PREFILL_FFN_BYTES))
#define PREFILL_PROJECTION_FACTOR_BASE \
    (PREFILL_HIDDEN_FACTOR_BASE + PREFILL_ALIGN64_BYTES(PREFILL_FACTOR_BYTES))
#define PREFILL_SCRATCH_BASE \
    (PREFILL_PROJECTION_FACTOR_BASE + PREFILL_ALIGN64_BYTES(PREFILL_FACTOR_BYTES))
#define PREFILL_PROJECTION_STREAM_BASE \
    (PREFILL_SCRATCH_BASE + PREFILL_ALIGN64_BYTES(PREFILL_SCRATCH_BYTES))
#define PREFILL_PROJECTION_ROWS_BASE \
    (PREFILL_PROJECTION_STREAM_BASE + PREFILL_ALIGN64_BYTES(PREFILL_PROJECTION_BYTES))
#define PREFILL_WORKSPACE_END \
    (PREFILL_PROJECTION_ROWS_BASE + PREFILL_ALIGN64_BYTES(PREFILL_PROJECTION_BYTES))
#define LAYER_MAX_PRELOAD_LAYERS 4u
#define LAYER_WEIGHT_REF_RESIDENT 0x80000000u
#define LAYER_WEIGHT_REF_MASK 0x7FFFFFFFu
#define LAYER_Q_WEIGHT_BYTES ((LAYER_HIDDEN_SIZE / 4u) * LAYER_HIDDEN_SIZE)
#define LAYER_K_WEIGHT_BYTES ((LAYER_KV_SIZE / 4u) * LAYER_HIDDEN_SIZE)
#define LAYER_V_WEIGHT_BYTES ((LAYER_KV_SIZE / 4u) * LAYER_HIDDEN_SIZE)
#define LAYER_O_WEIGHT_BYTES ((LAYER_HIDDEN_SIZE / 4u) * LAYER_HIDDEN_SIZE)
#define LAYER_GATE_WEIGHT_BYTES ((LAYER_FFN_SIZE / 4u) * LAYER_HIDDEN_SIZE)
#define LAYER_UP_WEIGHT_BYTES ((LAYER_FFN_SIZE / 4u) * LAYER_HIDDEN_SIZE)
#define LAYER_DOWN_WEIGHT_BYTES ((LAYER_HIDDEN_SIZE / 4u) * LAYER_FFN_SIZE)
#define LAYER_TIGHT_Q_OFFSET 0x00000000u
#define LAYER_TIGHT_K_OFFSET (LAYER_TIGHT_Q_OFFSET + LAYER_Q_WEIGHT_BYTES)
#define LAYER_TIGHT_V_OFFSET (LAYER_TIGHT_K_OFFSET + LAYER_K_WEIGHT_BYTES)
#define LAYER_TIGHT_O_OFFSET (LAYER_TIGHT_V_OFFSET + LAYER_V_WEIGHT_BYTES)
#define LAYER_TIGHT_GATE_OFFSET (LAYER_TIGHT_O_OFFSET + LAYER_O_WEIGHT_BYTES)
#define LAYER_TIGHT_UP_OFFSET (LAYER_TIGHT_GATE_OFFSET + LAYER_GATE_WEIGHT_BYTES)
#define LAYER_TIGHT_DOWN_OFFSET (LAYER_TIGHT_UP_OFFSET + LAYER_UP_WEIGHT_BYTES)
#define LAYER_TIGHT_BYTES (LAYER_TIGHT_DOWN_OFFSET + LAYER_DOWN_WEIGHT_BYTES)
#define MODEL_U8_RESIDENT_BYTES (LAYER_TIGHT_BYTES * MODEL_TOTAL_LAYERS)
#define LAYER_DOUBLEBUF_BASE ((uintptr_t)0x42000000u)
#define LAYER_DOUBLEBUF_STRIDE ((uintptr_t)LAYER_TIGHT_BYTES)
#define LAYER_WEIGHT_REF_DOUBLEBUF 0x40000000u
#define LAYER_WEIGHT_REF_DOUBLEBUF_MASK 0x3FFFFFFFu
#define EMMC_MAX_LAYERS MODEL_TOTAL_LAYERS
#define LAYER_TENSOR_Q 1u
#define LAYER_TENSOR_K 2u
#define LAYER_TENSOR_V 3u
#define LAYER_TENSOR_O 4u
#define LAYER_TENSOR_GATE 5u
#define LAYER_TENSOR_UP 6u
#define LAYER_TENSOR_DOWN 7u

#define REAL_CONTROL_BASE ((uintptr_t)0x4F000000u)
#define REAL_TX_BASE ((uintptr_t)0x50000000u)
#define REAL_WEIGHT_BASE ((uintptr_t)0x52000000u)
#define REAL_ACT_BASE ((uintptr_t)0x54000000u)
#define REAL_OUTPUT_BASE ((uintptr_t)0x54100000u)
#define REAL_CONTROL_MAGIC 0x42574131u
#define REAL_TX_BYTES (16u * 1024u * 1024u)
#define REAL_MAX_K 8192u
#define REAL_CHUNK_OUT 8192u
#define PL_ATTENTION_INPUT_MAGIC 0x41544E31u
#define PL_ATTENTION_OUTPUT_MAGIC 0x41544E4Fu
#define PL_ATTENTION_HEADER_WORDS 6u
#define PL_ATTENTION_ROPE_WORDS (MODEL_HEAD_DIM / 2u)
#define PL_ATTENTION_QKV_WORDS (LAYER_HIDDEN_SIZE + (2u * LAYER_KV_SIZE))
#define PL_ATTENTION_CACHE_WORDS_PER_POS (2u + (LAYER_KV_SIZE / 2u))
#define PL_ATTENTION_OUTPUT_HEADER_WORDS 5u
#define PL_ATTENTION_OUTPUT_WORDS \
    (PL_ATTENTION_OUTPUT_HEADER_WORDS + (LAYER_HIDDEN_SIZE / 4u) + \
     (LAYER_KV_SIZE / 2u))
#define STATUS_SAMPLE_N TEST_N

#define SMOKE_STATUS_BASE ((uintptr_t)0x70000000u)
#define SMOKE_STATUS_MAGIC 0x42545354u
#define SMOKE_STATUS_VERSION 1u

#define SMOKE_STAGE_BOOT 1u
#define SMOKE_STAGE_DMA_LOOKUP 2u
#define SMOKE_STAGE_DMA_INIT 3u
#define SMOKE_STAGE_STREAM_BUILD 4u
#define SMOKE_STAGE_DMA_RUN 5u
#define SMOKE_STAGE_COMPARE 6u
#define SMOKE_STAGE_DONE 7u
#define SMOKE_STAGE_REAL_DETECT 8u
#define SMOKE_STAGE_REAL_LOAD 9u
#define SMOKE_STAGE_LAYER_DETECT 10u
#define SMOKE_STAGE_LAYER_RUN 11u
#define SMOKE_STAGE_EMMC_DETECT 12u
#define SMOKE_STAGE_EMMC_MOUNT 13u
#define SMOKE_STAGE_EMMC_LOAD 14u
#define SMOKE_STAGE_EMMC_RUN 15u
#define SMOKE_STAGE_EMMC_INSTALL_DETECT 16u
#define SMOKE_STAGE_EMMC_INSTALL_MOUNT 17u
#define SMOKE_STAGE_EMMC_INSTALL_WRITE 18u
#define SMOKE_STAGE_EMMC_INSTALL_DONE 19u
#define SMOKE_STAGE_INFER_DETECT 20u
#define SMOKE_STAGE_INFER_INPUT 21u
#define SMOKE_STAGE_INFER_TOKENIZE 22u
#define SMOKE_STAGE_INFER_LOAD 23u
#define SMOKE_STAGE_INFER_RUN 24u
#define SMOKE_STAGE_INFER_LOGITS 25u
#define SMOKE_STAGE_INFER_DONE 26u

#define SMOKE_RESULT_RUNNING 0u
#define SMOKE_RESULT_PASS 1u
#define SMOKE_RESULT_FAIL 0xFFFFFFFFu

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t op;
    uint32_t data_addr;
    uint32_t file_size;
    uint32_t checksum;
    uint32_t file_index;
    uint32_t file_count;
    char path[INSTALL_PATH_MAX];
} EmmcInstallControl;

typedef struct {
    uint32_t magic;
    uint32_t first_layer;
    uint32_t num_layers;
    uint32_t activation_seed;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} EmmcLayerChainControl;

typedef struct {
    uint32_t magic;
    uint32_t first_layer;
    uint32_t num_layers;
    uint32_t activation_seed;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} LayerChainControl;

typedef BitnetInferControl InferControl;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t max_token_bytes;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t nodes_offset;
    uint32_t edges_offset;
    uint32_t reserved0;
    uint32_t reserved1;
} TokenizerAssetHeader;

typedef struct {
    uint32_t child_base;
    uint16_t child_count;
    uint16_t reserved;
    uint32_t token_id;
} TokenizerNode;

typedef struct {
    uint8_t byte_value;
    uint8_t reserved0;
    uint16_t reserved1;
    uint32_t child_index;
} TokenizerEdge;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t vocab_size;
    uint32_t entries_offset;
    uint32_t payload_offset;
    uint32_t special_count;
    uint32_t reserved0;
} TokenizerDecodeHeader;

typedef struct {
    uint32_t byte_offset;
    uint16_t byte_count;
    uint16_t flags;
} TokenizerDecodeEntry;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t range_count;
    uint32_t ranges_offset;
    uint32_t letter_flag;
    uint32_t number_flag;
    uint32_t space_flag;
} TokenizerPretokHeader;

typedef struct {
    uint32_t start;
    uint32_t end;
    uint32_t flags;
} TokenizerClassRange;

typedef BitnetInferOutput InferOutputText;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t hidden_size;
    uint32_t ffn_size;
    uint32_t num_layers;
    uint32_t scale_count;
    uint32_t final_norm_offset;
    uint32_t input_norm_offset;
    uint32_t post_norm_offset;
    uint32_t attn_sub_norm_offset;
    uint32_t ffn_sub_norm_offset;
    uint32_t scale_offset;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} ModelAuxHeader;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t max_positions;
    uint32_t head_dim;
    uint32_t pair_count;
    uint32_t table_offset;
    uint32_t q15_scale;
    uint32_t theta_milli;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} RopeAssetHeader;

typedef struct {
    uint32_t cache_hits;
    uint32_t cache_misses;
    uint64_t emmc_cycles;
    uint64_t accel_cycles;
    uint64_t total_cycles;
    uint64_t prefill_cycles;
    uint64_t decode_cycles;
    uint64_t lm_head_cycles;
    uint64_t attention_block_cycles;
    uint64_t mlp_block_cycles;
    uint64_t attention_projection_cycles;
    uint64_t attention_core_cycles;
    uint64_t attention_nonlinear_cycles;
    uint64_t mlp_projection_cycles;
    uint64_t mlp_nonlinear_cycles;
    uint64_t layer_forward_cycles;
} InferPerf;

typedef struct {
    int8_t *hidden_act;
    int8_t *att_act;
    int8_t *ffn_act;
    int32_t *q_out;
    int32_t *k_out;
    int32_t *v_out;
    int32_t *o_out;
    int32_t *gate_out;
    int32_t *up_out;
    int32_t *down_out;
    float hidden_factor;
} InferBuffers;

typedef struct {
    int8_t *hidden;
    int8_t *norm;
    int8_t *attention;
    int8_t *ffn;
    float *hidden_factors;
    float *projection_factors;
    float *scratch;
    int32_t *projection_stream;
    int32_t *projection_rows;
} PrefillBatchBuffers;

typedef char PrefillWorkspaceMustNotOverlapKvCache[
    (PREFILL_WORKSPACE_END <= KV_CACHE_BASE) ? 1 : -1];

typedef struct {
    uint32_t magic;
    uint32_t out_features;
    uint32_t in_features;
    uint32_t flags;
    uint32_t weight_bytes;
    uint32_t tensor_kind;
    uint32_t activation_seed;
    uint32_t reserved;
} RealTensorControl;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t test_n;
    uint32_t test_k;
    uint32_t stage;
    uint32_t result;
    uint32_t status_code;
    uint32_t tx_words;
    uint32_t dma_mm2s_cr;
    uint32_t dma_mm2s_sr;
    uint32_t dma_s2mm_cr;
    uint32_t dma_s2mm_sr;
    uint32_t mismatches;
    uint32_t first_mismatch;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t hw[STATUS_SAMPLE_N];
    uint32_t ref[STATUS_SAMPLE_N];
    uint32_t perf[16];
} SmokeStatus;

static XAxiDma AxiDma;
#if defined(BITNET_LINUX)
#define BITNET_DUAL_UNAVAILABLE 0x7Fu
/* The PL-DDR candidate exposes a second MM2S-only DMA for the dual lane.
 * Stable bitstreams simply leave this handle unavailable and keep the
 * original single-DMA path. */
static XAxiDma AxiDmaLane1;
static uint8_t dual_lane_ready;

typedef struct {
    uint8_t valid;
    uint32_t batch_rows;
    uint32_t out_features;
    uint32_t in_features;
    uint32_t lane_words;
    uint64_t lane0_addr;
    uint64_t lane1_addr;
} PlddrPacketCacheEntry;

/* One entry per layer/tensor.  The complete packed weight packet is written
 * once to PL DDR; subsequent decode calls update only the prefix. */
static PlddrPacketCacheEntry plddr_packet_cache[MODEL_TOTAL_LAYERS][8];
static uint64_t plddr_packet_cache_next;
static uint64_t psddr_packet_cache_next;
static uint8_t split_memory_ready;
#endif
static volatile SmokeStatus * const smoke_status = (volatile SmokeStatus *)SMOKE_STATUS_BASE;
static FATFS emmc_fatfs;
static uint8_t emmc_mkfs_work[INSTALL_MKFS_WORK_BYTES] __attribute__((aligned(64)));
static uint8_t infer_resident_valid[MODEL_TOTAL_LAYERS];
static uint8_t infer_layer_streaming;
static char infer_prompt_text[INFER_PROMPT_MAX_BYTES + 1u] __attribute__((aligned(64)));
static const TokenizerAssetHeader *tokenizer_header = 0;
static const TokenizerNode *tokenizer_nodes = 0;
static const TokenizerEdge *tokenizer_edges = 0;
static const TokenizerDecodeHeader *tokenizer_decode_header = 0;
static const TokenizerDecodeEntry *tokenizer_decode_entries = 0;
static const TokenizerPretokHeader *tokenizer_pretok_header = 0;
static const TokenizerClassRange *tokenizer_class_ranges = 0;
static const ModelAuxHeader *model_aux_header = 0;
static uint8_t model_aux_available = 0u;
static const RopeAssetHeader *rope_header = 0;
static const int16_t *rope_q15_table = 0;
static uint8_t rope_available = 0u;
static uint8_t lm_head_i8_resident_ready = 0u;
static uint32_t lm_head_top_tokens[INFER_TOPK_MAX] __attribute__((aligned(64)));
static int32_t lm_head_top_scores[INFER_TOPK_MAX] __attribute__((aligned(64)));
static float lm_head_top_scores_float[INFER_TOPK_MAX] __attribute__((aligned(64)));

static uint32_t tx_stream[TX_WORD_CAP] __attribute__((aligned(64)));
static int32_t rx_outputs[TEST_N] __attribute__((aligned(64)));

static int8_t activations[TEST_K] = {
     1, -2,  3,  4, -5,  6,  7, -8
};

static int8_t gemm_activations[TEST_GEMM_BATCH * TEST_K] = {
     1, -2,  3,  4, -5,  6,  7, -8,
     2,  0, -3,  5,  1, -1,  4, -6,
    -1,  3,  0,  2, -4,  8, -7,  5
};

static int32_t gemm_outputs[TEST_GEMM_BATCH * TEST_N] __attribute__((aligned(64)));

static void smoke_status_flush(void) {
    Xil_DCacheFlushRange((UINTPTR)SMOKE_STATUS_BASE, sizeof(SmokeStatus));
}

static void smoke_status_clear(void) {
    volatile uint32_t *words = (volatile uint32_t *)SMOKE_STATUS_BASE;
    uint32_t word_count = (uint32_t)(sizeof(SmokeStatus) / sizeof(uint32_t));

    for (uint32_t i = 0u; i < word_count; i++) {
        words[i] = 0u;
    }
    smoke_status_flush();
}

static void smoke_status_update(uint32_t stage, uint32_t result, int status_code) {
    smoke_status->magic = SMOKE_STATUS_MAGIC;
    smoke_status->version = SMOKE_STATUS_VERSION;
    smoke_status->stage = stage;
    smoke_status->result = result;
    smoke_status->status_code = (uint32_t)status_code;
    smoke_status_flush();
}

static void smoke_status_set_dims(uint32_t out_features, uint32_t in_features) {
    smoke_status->test_n = out_features;
    smoke_status->test_k = in_features;
    smoke_status_flush();
}

static void smoke_status_record_dma(void) {
    if (AxiDma.RegBase == 0u) {
        return;
    }

    smoke_status->dma_mm2s_cr = XAxiDma_ReadReg(AxiDma.RegBase + XAXIDMA_TX_OFFSET,
                                                XAXIDMA_CR_OFFSET);
    smoke_status->dma_mm2s_sr = XAxiDma_ReadReg(AxiDma.RegBase + XAXIDMA_TX_OFFSET,
                                                XAXIDMA_SR_OFFSET);
    smoke_status->dma_s2mm_cr = XAxiDma_ReadReg(AxiDma.RegBase + XAXIDMA_RX_OFFSET,
                                                XAXIDMA_CR_OFFSET);
    smoke_status->dma_s2mm_sr = XAxiDma_ReadReg(AxiDma.RegBase + XAXIDMA_RX_OFFSET,
                                                XAXIDMA_SR_OFFSET);
    smoke_status_flush();
}

static uint64_t perf_now_cycles(void) {
    XTime t;

    XTime_GetTime(&t);
    return (uint64_t)t;
}

static uint32_t cycles_to_us32(uint64_t cycles) {
    uint64_t us = (cycles * 1000000ull) / (uint64_t)COUNTS_PER_SECOND;

    if (us > 0xFFFFFFFFull) {
        return 0xFFFFFFFFu;
    }
    return (uint32_t)us;
}

static uint32_t token_rate_milli(uint32_t tokens, uint64_t cycles) {
    uint64_t rate;

    if (tokens == 0u || cycles == 0u) {
        return 0u;
    }
    rate = ((uint64_t)tokens * (uint64_t)COUNTS_PER_SECOND * 1000ull) /
           cycles;
    return (rate > UINT32_MAX) ? UINT32_MAX : (uint32_t)rate;
}

static void infer_perf_publish(const InferPerf *perf, uint32_t token_count,
                               uint32_t generated_count, uint32_t last_token,
                               uint32_t pseudo_score) {
    smoke_status->hw[0] = token_count;
    smoke_status->hw[1] = generated_count;
    smoke_status->hw[2] = perf->cache_hits;
    smoke_status->hw[3] = perf->cache_misses;
    smoke_status->hw[4] = last_token;
    smoke_status->hw[5] = pseudo_score;
    smoke_status->perf[0] = cycles_to_us32(perf->emmc_cycles);
    smoke_status->perf[1] = cycles_to_us32(perf->accel_cycles);
    smoke_status->perf[2] = cycles_to_us32(perf->total_cycles);
    smoke_status->perf[3] = cycles_to_us32(perf->prefill_cycles);
    smoke_status->perf[4] = cycles_to_us32(perf->decode_cycles);
    smoke_status->perf[5] = cycles_to_us32(perf->lm_head_cycles);
    smoke_status->perf[6] = INFER_CONTEXT_MAX_TOKENS;
    smoke_status->perf[7] = KV_CACHE_TOTAL_BYTES >> 20;
    smoke_status->perf[8] = cycles_to_us32(perf->attention_block_cycles);
    smoke_status->perf[9] = cycles_to_us32(perf->mlp_block_cycles);
    smoke_status->perf[10] = cycles_to_us32(perf->attention_projection_cycles);
    smoke_status->perf[11] = cycles_to_us32(perf->attention_core_cycles);
    smoke_status->perf[12] = cycles_to_us32(perf->attention_nonlinear_cycles);
    smoke_status->perf[13] = cycles_to_us32(perf->mlp_projection_cycles);
    smoke_status->perf[14] = cycles_to_us32(perf->mlp_nonlinear_cycles);
    smoke_status->perf[15] = cycles_to_us32(perf->layer_forward_cycles);
    smoke_status_flush();
}

#if defined(BITNET_LINUX)
static uint64_t plddr_cache_align(uint64_t value) {
    return (value + UINT64_C(0xFFF)) & ~UINT64_C(0xFFF);
}

static void plddr_packet_cache_reset(void) {
    memset(plddr_packet_cache, 0, sizeof(plddr_packet_cache));
    plddr_packet_cache_next = XPAR_PL_DDR4_0_BASEADDR;
    psddr_packet_cache_next = XPAR_PS_DDR_CACHE_BASEADDR;
    split_memory_ready = bitnet_psddr_cache_enabled() ? 1u : 0u;
}

static int plddr_packet_cache_alloc(uint32_t lane_bytes,
                                    uint64_t *lane0_addr,
                                    uint64_t *lane1_addr) {
    uint64_t aligned_bytes = plddr_cache_align(lane_bytes);
    uint64_t plddr_limit =
        (uint64_t)XPAR_PL_DDR4_0_HIGHADDR + UINT64_C(1);

    if (lane_bytes == 0u || lane0_addr == 0 || lane1_addr == 0) {
        return XST_BUFFER_TOO_SMALL;
    }
    if (split_memory_ready != 0u) {
        uint64_t psddr_limit =
            (uint64_t)XPAR_PS_DDR_CACHE_HIGHADDR + UINT64_C(1);

        if (psddr_packet_cache_next > psddr_limit ||
            aligned_bytes > psddr_limit - psddr_packet_cache_next ||
            plddr_packet_cache_next > plddr_limit ||
            aligned_bytes > plddr_limit - plddr_packet_cache_next) {
            return XST_BUFFER_TOO_SMALL;
        }
        *lane0_addr = psddr_packet_cache_next;
        *lane1_addr = plddr_packet_cache_next;
        psddr_packet_cache_next += aligned_bytes;
        plddr_packet_cache_next += aligned_bytes;
    } else {
        uint64_t pair_bytes = aligned_bytes * 2u;

        if (plddr_packet_cache_next > plddr_limit ||
            pair_bytes > plddr_limit - plddr_packet_cache_next) {
            return XST_BUFFER_TOO_SMALL;
        }
        *lane0_addr = plddr_packet_cache_next;
        *lane1_addr = plddr_packet_cache_next + aligned_bytes;
        plddr_packet_cache_next += pair_bytes;
    }
    return XST_SUCCESS;
}

static int packet_cache_copy_to(uint64_t physical, const void *source,
                                size_t length) {
    if (physical >= XPAR_PS_DDR_CACHE_BASEADDR &&
        physical <= XPAR_PS_DDR_CACHE_HIGHADDR) {
        return bitnet_psddr_cache_copy_to(physical, source, length);
    }
    return bitnet_plddr_copy_to(physical, source, length);
}

/*
 * Streaming mode keeps raw weights in two PS-DDR slots while the packet
 * cache is being filled.  Once the batch-1 packets for a layer are resident
 * in PL-DDR, re-reading that layer from eMMC on every decode token is pure
 * overhead: the dual-lane path only needs the activation prefix on a cache
 * hit.  Keep the shape check here so a prefill (batch-16) packet is not
 * mistaken for a decode packet.
 */
static int plddr_decode_cache_entry_ready(uint32_t layer_index,
                                          uint32_t tensor_id,
                                          uint32_t out_features,
                                          uint32_t in_features) {
    const PlddrPacketCacheEntry *entry;
    uint32_t lane_out;
    uint32_t lane_words;

    if (layer_index >= MODEL_TOTAL_LAYERS || tensor_id >= 8u ||
        (out_features & 0x7u) != 0u || out_features < 8u ||
        in_features == 0u) {
        return 0;
    }
    lane_out = out_features / 2u;
    lane_words = bitnet_accel_stream_words(lane_out, in_features);
    entry = &plddr_packet_cache[layer_index][tensor_id];
    return entry->valid != 0u && entry->batch_rows == 1u &&
           entry->out_features == out_features &&
           entry->in_features == in_features &&
           entry->lane_words == lane_words;
}

static int plddr_decode_layer_ready(uint32_t layer_index) {
    return plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_Q,
               LAYER_HIDDEN_SIZE + (2u * LAYER_KV_SIZE), LAYER_HIDDEN_SIZE) &&
           plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_K,
               LAYER_KV_SIZE, LAYER_HIDDEN_SIZE) &&
           plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_V,
               LAYER_KV_SIZE, LAYER_HIDDEN_SIZE) &&
           plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_O,
               LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE) &&
           plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_GATE,
               LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE) &&
           plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_UP,
               LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE) &&
           plddr_decode_cache_entry_ready(
               layer_index, LAYER_TENSOR_DOWN,
               LAYER_HIDDEN_SIZE, LAYER_FFN_SIZE);
}

static int plddr_decode_cache_all_ready(uint32_t first_layer,
                                        uint32_t num_layers) {
    if (num_layers == 0u || first_layer >= MODEL_TOTAL_LAYERS ||
        num_layers > MODEL_TOTAL_LAYERS - first_layer) {
        return 0;
    }
    for (uint32_t i = 0u; i < num_layers; i++) {
        if (!plddr_decode_layer_ready(first_layer + i)) {
            return 0;
        }
    }
    return 1;
}
#endif

static uint8_t enc_weight(int w) {
    if (w < 0) {
        return 0x1u;
    }
    if (w > 0) {
        return 0x3u;
    }
    return 0x2u;
}

static uint8_t pack4(int w0, int w1, int w2, int w3) {
    return (uint8_t)(enc_weight(w0) |
                     (enc_weight(w1) << 2) |
                     (enc_weight(w2) << 4) |
                     (enc_weight(w3) << 6));
}

static void make_test_weights(uint8_t *packed) {
    int weights[TEST_N][TEST_K];

    for (uint32_t n = 0u; n < TEST_N; n++) {
        for (uint32_t k = 0u; k < TEST_K; k++) {
            weights[n][k] = (int)((n + (2u * k)) % 3u) - 1;
        }
    }

    weights[0][0] =  1;
    weights[0][1] = -1;
    weights[1][2] =  1;
    weights[2][3] = -1;
    weights[3][4] =  1;
    weights[4][5] = -1;
    weights[5][6] =  1;
    weights[6][7] = -1;
    weights[7][0] =  0;

    for (uint32_t g = 0u; g < TEST_N / 4u; g++) {
        for (uint32_t k = 0u; k < TEST_K; k++) {
            packed[g * TEST_K + k] = pack4(weights[g * 4u + 0u][k],
                                           weights[g * 4u + 1u][k],
                                           weights[g * 4u + 2u][k],
                                           weights[g * 4u + 3u][k]);
        }
    }
}

static int init_dma(uint32_t fail_stage) {
    XAxiDma_Config *cfg;
    int status;

    cfg = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    if (cfg == 0) {
        xil_printf("XAxiDma_LookupConfig failed\r\n");
        smoke_status_update(SMOKE_STAGE_DMA_LOOKUP, SMOKE_RESULT_FAIL, XST_FAILURE);
        return XST_FAILURE;
    }

    status = XAxiDma_CfgInitialize(&AxiDma, cfg);
    if (status != XST_SUCCESS) {
        xil_printf("XAxiDma_CfgInitialize failed: %d\r\n", status);
        smoke_status_update(fail_stage, SMOKE_RESULT_FAIL, status);
        return status;
    }
    smoke_status_record_dma();

    if (XAxiDma_HasSg(&AxiDma)) {
        xil_printf("DMA is in SG mode; this example expects simple mode\r\n");
        smoke_status_update(fail_stage, SMOKE_RESULT_FAIL, XST_FAILURE);
        return XST_FAILURE;
    }

#if defined(BITNET_LINUX)
    dual_lane_ready = 0u;
    cfg = XAxiDma_LookupConfig(XPAR_AXI_DMA_1_DEVICE_ID);
    if (cfg != 0) {
        uint32_t plddr_status = bitnet_plddr_status();
        uint32_t hardware_split =
            (plddr_status & XPAR_PLDDR_STATUS_SPLIT_MEMORY_MASK) != 0u;
        uint32_t software_split = bitnet_psddr_cache_enabled() != 0;

        status = XAxiDma_CfgInitialize(&AxiDmaLane1, cfg);
        if (status == XST_SUCCESS && !XAxiDma_HasSg(&AxiDmaLane1) &&
            (plddr_status & XPAR_PLDDR_STATUS_CALIBRATED_MASK) != 0u &&
            hardware_split == software_split) {
            dual_lane_ready = 1u;
            xil_printf("Dual lane enabled: %s\r\n",
                       software_split != 0u ?
                           "lane0=PS-DDR/HP1 lane1=PL-DDR" :
                           "lane0=PL-DDR lane1=PL-DDR");
        } else if (hardware_split != software_split) {
            xil_printf("Dual lane topology mismatch: hardware_split=%lu software_split=%lu\r\n",
                       (unsigned long)hardware_split,
                       (unsigned long)software_split);
            smoke_status_update(fail_stage, SMOKE_RESULT_FAIL, XST_FAILURE);
            return XST_FAILURE;
        } else {
            xil_printf("PL-DDR dual lane unavailable; using single DMA path\r\n");
        }
    }
#endif

    return XST_SUCCESS;
}

static int run_smoke_test(void) {
    uint8_t packed_weights[(TEST_N / 4u) * TEST_K] __attribute__((aligned(64)));
    uint32_t tx_words;
    uint32_t mismatches = 0u;
    int status;

    smoke_status_set_dims(TEST_N, TEST_K);
    smoke_status_update(SMOKE_STAGE_BOOT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    xil_printf("BitNet AXU3EGB accelerator smoke test\r\n");

    status = init_dma(SMOKE_STAGE_DMA_INIT);
    if (status != XST_SUCCESS) {
        return status;
    }
#if defined(BITNET_LINUX)
    plddr_packet_cache_reset();
#endif
    smoke_status_update(SMOKE_STAGE_DMA_INIT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    make_test_weights(packed_weights);
    tx_words = bitnet_accel_chunked_stream_words(TEST_N, TEST_K, TEST_CHUNK_OUT);
    smoke_status->tx_words = tx_words;
    smoke_status->reserved0 = TEST_CHUNK_OUT;
    smoke_status->reserved1 = bitnet_accel_chunk_count(TEST_N, TEST_CHUNK_OUT);
    smoke_status_flush();

    smoke_status_update(SMOKE_STAGE_STREAM_BUILD, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    for (uint32_t i = 0u; i < TEST_N; i++) {
        rx_outputs[i] = 0;
    }

    smoke_status_record_dma();
    smoke_status_update(SMOKE_STAGE_DMA_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    status = bitnet_accel_run_gemv_chunked(&AxiDma, tx_stream, TX_WORD_CAP,
                                           activations, packed_weights,
                                           TEST_N, TEST_K, TEST_CHUNK_OUT,
                                           BITNET_ACCEL_MAP_GPU_PLUS2,
                                           rx_outputs);
    smoke_status_record_dma();
    if (status != XST_SUCCESS) {
        xil_printf("bitnet_accel_run_gemv_chunked failed: %d\r\n", status);
        smoke_status_update(SMOKE_STAGE_DMA_RUN, SMOKE_RESULT_FAIL, status);
        return status;
    }

    smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    for (uint32_t n = 0u; n < TEST_N; n++) {
        int32_t ref = bitnet_accel_reference_dot4_layout(
            activations, packed_weights, n, TEST_K, BITNET_ACCEL_MAP_GPU_PLUS2);
        smoke_status->hw[n] = (uint32_t)rx_outputs[n];
        smoke_status->ref[n] = (uint32_t)ref;
        xil_printf("out[%lu] hw=%ld ref=%ld\r\n", (unsigned long)n,
                   (long)rx_outputs[n], (long)ref);
        if (rx_outputs[n] != ref) {
            xil_printf("Mismatch at output %lu\r\n", (unsigned long)n);
            if (mismatches == 0u) {
                smoke_status->first_mismatch = n;
            }
            mismatches++;
        }
        smoke_status->mismatches = mismatches;
        smoke_status_flush();
    }

    if (mismatches != 0u) {
        smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_FAIL, XST_FAILURE);
        return XST_FAILURE;
    }

    for (uint32_t i = 0u; i < (TEST_GEMM_BATCH * TEST_N); i++) {
        gemm_outputs[i] = 0;
    }

    smoke_status_update(SMOKE_STAGE_DMA_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    status = bitnet_accel_run_gemm_chunked_direct(&AxiDma, tx_stream, TX_WORD_CAP,
                                                  gemm_activations, packed_weights,
                                                  TEST_GEMM_BATCH, TEST_N, TEST_K,
                                                  TEST_N, BITNET_ACCEL_MAP_GPU_PLUS2,
                                                  gemm_outputs, 0u);
    smoke_status_record_dma();
    if (status != XST_SUCCESS) {
        xil_printf("bitnet_accel_run_gemm_chunked_direct failed: %d\r\n", status);
        smoke_status_update(SMOKE_STAGE_DMA_RUN, SMOKE_RESULT_FAIL, status);
        return status;
    }

    smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    for (uint32_t group = 0u; group < (TEST_N / 4u); group++) {
        for (uint32_t row = 0u; row < TEST_GEMM_BATCH; row++) {
            for (uint32_t lane = 0u; lane < 4u; lane++) {
                uint32_t out_channel = (group * 4u) + lane;
                uint32_t index = (group * TEST_GEMM_BATCH * 4u) + (row * 4u) + lane;
                int32_t ref = bitnet_accel_reference_gemm_dot4_layout(
                    gemm_activations, packed_weights, row, out_channel, TEST_K,
                    BITNET_ACCEL_MAP_GPU_PLUS2);

                xil_printf("gemm[%lu,%lu] hw=%ld ref=%ld\r\n",
                           (unsigned long)row, (unsigned long)out_channel,
                           (long)gemm_outputs[index], (long)ref);
                if (gemm_outputs[index] != ref) {
                    xil_printf("GEMM mismatch at row %lu output %lu\r\n",
                               (unsigned long)row, (unsigned long)out_channel);
                    if (mismatches == 0u) {
                        smoke_status->first_mismatch = index;
                    }
                    mismatches++;
                }
                smoke_status->mismatches = mismatches;
                smoke_status_flush();
            }
        }
    }

    if (mismatches != 0u) {
        smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_FAIL, XST_FAILURE);
        return XST_FAILURE;
    }

    xil_printf("PASS GEMV+GEMM\r\n");
    smoke_status_update(SMOKE_STAGE_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    return XST_SUCCESS;
}

static int real_tensor_control_loaded(RealTensorControl *dst) {
    const volatile RealTensorControl *ctrl = (const volatile RealTensorControl *)REAL_CONTROL_BASE;

    Xil_DCacheInvalidateRange((UINTPTR)REAL_CONTROL_BASE, sizeof(RealTensorControl));
    if (ctrl->magic != REAL_CONTROL_MAGIC) {
        return 0;
    }

    dst->magic = ctrl->magic;
    dst->out_features = ctrl->out_features;
    dst->in_features = ctrl->in_features;
    dst->flags = ctrl->flags;
    dst->weight_bytes = ctrl->weight_bytes;
    dst->tensor_kind = ctrl->tensor_kind;
    dst->activation_seed = ctrl->activation_seed;
    dst->reserved = ctrl->reserved;
    return 1;
}

static void make_real_activation(int8_t *activations, uint32_t in_features, uint32_t seed) {
    for (uint32_t k = 0u; k < in_features; k++) {
        uint32_t mixed = (k * 29u) + (seed * 17u) + (k >> 3);
        int32_t value = (int32_t)(mixed % 15u) - 7;
        if (value == 0) {
            value = 1;
        }
        activations[k] = (int8_t)value;
    }
}

static int layer_chain_control_loaded(LayerChainControl *dst) {
    const volatile LayerChainControl *ctrl = (const volatile LayerChainControl *)LAYER_CONTROL_BASE;

    Xil_DCacheInvalidateRange((UINTPTR)LAYER_CONTROL_BASE, sizeof(LayerChainControl));
    if (ctrl->magic != LAYER_CONTROL_MAGIC) {
        return 0;
    }

    dst->magic = ctrl->magic;
    dst->first_layer = ctrl->first_layer;
    dst->num_layers = ctrl->num_layers;
    dst->activation_seed = ctrl->activation_seed;
    dst->flags = ctrl->flags;
    dst->reserved0 = ctrl->reserved0;
    dst->reserved1 = ctrl->reserved1;
    dst->reserved2 = ctrl->reserved2;
    return 1;
}

static int emmc_layer_chain_control_loaded(EmmcLayerChainControl *dst) {
    const volatile EmmcLayerChainControl *ctrl = (const volatile EmmcLayerChainControl *)EMMC_CONTROL_BASE;

    Xil_DCacheInvalidateRange((UINTPTR)EMMC_CONTROL_BASE, sizeof(EmmcLayerChainControl));
    if (ctrl->magic != EMMC_CONTROL_MAGIC) {
        return 0;
    }

    dst->magic = ctrl->magic;
    dst->first_layer = ctrl->first_layer;
    dst->num_layers = ctrl->num_layers;
    dst->activation_seed = ctrl->activation_seed;
    dst->flags = ctrl->flags;
    dst->reserved0 = ctrl->reserved0;
    dst->reserved1 = ctrl->reserved1;
    dst->reserved2 = ctrl->reserved2;
    return 1;
}

static int infer_control_loaded(InferControl *dst) {
    const volatile InferControl *ctrl = (const volatile InferControl *)INFER_CONTROL_BASE;

    Xil_DCacheInvalidateRange((UINTPTR)INFER_CONTROL_BASE, sizeof(InferControl));
    if (ctrl->magic != INFER_CONTROL_MAGIC) {
        return 0;
    }

    dst->magic = ctrl->magic;
    dst->version = ctrl->version;
    dst->mode = ctrl->mode;
    dst->prompt_addr = ctrl->prompt_addr;
    dst->prompt_length = ctrl->prompt_length;
    dst->first_layer = ctrl->first_layer;
    dst->num_layers = ctrl->num_layers;
    dst->activation_seed = ctrl->activation_seed;
    dst->flags = ctrl->flags;
    dst->max_new_tokens = ctrl->max_new_tokens;
    dst->temperature_milli = ctrl->temperature_milli;
    dst->top_k = ctrl->top_k;
    dst->prefill_batch_rows = ctrl->prefill_batch_rows;
    dst->reserved1 = ctrl->reserved1;
    dst->reserved2 = ctrl->reserved2;
    dst->reserved3 = ctrl->reserved3;
    return 1;
}

static int8_t compress_i32_to_i8(int32_t value, uint32_t divisor) {
    int32_t scaled;

    if (divisor == 0u) {
        divisor = 1u;
    }
    scaled = value / (int32_t)divisor;
    if ((scaled == 0) && (value != 0)) {
        scaled = (value > 0) ? 1 : -1;
    }
    if (scaled > 127) {
        scaled = 127;
    } else if (scaled < -128) {
        scaled = -128;
    }
    return (int8_t)scaled;
}

static int8_t float_to_i8_scaled(float value, float scale);

static void compress_vector_i32_to_i8(const int32_t *src, int8_t *dst,
                                      uint32_t count, uint32_t divisor) {
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = compress_i32_to_i8(src[i], divisor);
    }
}

static float quantize_i32_dynamic(const int32_t *src, int8_t *dst,
                                  uint32_t count, float src_factor) {
    uint32_t max_abs = 0u;

    for (uint32_t i = 0u; i < count; i++) {
        uint32_t magnitude = (src[i] < 0) ? (uint32_t)(-(int64_t)src[i]) : (uint32_t)src[i];
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    if (max_abs == 0u) {
        for (uint32_t i = 0u; i < count; i++) {
            dst[i] = 0;
        }
        return 1.0f;
    }

    float quant_scale = 127.0f / (float)max_abs;
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = float_to_i8_scaled((float)src[i], quant_scale);
    }
    return src_factor / quant_scale;
}

static float quantize_float_dynamic(const float *src, int8_t *dst, uint32_t count) {
    float max_abs = 0.0f;

    for (uint32_t i = 0u; i < count; i++) {
        float magnitude = (src[i] < 0.0f) ? -src[i] : src[i];
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

    float quant_scale = 127.0f / max_abs;
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = float_to_i8_scaled(src[i], quant_scale);
    }
    return 1.0f / quant_scale;
}

static void residual_add_scaled(int8_t *hidden_act, float *hidden_factor,
                                const int32_t *delta, float delta_factor,
                                uint32_t count) {
    float max_abs = 0.0f;

    for (uint32_t i = 0u; i < count; i++) {
        float value = ((float)hidden_act[i] * *hidden_factor) +
                      ((float)delta[i] * delta_factor);
        float magnitude = (value < 0.0f) ? -value : value;
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    if (max_abs <= 1.0e-20f) {
        for (uint32_t i = 0u; i < count; i++) {
            hidden_act[i] = 0;
        }
        *hidden_factor = 1.0f;
        return;
    }

    float quant_scale = 127.0f / max_abs;
    for (uint32_t i = 0u; i < count; i++) {
        float value = ((float)hidden_act[i] * *hidden_factor) +
                      ((float)delta[i] * delta_factor);
        hidden_act[i] = float_to_i8_scaled(value, quant_scale);
    }
    *hidden_factor = 1.0f / quant_scale;
}

static void make_ffn_activation(const int32_t *gate, const int32_t *up, int8_t *dst) {
    for (uint32_t i = 0u; i < LAYER_FFN_SIZE; i++) {
        int32_t g = gate[i] / 256;
        int32_t u = up[i] / 256;
        int32_t relu2;
        int32_t mixed;

        if (g < 0) {
            g = 0;
        } else if (g > 63) {
            g = 63;
        }
        relu2 = (g * g) / 16;
        mixed = (relu2 * u) / 32;
        dst[i] = compress_i32_to_i8(mixed, 1u);
    }
}

static void kv_cache_clear(void) {
    Xil_DCacheInvalidateRange((UINTPTR)KV_CACHE_BASE, KV_CACHE_TOTAL_BYTES);
    Xil_DCacheInvalidateRange((UINTPTR)KV_SCALE_BASE, KV_SCALE_TOTAL_BYTES);
}

static uintptr_t kv_cache_layer_pos_addr(uint32_t layer_index, uint32_t seq_pos, uint32_t is_value) {
    uintptr_t layer_stride = (uintptr_t)KV_CACHE_LAYER_BYTES;
    uintptr_t pos_stride = (uintptr_t)LAYER_KV_SIZE * 2u;
    uintptr_t value_offset = (is_value != 0u) ? (uintptr_t)LAYER_KV_SIZE : 0u;

    return KV_CACHE_BASE +
           ((uintptr_t)layer_index * layer_stride) +
           ((uintptr_t)seq_pos * pos_stride) +
           value_offset;
}

static uintptr_t kv_cache_scale_addr(uint32_t layer_index, uint32_t seq_pos, uint32_t is_value) {
    return KV_SCALE_BASE +
           ((uintptr_t)layer_index * KV_SCALE_LAYER_BYTES) +
           ((uintptr_t)is_value * INFER_TOTAL_MAX_TOKENS * sizeof(float)) +
           ((uintptr_t)seq_pos * sizeof(float));
}

static uint32_t float_as_u32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float u32_as_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int run_attention_pl(uint32_t layer_index, uint32_t seq_pos,
                            const int32_t *q_out, float q_factor,
                            const int32_t *k_out, float k_factor,
                            const int32_t *v_out, float v_factor,
                            int8_t *att_act, float *att_factor,
                            InferPerf *perf) {
    uint32_t *tx = (uint32_t *)REAL_TX_BASE;
    uint32_t *rx = (uint32_t *)REAL_OUTPUT_BASE;
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint32_t required_words = PL_ATTENTION_HEADER_WORDS +
                              PL_ATTENTION_ROPE_WORDS +
                              PL_ATTENTION_QKV_WORDS +
                              (seq_pos * PL_ATTENTION_CACHE_WORDS_PER_POS);
    uint32_t word_index = 0u;
    uint64_t t0;
    uint64_t t1;
    int status;

    if ((q_out == 0) || (k_out == 0) || (v_out == 0) ||
        (att_act == 0) || (att_factor == 0) || (perf == 0) ||
        (seq_pos >= INFER_TOTAL_MAX_TOKENS) ||
        (required_words > tx_capacity_words) ||
        (rope_available == 0u) || (rope_header == 0) ||
        (rope_q15_table == 0) || (rope_header->q15_scale != 32767u)) {
        return XST_INVALID_PARAM;
    }

    tx[word_index++] = PL_ATTENTION_INPUT_MAGIC;
    tx[word_index++] = seq_pos + 1u;
    tx[word_index++] = float_as_u32(q_factor);
    tx[word_index++] = float_as_u32(k_factor);
    tx[word_index++] = float_as_u32(v_factor);
    tx[word_index++] = rope_header->q15_scale;

    const int16_t *position_table =
        &rope_q15_table[(seq_pos * rope_header->pair_count) * 2u];
    memcpy(&tx[word_index], position_table,
           PL_ATTENTION_ROPE_WORDS * sizeof(uint32_t));
    word_index += PL_ATTENTION_ROPE_WORDS;

    memcpy(&tx[word_index], q_out, LAYER_HIDDEN_SIZE * sizeof(int32_t));
    word_index += LAYER_HIDDEN_SIZE;
    memcpy(&tx[word_index], k_out, LAYER_KV_SIZE * sizeof(int32_t));
    word_index += LAYER_KV_SIZE;
    memcpy(&tx[word_index], v_out, LAYER_KV_SIZE * sizeof(int32_t));
    word_index += LAYER_KV_SIZE;

    for (uint32_t pos = 0u; pos < seq_pos; pos++) {
        const int8_t *past_k =
            (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 0u);
        const int8_t *past_v =
            (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 1u);
        const float *past_k_scale =
            (const float *)kv_cache_scale_addr(layer_index, pos, 0u);
        const float *past_v_scale =
            (const float *)kv_cache_scale_addr(layer_index, pos, 1u);

        tx[word_index++] = float_as_u32(*past_k_scale);
        tx[word_index++] = float_as_u32(*past_v_scale);
        memcpy(&tx[word_index], past_k, LAYER_KV_SIZE);
        word_index += LAYER_KV_SIZE / sizeof(uint32_t);
        memcpy(&tx[word_index], past_v, LAYER_KV_SIZE);
        word_index += LAYER_KV_SIZE / sizeof(uint32_t);
    }

    if (word_index != required_words) {
        return XST_FAILURE;
    }

    t0 = perf_now_cycles();
#if defined(BITNET_LINUX)
    if (bitnet_plddr_status() != 0u) {
        status = bitnet_plddr_copy_to(
            XPAR_PL_DDR4_0_BASEADDR, tx,
            (size_t)word_index * sizeof(uint32_t));
        if (status == XST_SUCCESS) {
            status = bitnet_accel_run_physical(
                &AxiDma, XPAR_PL_DDR4_0_BASEADDR, word_index,
                (int32_t *)rx, PL_ATTENTION_OUTPUT_WORDS);
        }
    } else {
        status = bitnet_accel_run(&AxiDma, tx, word_index, (int32_t *)rx,
                                  PL_ATTENTION_OUTPUT_WORDS);
    }
#else
    status = bitnet_accel_run(&AxiDma, tx, word_index, (int32_t *)rx,
                              PL_ATTENTION_OUTPUT_WORDS);
#endif
    t1 = perf_now_cycles();
    perf->accel_cycles += (t1 - t0);
    perf->attention_core_cycles += (t1 - t0);
    if (status != XST_SUCCESS) {
        return status;
    }
    if ((rx[0] != PL_ATTENTION_OUTPUT_MAGIC) || (rx[1] != 0u)) {
        return XST_FAILURE;
    }

    *att_factor = u32_as_float(rx[2]);
    float *current_k_scale =
        (float *)kv_cache_scale_addr(layer_index, seq_pos, 0u);
    float *current_v_scale =
        (float *)kv_cache_scale_addr(layer_index, seq_pos, 1u);
    int8_t *current_k =
        (int8_t *)kv_cache_layer_pos_addr(layer_index, seq_pos, 0u);
    int8_t *current_v =
        (int8_t *)kv_cache_layer_pos_addr(layer_index, seq_pos, 1u);
    uint32_t output_index = PL_ATTENTION_OUTPUT_HEADER_WORDS;

    *current_k_scale = u32_as_float(rx[3]);
    *current_v_scale = u32_as_float(rx[4]);
    memcpy(att_act, &rx[output_index], LAYER_HIDDEN_SIZE);
    output_index += LAYER_HIDDEN_SIZE / sizeof(uint32_t);
    memcpy(current_k, &rx[output_index], LAYER_KV_SIZE);
    output_index += LAYER_KV_SIZE / sizeof(uint32_t);
    memcpy(current_v, &rx[output_index], LAYER_KV_SIZE);

    Xil_DCacheFlushRange((UINTPTR)att_act, LAYER_HIDDEN_SIZE);
    Xil_DCacheFlushRange((UINTPTR)current_k, LAYER_KV_SIZE * 2u);
    Xil_DCacheFlushRange((UINTPTR)current_k_scale, sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)current_v_scale, sizeof(float));
    return XST_SUCCESS;
}

static int32_t q15_mul_i32(int32_t value, int32_t coeff, uint32_t scale) {
    int64_t prod = ((int64_t)value) * ((int64_t)coeff);

    if (scale == 0u) {
        scale = 1u;
    }
    if (prod >= 0) {
        prod += (int64_t)(scale >> 1);
    } else {
        prod -= (int64_t)(scale >> 1);
    }
    return (int32_t)(prod / (int64_t)scale);
}

static void rope_rotate_vector_i32(int32_t *vec, uint32_t head_count, uint32_t seq_pos) {
    uint32_t pair_count;
    const int16_t *pos_table;

    if ((rope_available == 0u) || (rope_header == 0) || (rope_q15_table == 0) ||
        (vec == 0) || (seq_pos >= rope_header->max_positions) ||
        (rope_header->head_dim != MODEL_HEAD_DIM)) {
        return;
    }

    pair_count = rope_header->pair_count;
    if (pair_count > (MODEL_HEAD_DIM / 2u)) {
        pair_count = MODEL_HEAD_DIM / 2u;
    }
    pos_table = &rope_q15_table[(seq_pos * rope_header->pair_count) * 2u];

    for (uint32_t head = 0u; head < head_count; head++) {
        uint32_t head_base = head * MODEL_HEAD_DIM;

        for (uint32_t pair = 0u; pair < pair_count; pair++) {
            uint32_t lo = head_base + pair;
            uint32_t hi = head_base + pair + pair_count;
            int32_t x0 = vec[lo];
            int32_t x1 = vec[hi];
            int32_t cos_q15 = (int32_t)pos_table[(pair * 2u) + 0u];
            int32_t sin_q15 = (int32_t)pos_table[(pair * 2u) + 1u];

            vec[lo] = q15_mul_i32(x0, cos_q15, rope_header->q15_scale) -
                      q15_mul_i32(x1, sin_q15, rope_header->q15_scale);
            vec[hi] = q15_mul_i32(x1, cos_q15, rope_header->q15_scale) +
                      q15_mul_i32(x0, sin_q15, rope_header->q15_scale);
        }
    }
}

static void __attribute__((unused)) apply_rope_i32(
    int32_t *q_out, int32_t *k_out, uint32_t seq_pos) {
    rope_rotate_vector_i32(q_out, MODEL_NUM_Q_HEADS, seq_pos);
    rope_rotate_vector_i32(k_out, MODEL_NUM_KV_HEADS, seq_pos);
}

static int32_t attention_exp_q15(int32_t shifted_score) {
    uint32_t decay;

    if (shifted_score >= 0) {
        return 32767;
    }
    if (shifted_score <= -((int32_t)(ATTENTION_EXP_STEP * ATTENTION_EXP_MAX_DECAY))) {
        return 1;
    }

    decay = (uint32_t)((-shifted_score + (ATTENTION_EXP_STEP - 1)) / ATTENTION_EXP_STEP);
    if (decay >= ATTENTION_EXP_MAX_DECAY) {
        return 1;
    }
    return (int32_t)(32767u >> decay);
}

static void __attribute__((unused)) build_attention_from_kv_cache(
    uint32_t layer_index, uint32_t seq_pos,
    const int32_t *q_out, const int32_t *k_out,
    const int32_t *v_out, int8_t *att_act) {
    int8_t *q8 = (int8_t *)LAYER_Q8_BASE;
    int8_t *k8 = (int8_t *)LAYER_K8_BASE;
    int8_t *v8 = (int8_t *)LAYER_V8_BASE;
    int8_t *k_cache = (int8_t *)kv_cache_layer_pos_addr(layer_index, seq_pos, 0u);
    int8_t *v_cache = (int8_t *)kv_cache_layer_pos_addr(layer_index, seq_pos, 1u);
    int32_t *scores = (int32_t *)ATTN_SCORE_BASE;
    int32_t *weights = (int32_t *)ATTN_WEIGHT_BASE;
    uint32_t context_len = seq_pos + 1u;

    if (context_len > INFER_TOTAL_MAX_TOKENS) {
        context_len = INFER_TOTAL_MAX_TOKENS;
    }

    compress_vector_i32_to_i8(q_out, q8, LAYER_HIDDEN_SIZE, 256u);
    compress_vector_i32_to_i8(k_out, k8, LAYER_KV_SIZE, 256u);
    compress_vector_i32_to_i8(v_out, v8, LAYER_KV_SIZE, 256u);

    for (uint32_t i = 0u; i < LAYER_KV_SIZE; i++) {
        k_cache[i] = k8[i];
        v_cache[i] = v8[i];
    }
    Xil_DCacheFlushRange((UINTPTR)k_cache, LAYER_KV_SIZE * 2u);

    for (uint32_t pos = 0u; pos < context_len; pos++) {
        const int8_t *past_k = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 0u);
        const int8_t *past_v = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 1u);
        Xil_DCacheInvalidateRange((UINTPTR)past_k, LAYER_KV_SIZE);
        Xil_DCacheInvalidateRange((UINTPTR)past_v, LAYER_KV_SIZE);
    }

    for (uint32_t q_head = 0u; q_head < MODEL_NUM_Q_HEADS; q_head++) {
        uint32_t q_base = q_head * MODEL_HEAD_DIM;
        uint32_t kv_head = q_head / MODEL_KV_GROUP_SIZE;
        uint32_t kv_base = kv_head * MODEL_HEAD_DIM;
        int32_t max_score = (int32_t)0x80000000;
        int32_t weight_sum = 0;

        for (uint32_t pos = 0u; pos < context_len; pos++) {
            const int8_t *past_k = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 0u);
            int32_t score = 0;

            for (uint32_t i = 0u; i < MODEL_HEAD_DIM; i++) {
                score += ((int32_t)q8[q_base + i]) * ((int32_t)past_k[kv_base + i]);
            }
            score /= ATTENTION_SCORE_DIVISOR;
            scores[pos] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        for (uint32_t pos = 0u; pos < context_len; pos++) {
            int32_t w = attention_exp_q15(scores[pos] - max_score);
            weights[pos] = w;
            weight_sum += w;
        }
        if (weight_sum == 0) {
            weight_sum = 1;
        }

        for (uint32_t i = 0u; i < MODEL_HEAD_DIM; i++) {
            int64_t acc = 0;

            for (uint32_t pos = 0u; pos < context_len; pos++) {
                const int8_t *past_v = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 1u);
                acc += ((int64_t)weights[pos]) * ((int64_t)past_v[kv_base + i]);
            }
            att_act[q_base + i] = compress_i32_to_i8((int32_t)(acc / weight_sum), 1u);
        }
    }
}

static float __attribute__((unused)) build_attention_scaled(
    uint32_t layer_index, uint32_t seq_pos,
    const int32_t *q_out, float q_factor,
    const int32_t *k_out, float k_factor,
    const int32_t *v_out, float v_factor,
    int8_t *att_act) {
    int8_t *q8 = (int8_t *)LAYER_Q8_BASE;
    int8_t *k8 = (int8_t *)LAYER_K8_BASE;
    int8_t *v8 = (int8_t *)LAYER_V8_BASE;
    int8_t *k_cache = (int8_t *)kv_cache_layer_pos_addr(layer_index, seq_pos, 0u);
    int8_t *v_cache = (int8_t *)kv_cache_layer_pos_addr(layer_index, seq_pos, 1u);
    float *k_scale = (float *)kv_cache_scale_addr(layer_index, seq_pos, 0u);
    float *v_scale = (float *)kv_cache_scale_addr(layer_index, seq_pos, 1u);
    float *scores = (float *)ATTN_SCORE_BASE;
    float *weights = (float *)ATTN_WEIGHT_BASE;
    float *att_float = (float *)LAYER_ATT_FLOAT_BASE;
    uint32_t context_len = seq_pos + 1u;

    if (context_len > INFER_TOTAL_MAX_TOKENS) {
        context_len = INFER_TOTAL_MAX_TOKENS;
    }

    q_factor = quantize_i32_dynamic(q_out, q8, LAYER_HIDDEN_SIZE, q_factor);
    *k_scale = quantize_i32_dynamic(k_out, k8, LAYER_KV_SIZE, k_factor);
    *v_scale = quantize_i32_dynamic(v_out, v8, LAYER_KV_SIZE, v_factor);

    for (uint32_t i = 0u; i < LAYER_KV_SIZE; i++) {
        k_cache[i] = k8[i];
        v_cache[i] = v8[i];
    }
    Xil_DCacheFlushRange((UINTPTR)k_cache, LAYER_KV_SIZE * 2u);
    Xil_DCacheFlushRange((UINTPTR)k_scale, sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)v_scale, sizeof(float));

    for (uint32_t pos = 0u; pos < context_len; pos++) {
        const int8_t *past_k = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 0u);
        const int8_t *past_v = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 1u);
        const float *past_k_scale = (const float *)kv_cache_scale_addr(layer_index, pos, 0u);
        const float *past_v_scale = (const float *)kv_cache_scale_addr(layer_index, pos, 1u);
        Xil_DCacheInvalidateRange((UINTPTR)past_k, LAYER_KV_SIZE);
        Xil_DCacheInvalidateRange((UINTPTR)past_v, LAYER_KV_SIZE);
        Xil_DCacheInvalidateRange((UINTPTR)past_k_scale, sizeof(float));
        Xil_DCacheInvalidateRange((UINTPTR)past_v_scale, sizeof(float));
    }

    for (uint32_t q_head = 0u; q_head < MODEL_NUM_Q_HEADS; q_head++) {
        uint32_t q_base = q_head * MODEL_HEAD_DIM;
        uint32_t kv_head = q_head / MODEL_KV_GROUP_SIZE;
        uint32_t kv_base = kv_head * MODEL_HEAD_DIM;
        float max_score = -3.402823466e+38f;
        float weight_sum = 0.0f;

        for (uint32_t pos = 0u; pos < context_len; pos++) {
            const int8_t *past_k = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 0u);
            const float *past_k_scale = (const float *)kv_cache_scale_addr(layer_index, pos, 0u);
            int32_t dot = 0;

            for (uint32_t i = 0u; i < MODEL_HEAD_DIM; i++) {
                dot += ((int32_t)q8[q_base + i]) * ((int32_t)past_k[kv_base + i]);
            }
            scores[pos] = ((float)dot) * q_factor * *past_k_scale /
                          sqrtf((float)MODEL_HEAD_DIM);
            if (scores[pos] > max_score) {
                max_score = scores[pos];
            }
        }

        for (uint32_t pos = 0u; pos < context_len; pos++) {
            weights[pos] = expf(scores[pos] - max_score);
            weight_sum += weights[pos];
        }
        if (weight_sum <= 0.0f) {
            weight_sum = 1.0f;
        }

        for (uint32_t i = 0u; i < MODEL_HEAD_DIM; i++) {
            float acc = 0.0f;

            for (uint32_t pos = 0u; pos < context_len; pos++) {
                const int8_t *past_v = (const int8_t *)kv_cache_layer_pos_addr(layer_index, pos, 1u);
                const float *past_v_scale = (const float *)kv_cache_scale_addr(layer_index, pos, 1u);
                acc += weights[pos] * ((float)past_v[kv_base + i]) * *past_v_scale;
            }
            att_float[q_base + i] = acc / weight_sum;
        }
    }

    return quantize_float_dynamic(att_float, att_act, LAYER_HIDDEN_SIZE);
}

static int run_attention_dispatch(uint32_t layer_index, uint32_t seq_pos,
                                  const int32_t *q_out, float q_factor,
                                  const int32_t *k_out, float k_factor,
                                  const int32_t *v_out, float v_factor,
                                  int8_t *att_act, float *att_factor,
                                  InferPerf *perf) {
#if defined(BITNET_LINUX)
    if (dual_lane_ready != 0u) {
        uint64_t t0 = perf_now_cycles();
        uint64_t t1;

        /* The PLDDR dual wrapper contains GEMV lanes only.  Preserve the
         * transformer path by running RoPE/GQA/softmax on the A53 while the
         * projection GEMVs remain on the two PL lanes. */
        apply_rope_i32((int32_t *)q_out, (int32_t *)k_out, seq_pos);
        *att_factor = build_attention_scaled(
            layer_index, seq_pos, q_out, q_factor, k_out, k_factor,
            v_out, v_factor, att_act);
        t1 = perf_now_cycles();
        perf->attention_core_cycles += t1 - t0;
        return XST_SUCCESS;
    }
#endif
    return run_attention_pl(layer_index, seq_pos, q_out, q_factor,
                            k_out, k_factor, v_out, v_factor,
                            att_act, att_factor, perf);
}

static uint32_t layer_tensor_offset(uint32_t tensor_id) {
    switch (tensor_id) {
    case LAYER_TENSOR_Q:
        return LAYER_WEIGHT_Q_OFFSET;
    case LAYER_TENSOR_K:
        return LAYER_WEIGHT_K_OFFSET;
    case LAYER_TENSOR_V:
        return LAYER_WEIGHT_V_OFFSET;
    case LAYER_TENSOR_O:
        return LAYER_WEIGHT_O_OFFSET;
    case LAYER_TENSOR_GATE:
        return LAYER_WEIGHT_GATE_OFFSET;
    case LAYER_TENSOR_UP:
        return LAYER_WEIGHT_UP_OFFSET;
    case LAYER_TENSOR_DOWN:
        return LAYER_WEIGHT_DOWN_OFFSET;
    default:
        return 0u;
    }
}

static uint32_t layer_tensor_from_slot_offset(uint32_t tensor_offset) {
    switch (tensor_offset) {
    case LAYER_WEIGHT_Q_OFFSET:
        return LAYER_TENSOR_Q;
    case LAYER_WEIGHT_K_OFFSET:
        return LAYER_TENSOR_K;
    case LAYER_WEIGHT_V_OFFSET:
        return LAYER_TENSOR_V;
    case LAYER_WEIGHT_O_OFFSET:
        return LAYER_TENSOR_O;
    case LAYER_WEIGHT_GATE_OFFSET:
        return LAYER_TENSOR_GATE;
    case LAYER_WEIGHT_UP_OFFSET:
        return LAYER_TENSOR_UP;
    case LAYER_WEIGHT_DOWN_OFFSET:
        return LAYER_TENSOR_DOWN;
    default:
        return 0u;
    }
}

static uint32_t layer_tight_tensor_offset(uint32_t tensor_id) {
    switch (tensor_id) {
    case LAYER_TENSOR_Q:
        return LAYER_TIGHT_Q_OFFSET;
    case LAYER_TENSOR_K:
        return LAYER_TIGHT_K_OFFSET;
    case LAYER_TENSOR_V:
        return LAYER_TIGHT_V_OFFSET;
    case LAYER_TENSOR_O:
        return LAYER_TIGHT_O_OFFSET;
    case LAYER_TENSOR_GATE:
        return LAYER_TIGHT_GATE_OFFSET;
    case LAYER_TENSOR_UP:
        return LAYER_TIGHT_UP_OFFSET;
    case LAYER_TENSOR_DOWN:
        return LAYER_TIGHT_DOWN_OFFSET;
    default:
        return 0u;
    }
}

static uintptr_t layer_resident_tensor_addr(uint32_t layer_index, uint32_t tensor_id) {
    return MODEL_U8_RESIDENT_BASE +
           ((uintptr_t)layer_index * (uintptr_t)LAYER_TIGHT_BYTES) +
           (uintptr_t)layer_tight_tensor_offset(tensor_id);
}

static const uint8_t *layer_weight_ptr(uint32_t weight_ref, uint32_t tensor_offset) {
    uintptr_t addr;

    if ((weight_ref & LAYER_WEIGHT_REF_DOUBLEBUF) != 0u) {
        uint32_t slot = weight_ref & LAYER_WEIGHT_REF_DOUBLEBUF_MASK;
        uint32_t tensor_id = layer_tensor_from_slot_offset(tensor_offset);

        addr = LAYER_DOUBLEBUF_BASE +
               ((uintptr_t)slot * LAYER_DOUBLEBUF_STRIDE) +
               (uintptr_t)layer_tight_tensor_offset(tensor_id);
    } else if ((weight_ref & LAYER_WEIGHT_REF_RESIDENT) != 0u) {
        uint32_t layer_index = weight_ref & LAYER_WEIGHT_REF_MASK;
        uint32_t tensor_id = layer_tensor_from_slot_offset(tensor_offset);

        addr = layer_resident_tensor_addr(layer_index, tensor_id);
    } else {
        addr = LAYER_WEIGHT_BASE + ((uintptr_t)weight_ref * LAYER_WEIGHT_STRIDE) + tensor_offset;
    }

    return (const uint8_t *)addr;
}

static uint32_t layer_tensor_weight_bytes(uint32_t tensor_id) {
    switch (tensor_id) {
    case LAYER_TENSOR_Q:
    case LAYER_TENSOR_O:
        return LAYER_Q_WEIGHT_BYTES;
    case LAYER_TENSOR_K:
        return LAYER_K_WEIGHT_BYTES;
    case LAYER_TENSOR_V:
        return LAYER_V_WEIGHT_BYTES;
    case LAYER_TENSOR_GATE:
        return LAYER_GATE_WEIGHT_BYTES;
    case LAYER_TENSOR_UP:
        return LAYER_UP_WEIGHT_BYTES;
    case LAYER_TENSOR_DOWN:
        return LAYER_DOWN_WEIGHT_BYTES;
    default:
        return 0u;
    }
}

static const char *layer_tensor_file_name(uint32_t tensor_id) {
    switch (tensor_id) {
    case LAYER_TENSOR_Q:
        return "Q.BIN";
    case LAYER_TENSOR_K:
        return "K.BIN";
    case LAYER_TENSOR_V:
        return "V.BIN";
    case LAYER_TENSOR_O:
        return "O.BIN";
    case LAYER_TENSOR_GATE:
        return "GATE.BIN";
    case LAYER_TENSOR_UP:
        return "UP.BIN";
    case LAYER_TENSOR_DOWN:
        return "DOWN.BIN";
    default:
        return "BAD.BIN";
    }
}

static uint32_t layer_chain_total_words(uint32_t num_layers) {
    uint32_t per_layer = 0u;

    per_layer += bitnet_accel_chunked_stream_words(LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE, REAL_CHUNK_OUT);
    per_layer += bitnet_accel_chunked_stream_words(LAYER_KV_SIZE, LAYER_HIDDEN_SIZE, REAL_CHUNK_OUT);
    per_layer += bitnet_accel_chunked_stream_words(LAYER_KV_SIZE, LAYER_HIDDEN_SIZE, REAL_CHUNK_OUT);
    per_layer += bitnet_accel_chunked_stream_words(LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE, REAL_CHUNK_OUT);
    per_layer += bitnet_accel_chunked_stream_words(LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE, REAL_CHUNK_OUT);
    per_layer += bitnet_accel_chunked_stream_words(LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE, REAL_CHUNK_OUT);
    per_layer += bitnet_accel_chunked_stream_words(LAYER_HIDDEN_SIZE, LAYER_FFN_SIZE, REAL_CHUNK_OUT);

    return per_layer * num_layers;
}

static int emmc_mount(void) {
    FRESULT fr;

    smoke_status_update(SMOKE_STAGE_EMMC_MOUNT, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    fr = f_mount(&emmc_fatfs, "0:/", 1);
    if (fr != FR_OK) {
        xil_printf("eMMC mount failed: %d\r\n", (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_MOUNT, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static int emmc_install_control_loaded(EmmcInstallControl *dst) {
    const volatile EmmcInstallControl *ctrl = (const volatile EmmcInstallControl *)INSTALL_CONTROL_BASE;
    uint8_t *dst_bytes = (uint8_t *)dst;
    const volatile uint8_t *src_bytes = (const volatile uint8_t *)ctrl;

    Xil_DCacheInvalidateRange((UINTPTR)INSTALL_CONTROL_BASE, sizeof(EmmcInstallControl));
    if (ctrl->magic != INSTALL_CONTROL_MAGIC) {
        return 0;
    }

    for (uint32_t i = 0u; i < sizeof(EmmcInstallControl); i++) {
        dst_bytes[i] = src_bytes[i];
    }
    dst->path[INSTALL_PATH_MAX - 1u] = '\0';
    return 1;
}

static int emmc_install_mount(void) {
    FRESULT fr;

    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_MOUNT, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    fr = f_mount(&emmc_fatfs, "0:/", 1);
    if (fr != FR_OK) {
        xil_printf("eMMC install mount failed: %d\r\n", (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_MOUNT, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static int emmc_install_format_fat(const EmmcInstallControl *ctrl) {
    FRESULT fr;

    smoke_status_set_dims(0u, 0u);
    smoke_status->reserved0 = 0u;
    smoke_status->reserved1 = 0u;
    smoke_status_flush();
    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DETECT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    if ((ctrl->version != INSTALL_CONTROL_VERSION) ||
        (ctrl->op != INSTALL_OP_FORMAT_FAT) ||
        (ctrl->checksum != INSTALL_FORMAT_KEY)) {
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DETECT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    Xil_DCacheFlushRange((UINTPTR)emmc_mkfs_work, sizeof(emmc_mkfs_work));
    fr = f_mkfs("0:/", FM_FAT32, 0u, emmc_mkfs_work, sizeof(emmc_mkfs_work));
    if (fr != FR_OK) {
        xil_printf("eMMC mkfs failed: %d\r\n", (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    fr = f_mount(&emmc_fatfs, "0:/", 1);
    if (fr != FR_OK) {
        xil_printf("eMMC post-mkfs mount failed: %d\r\n", (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_MOUNT, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    xil_printf("Formatted eMMC FAT volume at 0:/\r\n");
    return XST_SUCCESS;
}

static int emmc_install_path_valid(const char *path) {
    uint32_t len = 0u;

    if ((path[0] != '0') || (path[1] != ':') || (path[2] != '/')) {
        return 0;
    }
    if ((path[3] != 'B') || (path[4] != 'I') || (path[5] != 'T') ||
        (path[6] != 'N') || (path[7] != 'E') || (path[8] != 'T') ||
        (path[9] != '/')) {
        return 0;
    }

    while ((len < INSTALL_PATH_MAX) && (path[len] != '\0')) {
        char c = path[len];

        if (((c >= 'A') && (c <= 'Z')) ||
            ((c >= 'a') && (c <= 'z')) ||
            ((c >= '0') && (c <= '9')) ||
            (c == ':') || (c == '/') || (c == '.') ||
            (c == '_') || (c == '-')) {
            len++;
            continue;
        }
        return 0;
    }

    return (len > 10u) && (len < INSTALL_PATH_MAX) && (path[len] == '\0');
}

static uint32_t emmc_install_checksum(uintptr_t data_addr, uint32_t file_size) {
    const volatile uint8_t *data = (const volatile uint8_t *)data_addr;
    uint32_t sum = 0u;

    for (uint32_t i = 0u; i < file_size; i++) {
        sum += data[i];
    }
    return sum;
}

static int emmc_install_ensure_parent_dirs(const char *path) {
    char dir[INSTALL_PATH_MAX];
    uint32_t len = 0u;

    while ((len < (INSTALL_PATH_MAX - 1u)) && (path[len] != '\0')) {
        dir[len] = path[len];
        len++;
    }
    dir[len] = '\0';

    for (uint32_t i = 3u; i < len; i++) {
        FRESULT fr;

        if (dir[i] != '/') {
            continue;
        }

        dir[i] = '\0';
        fr = f_mkdir(dir);
        dir[i] = '/';
        if ((fr != FR_OK) && (fr != FR_EXIST)) {
            xil_printf("mkdir failed: %s fr=%d\r\n", dir, (int)fr);
            smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_FAIL, (int)fr);
            return XST_FAILURE;
        }
    }

    return XST_SUCCESS;
}

static int emmc_install_write_file(const EmmcInstallControl *ctrl) {
    FIL file;
    FRESULT fr;
    UINT bytes_written;
    uintptr_t data_addr = (uintptr_t)ctrl->data_addr;
    uint32_t remaining = ctrl->file_size;
    uint32_t offset = 0u;
    const uint32_t chunk_bytes = 1024u * 1024u;

    if (emmc_install_ensure_parent_dirs(ctrl->path) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    fr = f_open(&file, ctrl->path, FA_CREATE_ALWAYS | FA_WRITE);
    if (fr != FR_OK) {
        xil_printf("Create failed: %s fr=%d\r\n", ctrl->path, (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    while (remaining != 0u) {
        uint32_t this_write = (remaining > chunk_bytes) ? chunk_bytes : remaining;
        uintptr_t src = data_addr + offset;

        Xil_DCacheInvalidateRange((UINTPTR)src, this_write);
        fr = f_write(&file, (const void *)src, this_write, &bytes_written);
        if ((fr != FR_OK) || (bytes_written != this_write)) {
            xil_printf("Write failed: %s fr=%d wrote=%lu expected=%lu\r\n", ctrl->path,
                       (int)fr, (unsigned long)bytes_written, (unsigned long)this_write);
            f_close(&file);
            smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_FAIL,
                                (fr == FR_OK) ? XST_FAILURE : (int)fr);
            return XST_FAILURE;
        }

        offset += this_write;
        remaining -= this_write;
        smoke_status->tx_words = offset;
        smoke_status_flush();
    }

    fr = f_close(&file);
    if (fr != FR_OK) {
        xil_printf("Close failed: %s fr=%d\r\n", ctrl->path, (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static int run_emmc_install(const EmmcInstallControl *ctrl) {
    uintptr_t data_addr = (uintptr_t)ctrl->data_addr;
    uintptr_t data_end = data_addr + (uintptr_t)ctrl->file_size;
    uint32_t actual_checksum;
    int status;

    if (ctrl->op == INSTALL_OP_FORMAT_FAT) {
        return emmc_install_format_fat(ctrl);
    }

    smoke_status_set_dims(ctrl->file_size, ctrl->file_count);
    smoke_status->reserved0 = ctrl->file_index;
    smoke_status->reserved1 = ctrl->file_count;
    smoke_status_flush();
    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DETECT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    if ((ctrl->version != INSTALL_CONTROL_VERSION) ||
        (ctrl->op != INSTALL_OP_WRITE_FILE) ||
        (ctrl->file_size == 0u) ||
        (data_addr < INSTALL_DATA_BASE) ||
        (data_end > INSTALL_DATA_LIMIT) ||
        (data_end < data_addr) ||
        !emmc_install_path_valid(ctrl->path)) {
        smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DETECT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    Xil_DCacheInvalidateRange((UINTPTR)data_addr, ctrl->file_size);
    if (ctrl->checksum != 0u) {
        actual_checksum = emmc_install_checksum(data_addr, ctrl->file_size);
        smoke_status->mismatches = actual_checksum;
        smoke_status->first_mismatch = ctrl->checksum;
        smoke_status_flush();
        if (actual_checksum != ctrl->checksum) {
            smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DETECT, SMOKE_RESULT_FAIL, XST_FAILURE);
            return XST_FAILURE;
        }
    }

    status = emmc_install_mount();
    if (status != XST_SUCCESS) {
        return status;
    }

    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_WRITE, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    status = emmc_install_write_file(ctrl);
    if (status != XST_SUCCESS) {
        return status;
    }

    smoke_status_update(SMOKE_STAGE_EMMC_INSTALL_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    xil_printf("Installed %s (%lu bytes)\r\n", ctrl->path, (unsigned long)ctrl->file_size);
    return XST_SUCCESS;
}

static int emmc_read_exact(const char *path, uintptr_t dst_addr, uint32_t expected_bytes) {
    FIL file;
    FRESULT fr;
    UINT bytes_read = 0u;
    uint32_t remaining = expected_bytes;
    uint32_t offset = 0u;
    const uint32_t chunk_bytes = 1024u * 1024u;

    fr = f_open(&file, path, FA_READ);
    if (fr != FR_OK) {
        xil_printf("Open failed: %s fr=%d\r\n", path, (int)fr);
        smoke_status_update(SMOKE_STAGE_EMMC_LOAD, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    if ((uint32_t)f_size(&file) != expected_bytes) {
        xil_printf("Size mismatch: %s got=%lu expected=%lu\r\n", path,
                   (unsigned long)f_size(&file), (unsigned long)expected_bytes);
        f_close(&file);
        smoke_status_update(SMOKE_STAGE_EMMC_LOAD, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    while (remaining != 0u) {
        uint32_t this_read = (remaining > chunk_bytes) ? chunk_bytes : remaining;
        uintptr_t dst = dst_addr + offset;

        Xil_DCacheInvalidateRange((UINTPTR)dst, this_read);
        fr = f_read(&file, (void *)dst, this_read, &bytes_read);
        if ((fr != FR_OK) || (bytes_read != this_read)) {
            xil_printf("Read failed: %s fr=%d read=%lu expected=%lu\r\n", path,
                       (int)fr, (unsigned long)bytes_read, (unsigned long)this_read);
            f_close(&file);
            smoke_status_update(SMOKE_STAGE_EMMC_LOAD, SMOKE_RESULT_FAIL,
                                (fr == FR_OK) ? XST_FAILURE : (int)fr);
            return XST_FAILURE;
        }
        Xil_DCacheFlushRange((UINTPTR)dst, this_read);

        offset += this_read;
        remaining -= this_read;
    }

    f_close(&file);
    return XST_SUCCESS;
}

static int tokenizer_load_from_emmc(void) {
    FIL file;
    FRESULT fr;
    UINT bytes_read;
    uint32_t remaining;
    uint32_t offset = 0u;
    uint32_t total_bytes;
    const uint32_t chunk_bytes = 1024u * 1024u;
    const TokenizerAssetHeader *header = (const TokenizerAssetHeader *)TOKENIZER_ASSET_BASE;

    if (tokenizer_header != 0) {
        return XST_SUCCESS;
    }

    fr = f_open(&file, TOKENIZER_PATH, FA_READ);
    if (fr != FR_OK) {
        xil_printf("Open tokenizer failed: %s fr=%d\r\n", TOKENIZER_PATH, (int)fr);
        smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, (int)fr);
        return XST_FAILURE;
    }

    if ((uint64_t)f_size(&file) > (uint64_t)TOKENIZER_ASSET_MAX_BYTES) {
        xil_printf("Tokenizer asset too large: %lu\r\n", (unsigned long)f_size(&file));
        f_close(&file);
        smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_BUFFER_TOO_SMALL);
        return XST_BUFFER_TOO_SMALL;
    }

    total_bytes = (uint32_t)f_size(&file);
    remaining = total_bytes;
    while (remaining != 0u) {
        uint32_t this_read = (remaining > chunk_bytes) ? chunk_bytes : remaining;
        uintptr_t dst = TOKENIZER_ASSET_BASE + offset;

        Xil_DCacheInvalidateRange((UINTPTR)dst, this_read);
        fr = f_read(&file, (void *)dst, this_read, &bytes_read);
        if ((fr != FR_OK) || (bytes_read != this_read)) {
            xil_printf("Tokenizer read failed: fr=%d read=%lu expected=%lu\r\n",
                       (int)fr, (unsigned long)bytes_read, (unsigned long)this_read);
            f_close(&file);
            smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL,
                                (fr == FR_OK) ? XST_FAILURE : (int)fr);
            return XST_FAILURE;
        }
        Xil_DCacheFlushRange((UINTPTR)dst, this_read);
        offset += this_read;
        remaining -= this_read;
    }
    f_close(&file);

    Xil_DCacheInvalidateRange((UINTPTR)TOKENIZER_ASSET_BASE, total_bytes);
    if ((header->magic != TOKENIZER_ASSET_MAGIC) ||
        (header->version != TOKENIZER_ASSET_VERSION) ||
        (header->total_bytes != total_bytes) ||
        (header->node_count == 0u) ||
        (header->nodes_offset >= total_bytes) ||
        (header->edges_offset >= total_bytes)) {
        xil_printf("Tokenizer header invalid\r\n");
        smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    if ((header->nodes_offset + (header->node_count * sizeof(TokenizerNode)) > total_bytes) ||
        (header->edges_offset + (header->edge_count * sizeof(TokenizerEdge)) > total_bytes)) {
        xil_printf("Tokenizer table range invalid\r\n");
        smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    tokenizer_header = header;
    tokenizer_nodes = (const TokenizerNode *)(TOKENIZER_ASSET_BASE + header->nodes_offset);
    tokenizer_edges = (const TokenizerEdge *)(TOKENIZER_ASSET_BASE + header->edges_offset);
    xil_printf("Tokenizer loaded: nodes=%lu edges=%lu max_token_bytes=%lu\r\n",
               (unsigned long)header->node_count,
               (unsigned long)header->edge_count,
               (unsigned long)header->max_token_bytes);
    return XST_SUCCESS;
}

static int tokenizer_decode_load_from_emmc(void) {
    FIL file;
    FRESULT fr;
    UINT bytes_read;
    uint32_t total_bytes;
    const TokenizerDecodeHeader *header = (const TokenizerDecodeHeader *)TOKENIZER_DECODE_BASE;

    if (tokenizer_decode_header != 0) {
        return XST_SUCCESS;
    }

    fr = f_open(&file, TOKENIZER_DECODE_PATH, FA_READ);
    if (fr != FR_OK) {
        xil_printf("Open tokenizer decode failed: %s fr=%d\r\n", TOKENIZER_DECODE_PATH, (int)fr);
        return XST_FAILURE;
    }

    if ((uint64_t)f_size(&file) > (uint64_t)TOKENIZER_DECODE_MAX_BYTES) {
        xil_printf("Tokenizer decode asset too large: %lu\r\n", (unsigned long)f_size(&file));
        f_close(&file);
        return XST_BUFFER_TOO_SMALL;
    }

    total_bytes = (uint32_t)f_size(&file);
    Xil_DCacheInvalidateRange((UINTPTR)TOKENIZER_DECODE_BASE, total_bytes);
    fr = f_read(&file, (void *)TOKENIZER_DECODE_BASE, total_bytes, &bytes_read);
    f_close(&file);
    if ((fr != FR_OK) || (bytes_read != total_bytes)) {
        xil_printf("Tokenizer decode read failed: fr=%d read=%lu expected=%lu\r\n",
                   (int)fr, (unsigned long)bytes_read, (unsigned long)total_bytes);
        return XST_FAILURE;
    }
    Xil_DCacheFlushRange((UINTPTR)TOKENIZER_DECODE_BASE, total_bytes);
    Xil_DCacheInvalidateRange((UINTPTR)TOKENIZER_DECODE_BASE, total_bytes);

    if ((header->magic != TOKENIZER_DECODE_MAGIC) ||
        (header->version != TOKENIZER_DECODE_VERSION) ||
        (header->total_bytes != total_bytes) ||
        (header->vocab_size != MODEL_VOCAB_SIZE) ||
        (header->entries_offset >= total_bytes) ||
        (header->payload_offset >= total_bytes) ||
        (header->entries_offset + (MODEL_VOCAB_SIZE * sizeof(TokenizerDecodeEntry)) >
         total_bytes)) {
        xil_printf("Tokenizer decode header invalid\r\n");
        return XST_INVALID_PARAM;
    }

    tokenizer_decode_header = header;
    tokenizer_decode_entries = (const TokenizerDecodeEntry *)(TOKENIZER_DECODE_BASE +
                                                              header->entries_offset);
    xil_printf("Tokenizer decode loaded: vocab=%lu bytes=%lu\r\n",
               (unsigned long)header->vocab_size, (unsigned long)total_bytes);
    return XST_SUCCESS;
}

static int tokenizer_pretok_load_from_emmc(void) {
    FIL file;
    FRESULT fr;
    UINT bytes_read;
    uint32_t total_bytes;
    const TokenizerPretokHeader *header = (const TokenizerPretokHeader *)TOKENIZER_PRETOKEN_BASE;

    if (tokenizer_pretok_header != 0) {
        return XST_SUCCESS;
    }

    fr = f_open(&file, TOKENIZER_PRETOKEN_PATH, FA_READ);
    if (fr != FR_OK) {
        xil_printf("Open tokenizer pretokenizer failed: %s fr=%d\r\n",
                   TOKENIZER_PRETOKEN_PATH, (int)fr);
        return XST_FAILURE;
    }

    if ((uint64_t)f_size(&file) > (uint64_t)TOKENIZER_PRETOKEN_MAX_BYTES) {
        xil_printf("Tokenizer pretokenizer asset too large: %lu\r\n", (unsigned long)f_size(&file));
        f_close(&file);
        return XST_BUFFER_TOO_SMALL;
    }

    total_bytes = (uint32_t)f_size(&file);
    Xil_DCacheInvalidateRange((UINTPTR)TOKENIZER_PRETOKEN_BASE, total_bytes);
    fr = f_read(&file, (void *)TOKENIZER_PRETOKEN_BASE, total_bytes, &bytes_read);
    f_close(&file);
    if ((fr != FR_OK) || (bytes_read != total_bytes)) {
        xil_printf("Tokenizer pretokenizer read failed: fr=%d read=%lu expected=%lu\r\n",
                   (int)fr, (unsigned long)bytes_read, (unsigned long)total_bytes);
        return XST_FAILURE;
    }
    Xil_DCacheFlushRange((UINTPTR)TOKENIZER_PRETOKEN_BASE, total_bytes);
    Xil_DCacheInvalidateRange((UINTPTR)TOKENIZER_PRETOKEN_BASE, total_bytes);

    if ((header->magic != TOKENIZER_PRETOKEN_MAGIC) ||
        (header->version != TOKENIZER_PRETOKEN_VERSION) ||
        (header->total_bytes != total_bytes) ||
        (header->range_count == 0u) ||
        (header->ranges_offset >= total_bytes) ||
        (header->ranges_offset + (header->range_count * sizeof(TokenizerClassRange)) >
         total_bytes)) {
        xil_printf("Tokenizer pretokenizer header invalid\r\n");
        return XST_INVALID_PARAM;
    }

    tokenizer_pretok_header = header;
    tokenizer_class_ranges = (const TokenizerClassRange *)(TOKENIZER_PRETOKEN_BASE +
                                                           header->ranges_offset);
    xil_printf("Tokenizer pretokenizer loaded: ranges=%lu bytes=%lu\r\n",
               (unsigned long)header->range_count, (unsigned long)total_bytes);
    return XST_SUCCESS;
}

static int tokenizer_find_child(uint32_t node_index, uint8_t byte_value, uint32_t *child_index) {
    const TokenizerNode *node;
    uint32_t low;
    uint32_t high;

    if ((tokenizer_nodes == 0) || (tokenizer_edges == 0) ||
        (node_index >= tokenizer_header->node_count)) {
        return 0;
    }

    node = &tokenizer_nodes[node_index];
    low = node->child_base;
    high = node->child_base + node->child_count;
    while (low < high) {
        uint32_t mid = low + ((high - low) >> 1);
        uint8_t mid_byte = tokenizer_edges[mid].byte_value;

        if (mid_byte == byte_value) {
            *child_index = tokenizer_edges[mid].child_index;
            return 1;
        }
        if (mid_byte < byte_value) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }

    return 0;
}

static int tokenizer_lookup_longest(const uint8_t *bytes, uint32_t byte_count,
                                    uint32_t start, uint32_t *token_id,
                                    uint32_t *token_bytes) {
    uint32_t node_index = 0u;
    uint32_t best_id = TOKENIZER_TOKEN_ID_NONE;
    uint32_t best_len = 0u;

    for (uint32_t i = start; i < byte_count; i++) {
        uint32_t child_index;

        if (!tokenizer_find_child(node_index, bytes[i], &child_index)) {
            break;
        }
        node_index = child_index;
        if (tokenizer_nodes[node_index].token_id != TOKENIZER_TOKEN_ID_NONE) {
            best_id = tokenizer_nodes[node_index].token_id;
            best_len = (i - start) + 1u;
        }
    }

    if (best_id == TOKENIZER_TOKEN_ID_NONE) {
        return XST_FAILURE;
    }

    *token_id = best_id;
    *token_bytes = best_len;
    return XST_SUCCESS;
}

static uint8_t ascii_lower_u8(uint8_t c) {
    if ((c >= (uint8_t)'A') && (c <= (uint8_t)'Z')) {
        return (uint8_t)(c + ((uint8_t)'a' - (uint8_t)'A'));
    }
    return c;
}

static int utf8_decode_one(const uint8_t *bytes, uint32_t byte_count, uint32_t pos,
                           uint32_t *codepoint, uint32_t *next_pos) {
    uint8_t b0;

    if ((bytes == 0) || (pos >= byte_count)) {
        return 0;
    }

    b0 = bytes[pos];
    if (b0 < 0x80u) {
        *codepoint = (uint32_t)b0;
        *next_pos = pos + 1u;
        return 1;
    }

    if (((b0 & 0xE0u) == 0xC0u) && ((pos + 1u) < byte_count)) {
        uint8_t b1 = bytes[pos + 1u];
        uint32_t cp;
        if ((b1 & 0xC0u) != 0x80u) {
            goto invalid;
        }
        cp = (((uint32_t)(b0 & 0x1Fu)) << 6) | ((uint32_t)(b1 & 0x3Fu));
        if (cp < 0x80u) {
            goto invalid;
        }
        *codepoint = cp;
        *next_pos = pos + 2u;
        return 1;
    }

    if (((b0 & 0xF0u) == 0xE0u) && ((pos + 2u) < byte_count)) {
        uint8_t b1 = bytes[pos + 1u];
        uint8_t b2 = bytes[pos + 2u];
        uint32_t cp;
        if (((b1 & 0xC0u) != 0x80u) || ((b2 & 0xC0u) != 0x80u)) {
            goto invalid;
        }
        cp = (((uint32_t)(b0 & 0x0Fu)) << 12) |
             (((uint32_t)(b1 & 0x3Fu)) << 6) |
             ((uint32_t)(b2 & 0x3Fu));
        if ((cp < 0x800u) || ((cp >= 0xD800u) && (cp <= 0xDFFFu))) {
            goto invalid;
        }
        *codepoint = cp;
        *next_pos = pos + 3u;
        return 1;
    }

    if (((b0 & 0xF8u) == 0xF0u) && ((pos + 3u) < byte_count)) {
        uint8_t b1 = bytes[pos + 1u];
        uint8_t b2 = bytes[pos + 2u];
        uint8_t b3 = bytes[pos + 3u];
        uint32_t cp;
        if (((b1 & 0xC0u) != 0x80u) || ((b2 & 0xC0u) != 0x80u) ||
            ((b3 & 0xC0u) != 0x80u)) {
            goto invalid;
        }
        cp = (((uint32_t)(b0 & 0x07u)) << 18) |
             (((uint32_t)(b1 & 0x3Fu)) << 12) |
             (((uint32_t)(b2 & 0x3Fu)) << 6) |
             ((uint32_t)(b3 & 0x3Fu));
        if ((cp < 0x10000u) || (cp > 0x10FFFFu)) {
            goto invalid;
        }
        *codepoint = cp;
        *next_pos = pos + 4u;
        return 1;
    }

invalid:
    *codepoint = (uint32_t)b0;
    *next_pos = pos + 1u;
    return 1;
}

static uint32_t tokenizer_unicode_class(uint32_t codepoint) {
    uint32_t low;
    uint32_t high;

    if ((tokenizer_class_ranges == 0) || (tokenizer_pretok_header == 0)) {
        if (((codepoint >= (uint32_t)'A') && (codepoint <= (uint32_t)'Z')) ||
            ((codepoint >= (uint32_t)'a') && (codepoint <= (uint32_t)'z'))) {
            return TOKENIZER_CLASS_LETTER;
        }
        if ((codepoint >= (uint32_t)'0') && (codepoint <= (uint32_t)'9')) {
            return TOKENIZER_CLASS_NUMBER;
        }
        if ((codepoint == 0x09u) || (codepoint == 0x0Au) || (codepoint == 0x0Bu) ||
            (codepoint == 0x0Cu) || (codepoint == 0x0Du) || (codepoint == 0x20u)) {
            return TOKENIZER_CLASS_SPACE;
        }
        return 0u;
    }

    low = 0u;
    high = tokenizer_pretok_header->range_count;
    while (low < high) {
        uint32_t mid = low + ((high - low) >> 1);
        const TokenizerClassRange *range = &tokenizer_class_ranges[mid];

        if (codepoint < range->start) {
            high = mid;
        } else if (codepoint > range->end) {
            low = mid + 1u;
        } else {
            return range->flags;
        }
    }
    return 0u;
}

static int tokenizer_is_letter(uint32_t codepoint) {
    return (tokenizer_unicode_class(codepoint) & TOKENIZER_CLASS_LETTER) != 0u;
}

static int tokenizer_is_number(uint32_t codepoint) {
    return (tokenizer_unicode_class(codepoint) & TOKENIZER_CLASS_NUMBER) != 0u;
}

static int tokenizer_is_space(uint32_t codepoint) {
    return (tokenizer_unicode_class(codepoint) & TOKENIZER_CLASS_SPACE) != 0u;
}

static int tokenizer_is_crlf(uint32_t codepoint) {
    return (codepoint == 0x0Au) || (codepoint == 0x0Du);
}

static int tokenizer_is_symbol_for_regex(uint32_t codepoint) {
    return !tokenizer_is_space(codepoint) &&
           !tokenizer_is_letter(codepoint) &&
           !tokenizer_is_number(codepoint);
}

static int tokenizer_match_contraction(const uint8_t *bytes, uint32_t byte_count,
                                       uint32_t pos, uint32_t *end_pos) {
    uint8_t c1;
    uint8_t c2;

    if (((pos + 2u) > byte_count) || (bytes[pos] != (uint8_t)'\'')) {
        return 0;
    }

    c1 = ascii_lower_u8(bytes[pos + 1u]);
    if ((c1 == (uint8_t)'s') || (c1 == (uint8_t)'t') ||
        (c1 == (uint8_t)'m') || (c1 == (uint8_t)'d')) {
        *end_pos = pos + 2u;
        return 1;
    }

    if ((pos + 3u) > byte_count) {
        return 0;
    }
    c2 = ascii_lower_u8(bytes[pos + 2u]);
    if (((c1 == (uint8_t)'r') && (c2 == (uint8_t)'e')) ||
        ((c1 == (uint8_t)'v') && (c2 == (uint8_t)'e')) ||
        ((c1 == (uint8_t)'l') && (c2 == (uint8_t)'l'))) {
        *end_pos = pos + 3u;
        return 1;
    }
    return 0;
}

static int tokenizer_match_letter_piece(const uint8_t *bytes, uint32_t byte_count,
                                        uint32_t pos, uint32_t *end_pos) {
    uint32_t cp;
    uint32_t next;
    uint32_t scan;

    if (!utf8_decode_one(bytes, byte_count, pos, &cp, &next)) {
        return 0;
    }

    if (tokenizer_is_letter(cp)) {
        scan = next;
    } else {
        uint32_t next_cp;
        uint32_t after_next;

        if (tokenizer_is_crlf(cp) || tokenizer_is_number(cp)) {
            return 0;
        }
        if (!utf8_decode_one(bytes, byte_count, next, &next_cp, &after_next) ||
            !tokenizer_is_letter(next_cp)) {
            return 0;
        }
        scan = after_next;
    }

    while (scan < byte_count) {
        uint32_t next_cp;
        uint32_t after_next;

        utf8_decode_one(bytes, byte_count, scan, &next_cp, &after_next);
        if (!tokenizer_is_letter(next_cp)) {
            break;
        }
        scan = after_next;
    }

    *end_pos = scan;
    return 1;
}

static int tokenizer_match_number_piece(const uint8_t *bytes, uint32_t byte_count,
                                        uint32_t pos, uint32_t *end_pos) {
    uint32_t scan = pos;
    uint32_t count = 0u;

    while ((scan < byte_count) && (count < 3u)) {
        uint32_t cp;
        uint32_t next;

        utf8_decode_one(bytes, byte_count, scan, &cp, &next);
        if (!tokenizer_is_number(cp)) {
            break;
        }
        scan = next;
        count++;
    }

    if (count == 0u) {
        return 0;
    }
    *end_pos = scan;
    return 1;
}

static int tokenizer_match_symbol_piece(const uint8_t *bytes, uint32_t byte_count,
                                        uint32_t pos, uint32_t *end_pos) {
    uint32_t scan = pos;
    uint32_t cp;
    uint32_t next;
    uint32_t symbol_count = 0u;

    utf8_decode_one(bytes, byte_count, scan, &cp, &next);
    if ((cp == (uint32_t)' ') && (next < byte_count)) {
        uint32_t next_cp;
        uint32_t after_next;

        utf8_decode_one(bytes, byte_count, next, &next_cp, &after_next);
        if (tokenizer_is_symbol_for_regex(next_cp)) {
            scan = next;
        }
    }

    while (scan < byte_count) {
        utf8_decode_one(bytes, byte_count, scan, &cp, &next);
        if (!tokenizer_is_symbol_for_regex(cp)) {
            break;
        }
        scan = next;
        symbol_count++;
    }

    if (symbol_count == 0u) {
        return 0;
    }

    while (scan < byte_count) {
        utf8_decode_one(bytes, byte_count, scan, &cp, &next);
        if (!tokenizer_is_crlf(cp)) {
            break;
        }
        scan = next;
    }

    *end_pos = scan;
    return 1;
}

static int tokenizer_match_newline_space_piece(const uint8_t *bytes, uint32_t byte_count,
                                               uint32_t pos, uint32_t *end_pos) {
    uint32_t scan = pos;
    uint32_t last_crlf_end = pos;

    while (scan < byte_count) {
        uint32_t cp;
        uint32_t next;

        utf8_decode_one(bytes, byte_count, scan, &cp, &next);
        if (!tokenizer_is_space(cp)) {
            break;
        }
        if (tokenizer_is_crlf(cp)) {
            last_crlf_end = next;
        }
        scan = next;
    }

    if (last_crlf_end == pos) {
        return 0;
    }
    *end_pos = last_crlf_end;
    return 1;
}

static int tokenizer_match_space_not_before_nonspace_piece(const uint8_t *bytes,
                                                           uint32_t byte_count,
                                                           uint32_t pos,
                                                           uint32_t *end_pos) {
    uint32_t scan = pos;
    uint32_t count = 0u;
    uint32_t last_start = pos;

    while (scan < byte_count) {
        uint32_t cp;
        uint32_t next;

        utf8_decode_one(bytes, byte_count, scan, &cp, &next);
        if (!tokenizer_is_space(cp)) {
            break;
        }
        last_start = scan;
        scan = next;
        count++;
    }

    if (count == 0u) {
        return 0;
    }
    if (scan >= byte_count) {
        *end_pos = scan;
        return 1;
    }
    if (count >= 2u) {
        *end_pos = last_start;
        return 1;
    }
    return 0;
}

static int tokenizer_match_space_piece(const uint8_t *bytes, uint32_t byte_count,
                                       uint32_t pos, uint32_t *end_pos) {
    uint32_t scan = pos;
    uint32_t count = 0u;

    while (scan < byte_count) {
        uint32_t cp;
        uint32_t next;

        utf8_decode_one(bytes, byte_count, scan, &cp, &next);
        if (!tokenizer_is_space(cp)) {
            break;
        }
        scan = next;
        count++;
    }

    if (count == 0u) {
        return 0;
    }
    *end_pos = scan;
    return 1;
}

static uint32_t tokenizer_next_regex_piece(const uint8_t *bytes, uint32_t byte_count,
                                           uint32_t pos) {
    uint32_t end_pos;
    uint32_t cp;
    uint32_t next;

    if (tokenizer_match_contraction(bytes, byte_count, pos, &end_pos) ||
        tokenizer_match_letter_piece(bytes, byte_count, pos, &end_pos) ||
        tokenizer_match_number_piece(bytes, byte_count, pos, &end_pos) ||
        tokenizer_match_symbol_piece(bytes, byte_count, pos, &end_pos) ||
        tokenizer_match_newline_space_piece(bytes, byte_count, pos, &end_pos) ||
        tokenizer_match_space_not_before_nonspace_piece(bytes, byte_count, pos, &end_pos) ||
        tokenizer_match_space_piece(bytes, byte_count, pos, &end_pos)) {
        return end_pos;
    }

    utf8_decode_one(bytes, byte_count, pos, &cp, &next);
    (void)cp;
    return next;
}

static int tokenizer_encode_piece(const uint8_t *bytes, uint32_t piece_start,
                                  uint32_t piece_end, uint32_t *tokens,
                                  uint32_t max_tokens, uint32_t *out_count) {
    uint32_t pos = piece_start;

    while (pos < piece_end) {
        uint32_t token_id;
        uint32_t used_bytes;
        int status;

        if (*out_count >= max_tokens) {
            smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_BUFFER_TOO_SMALL);
            return XST_BUFFER_TOO_SMALL;
        }

        status = tokenizer_lookup_longest(bytes, piece_end, pos, &token_id, &used_bytes);
        if (status != XST_SUCCESS) {
            xil_printf("Tokenizer no match at byte offset %lu value=0x%02x\r\n",
                       (unsigned long)pos, bytes[pos]);
            smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, status);
            return status;
        }
        tokens[(*out_count)++] = token_id;
        pos += used_bytes;
    }

    return XST_SUCCESS;
}

static int tokenizer_encode_fallback_bytes(const char *text, uint32_t byte_count,
                                           uint32_t *tokens, uint32_t max_tokens,
                                           uint32_t *token_count) {
    const uint8_t *bytes = (const uint8_t *)text;
    uint32_t out_count = 0u;

    if (max_tokens == 0u) {
        return XST_BUFFER_TOO_SMALL;
    }
    tokens[out_count++] = TOKENIZER_BOS_ID;

    for (uint32_t pos = 0u; pos < byte_count; pos++) {
        if (out_count >= max_tokens) {
            smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_BUFFER_TOO_SMALL);
            return XST_BUFFER_TOO_SMALL;
        }
        tokens[out_count++] = ((uint32_t)bytes[pos]) + 1u;
    }

    *token_count = out_count;
    xil_printf("Fallback byte tokenizer produced %lu token ids\r\n", (unsigned long)out_count);
    return XST_SUCCESS;
}

static int tokenizer_encode_text(const char *text, uint32_t byte_count,
                                 uint32_t *tokens, uint32_t max_tokens,
                                 uint32_t *token_count) {
    const uint8_t *bytes = (const uint8_t *)text;
    uint32_t out_count = 0u;
    uint32_t pos = 0u;
    int status;

    status = tokenizer_load_from_emmc();
    if (status != XST_SUCCESS) {
        xil_printf("Tokenizer trie unavailable; using fallback byte tokenizer\r\n");
        return tokenizer_encode_fallback_bytes(text, byte_count, tokens, max_tokens, token_count);
    }
    status = tokenizer_pretok_load_from_emmc();
    if (status != XST_SUCCESS) {
        xil_printf("Tokenizer Unicode pretokenizer unavailable; using ASCII class fallback\r\n");
    }

    if (max_tokens == 0u) {
        return XST_BUFFER_TOO_SMALL;
    }
    tokens[out_count++] = tokenizer_header->bos_token_id;

    while (pos < byte_count) {
        uint32_t piece_end = tokenizer_next_regex_piece(bytes, byte_count, pos);

        if ((piece_end <= pos) || (piece_end > byte_count)) {
            smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_FAILURE);
            return XST_FAILURE;
        }
        status = tokenizer_encode_piece(bytes, pos, piece_end, tokens, max_tokens, &out_count);
        if (status != XST_SUCCESS) {
            return status;
        }
        pos = piece_end;
    }

    *token_count = out_count;
    xil_printf("PS tokenizer produced %lu token ids with Unicode regex pretokenizer\r\n",
               (unsigned long)out_count);
    return XST_SUCCESS;
}

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
    int32_t scaled = (scaled_f >= 0.0f) ? (int32_t)(scaled_f + 0.5f) : (int32_t)(scaled_f - 0.5f);

    if (scaled > 127) {
        scaled = 127;
    } else if (scaled < -128) {
        scaled = -128;
    }
    return (int8_t)scaled;
}

static int model_aux_load_from_emmc(void) {
    FIL file;
    FRESULT fr;
    UINT bytes_read;
    uint32_t total_bytes;
    const ModelAuxHeader *header = (const ModelAuxHeader *)MODEL_AUX_BASE;

    if (model_aux_available != 0u) {
        return XST_SUCCESS;
    }

    fr = f_open(&file, MODEL_AUX_PATH, FA_READ);
    if (fr != FR_OK) {
        xil_printf("AUX asset not available, using fallback numeric path: fr=%d\r\n", (int)fr);
        model_aux_header = 0;
        model_aux_available = 0u;
        return XST_FAILURE;
    }

    if ((uint64_t)f_size(&file) > (uint64_t)MODEL_AUX_MAX_BYTES) {
        xil_printf("AUX asset too large: %lu\r\n", (unsigned long)f_size(&file));
        f_close(&file);
        return XST_BUFFER_TOO_SMALL;
    }

    total_bytes = (uint32_t)f_size(&file);
    Xil_DCacheInvalidateRange((UINTPTR)MODEL_AUX_BASE, total_bytes);
    fr = f_read(&file, (void *)MODEL_AUX_BASE, total_bytes, &bytes_read);
    f_close(&file);
    if ((fr != FR_OK) || (bytes_read != total_bytes)) {
        xil_printf("AUX read failed: fr=%d read=%lu expected=%lu\r\n",
                   (int)fr, (unsigned long)bytes_read, (unsigned long)total_bytes);
        return XST_FAILURE;
    }
    Xil_DCacheFlushRange((UINTPTR)MODEL_AUX_BASE, total_bytes);
    Xil_DCacheInvalidateRange((UINTPTR)MODEL_AUX_BASE, total_bytes);

    if ((header->magic != MODEL_AUX_MAGIC) ||
        (header->version != MODEL_AUX_VERSION) ||
        (header->total_bytes != total_bytes) ||
        (header->hidden_size != LAYER_HIDDEN_SIZE) ||
        (header->ffn_size != LAYER_FFN_SIZE) ||
        (header->num_layers != MODEL_TOTAL_LAYERS) ||
        (header->scale_count != 7u)) {
        xil_printf("AUX header invalid\r\n");
        return XST_INVALID_PARAM;
    }

    model_aux_header = header;
    model_aux_available = 1u;
    xil_printf("AUX loaded: bytes=%lu\r\n", (unsigned long)total_bytes);
    return XST_SUCCESS;
}

static int model_rope_load_from_emmc(void) {
    FIL file;
    FRESULT fr;
    UINT bytes_read;
    uint32_t total_bytes;
    const RopeAssetHeader *header = (const RopeAssetHeader *)ROPE_ASSET_BASE;

    if (rope_available != 0u) {
        return XST_SUCCESS;
    }

    fr = f_open(&file, ROPE_ASSET_PATH, FA_READ);
    if (fr != FR_OK) {
        xil_printf("RoPE asset not available, rotary embedding disabled: fr=%d\r\n", (int)fr);
        rope_header = 0;
        rope_q15_table = 0;
        rope_available = 0u;
        return XST_FAILURE;
    }

    if ((uint64_t)f_size(&file) > (uint64_t)ROPE_ASSET_MAX_BYTES) {
        xil_printf("RoPE asset too large: %lu\r\n", (unsigned long)f_size(&file));
        f_close(&file);
        return XST_BUFFER_TOO_SMALL;
    }

    total_bytes = (uint32_t)f_size(&file);
    Xil_DCacheInvalidateRange((UINTPTR)ROPE_ASSET_BASE, total_bytes);
    fr = f_read(&file, (void *)ROPE_ASSET_BASE, total_bytes, &bytes_read);
    f_close(&file);
    if ((fr != FR_OK) || (bytes_read != total_bytes)) {
        xil_printf("RoPE read failed: fr=%d read=%lu expected=%lu\r\n",
                   (int)fr, (unsigned long)bytes_read, (unsigned long)total_bytes);
        return XST_FAILURE;
    }
    Xil_DCacheFlushRange((UINTPTR)ROPE_ASSET_BASE, total_bytes);
    Xil_DCacheInvalidateRange((UINTPTR)ROPE_ASSET_BASE, total_bytes);

    if ((header->magic != ROPE_ASSET_MAGIC) ||
        (header->version != ROPE_ASSET_VERSION) ||
        (header->total_bytes != total_bytes) ||
        (header->max_positions < INFER_CONTEXT_MAX_TOKENS) ||
        (header->head_dim != MODEL_HEAD_DIM) ||
        (header->pair_count != (MODEL_HEAD_DIM / 2u)) ||
        (header->table_offset >= total_bytes) ||
        (header->q15_scale == 0u)) {
        xil_printf("RoPE header invalid\r\n");
        return XST_INVALID_PARAM;
    }

    rope_header = header;
    rope_q15_table = (const int16_t *)(ROPE_ASSET_BASE + header->table_offset);
    rope_available = 1u;
    xil_printf("RoPE loaded: positions=%lu head_dim=%lu bytes=%lu\r\n",
               (unsigned long)header->max_positions,
               (unsigned long)header->head_dim,
               (unsigned long)total_bytes);
    return XST_SUCCESS;
}

static const uint16_t *model_aux_bf16_ptr(uint32_t offset) {
    return (const uint16_t *)(MODEL_AUX_BASE + offset);
}

static const uint16_t *model_aux_input_norm(uint32_t layer_index) {
    if (model_aux_available == 0u) {
        return 0;
    }
    return model_aux_bf16_ptr(model_aux_header->input_norm_offset +
                              (layer_index * LAYER_HIDDEN_SIZE * 2u));
}

static const uint16_t *model_aux_post_norm(uint32_t layer_index) {
    if (model_aux_available == 0u) {
        return 0;
    }
    return model_aux_bf16_ptr(model_aux_header->post_norm_offset +
                              (layer_index * LAYER_HIDDEN_SIZE * 2u));
}

static const uint16_t *model_aux_attn_sub_norm(uint32_t layer_index) {
    if (model_aux_available == 0u) {
        return 0;
    }
    return model_aux_bf16_ptr(model_aux_header->attn_sub_norm_offset +
                              (layer_index * LAYER_HIDDEN_SIZE * 2u));
}

static const uint16_t *model_aux_ffn_sub_norm(uint32_t layer_index) {
    if (model_aux_available == 0u) {
        return 0;
    }
    return model_aux_bf16_ptr(model_aux_header->ffn_sub_norm_offset +
                              (layer_index * LAYER_FFN_SIZE * 2u));
}

static const uint16_t *model_aux_final_norm(void) {
    if (model_aux_available == 0u) {
        return 0;
    }
    return model_aux_bf16_ptr(model_aux_header->final_norm_offset);
}

static float model_aux_weight_scale(uint32_t layer_index, uint32_t tensor_id) {
    uint32_t scale_index;

    if ((model_aux_available == 0u) || (model_aux_header == 0) ||
        (tensor_id < LAYER_TENSOR_Q) || (tensor_id > LAYER_TENSOR_DOWN) ||
        (model_aux_header->scale_count < 7u)) {
        return 1.0f;
    }
    scale_index = (layer_index * model_aux_header->scale_count) + (tensor_id - 1u);
    return bf16_to_float(model_aux_bf16_ptr(model_aux_header->scale_offset)[scale_index]);
}

static int model_load_embedding_token_i8(uint32_t token_id, int8_t *hidden_act,
                                         float *hidden_factor) {
    FIL file;
    FRESULT fr;
    UINT bytes_read;
    char path[32];
    uint32_t shard;
    uint32_t row_in_shard;
    uint32_t row_offset;
    uint16_t *row_bf16 = (uint16_t *)MODEL_EMBED_ROW_BASE;
    float max_abs = 0.0f;

    if (token_id >= MODEL_VOCAB_SIZE) {
        return XST_INVALID_PARAM;
    }

    shard = token_id / MODEL_EMBED_ROWS_PER_SHARD;
    row_in_shard = token_id - (shard * MODEL_EMBED_ROWS_PER_SHARD);
    row_offset = row_in_shard * MODEL_EMBED_ROW_BYTES;
    snprintf(path, sizeof(path), "0:/BITNET/EMB/E%02lu.BIN", (unsigned long)shard);

    fr = f_open(&file, path, FA_READ);
    if (fr != FR_OK) {
        xil_printf("Embedding shard not available, using fallback hidden: %s fr=%d\r\n",
                   path, (int)fr);
        return XST_FAILURE;
    }
    fr = f_lseek(&file, row_offset);
    if (fr != FR_OK) {
        f_close(&file);
        return XST_FAILURE;
    }

    Xil_DCacheInvalidateRange((UINTPTR)MODEL_EMBED_ROW_BASE, MODEL_EMBED_ROW_BYTES);
    fr = f_read(&file, (void *)MODEL_EMBED_ROW_BASE, MODEL_EMBED_ROW_BYTES, &bytes_read);
    f_close(&file);
    if ((fr != FR_OK) || (bytes_read != MODEL_EMBED_ROW_BYTES)) {
        xil_printf("Embedding row read failed: fr=%d read=%lu\r\n",
                   (int)fr, (unsigned long)bytes_read);
        return XST_FAILURE;
    }
    Xil_DCacheInvalidateRange((UINTPTR)MODEL_EMBED_ROW_BASE, MODEL_EMBED_ROW_BYTES);

    for (uint32_t i = 0u; i < LAYER_HIDDEN_SIZE; i++) {
        float value = bf16_to_float(row_bf16[i]);
        float magnitude = (value < 0.0f) ? -value : value;
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    if (max_abs <= 1.0e-20f) {
        max_abs = 1.0f;
    }
    float quant_scale = 127.0f / max_abs;
    for (uint32_t i = 0u; i < LAYER_HIDDEN_SIZE; i++) {
        hidden_act[i] = float_to_i8_scaled(bf16_to_float(row_bf16[i]), quant_scale);
    }
    if (hidden_factor != 0) {
        *hidden_factor = 1.0f / quant_scale;
    }
    return XST_SUCCESS;
}

static int emmc_load_layer_weights(uint32_t layer_index, uint32_t layer_slot) {
    const uint32_t tensors[] = {
        LAYER_TENSOR_Q,
        LAYER_TENSOR_K,
        LAYER_TENSOR_V,
        LAYER_TENSOR_O,
        LAYER_TENSOR_GATE,
        LAYER_TENSOR_UP,
        LAYER_TENSOR_DOWN,
    };
    char path[64];

    smoke_status_update(SMOKE_STAGE_EMMC_LOAD, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    for (uint32_t i = 0u; i < (sizeof(tensors) / sizeof(tensors[0])); i++) {
        uint32_t tensor_id = tensors[i];
        uint32_t offset = layer_tensor_offset(tensor_id);
        uint32_t bytes = layer_tensor_weight_bytes(tensor_id);
        uintptr_t dst = LAYER_WEIGHT_BASE + ((uintptr_t)layer_slot * LAYER_WEIGHT_STRIDE) + offset;
        int status;

        smoke_status->reserved0 = tensor_id;
        smoke_status->reserved1 = layer_index + 1u;
        smoke_status_flush();

        snprintf(path, sizeof(path), "0:/BITNET/L%02lu/%s",
                 (unsigned long)layer_index, layer_tensor_file_name(tensor_id));
        xil_printf("Loading %s -> 0x%08lx (%lu bytes)\r\n",
                   path, (unsigned long)dst, (unsigned long)bytes);
        status = emmc_read_exact(path, dst, bytes);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    return XST_SUCCESS;
}

/*
 * Streaming mode keeps only two tightly packed layer slots.  The layout is
 * identical to MODEL_U8_RESIDENT_BASE, so the fused QKV and Gate/Up paths
 * remain enabled while the inactive slot is filled with the next layer.
 */
static int emmc_load_layer_weights_tight(uint32_t layer_index, uint32_t slot,
                                         uint64_t *elapsed_cycles) {
    const uint32_t tensors[] = {
        LAYER_TENSOR_Q, LAYER_TENSOR_K, LAYER_TENSOR_V, LAYER_TENSOR_O,
        LAYER_TENSOR_GATE, LAYER_TENSOR_UP, LAYER_TENSOR_DOWN,
    };
    char path[64];
    uint64_t t0 = perf_now_cycles();

    if (slot > 1u) {
        return XST_INVALID_PARAM;
    }
    for (uint32_t i = 0u; i < (sizeof(tensors) / sizeof(tensors[0])); i++) {
        uint32_t tensor_id = tensors[i];
        uint32_t bytes = layer_tensor_weight_bytes(tensor_id);
        uintptr_t dst = LAYER_DOUBLEBUF_BASE +
                        ((uintptr_t)slot * LAYER_DOUBLEBUF_STRIDE) +
                        (uintptr_t)layer_tight_tensor_offset(tensor_id);
        int status;

        snprintf(path, sizeof(path), "0:/BITNET/L%02lu/%s",
                 (unsigned long)layer_index, layer_tensor_file_name(tensor_id));
        status = emmc_read_exact(path, dst, bytes);
        if (status != XST_SUCCESS) {
            return status;
        }
    }
    if (elapsed_cycles != 0) {
        *elapsed_cycles = perf_now_cycles() - t0;
    }
    return XST_SUCCESS;
}

#if defined(BITNET_LINUX)
typedef struct {
    pthread_t thread;
    uint32_t layer_index;
    uint32_t slot;
    int status;
    uint64_t elapsed_cycles;
    uint8_t active;
} LayerPrefetchJob;

static void *layer_prefetch_main(void *argument) {
    LayerPrefetchJob *job = (LayerPrefetchJob *)argument;

    job->status = emmc_load_layer_weights_tight(job->layer_index, job->slot,
                                                &job->elapsed_cycles);
    return 0;
}
#else
typedef struct {
    uint32_t layer_index;
    uint32_t slot;
    int status;
    uint64_t elapsed_cycles;
    uint8_t active;
} LayerPrefetchJob;
#endif

static int layer_prefetch_start(LayerPrefetchJob *job, uint32_t layer_index,
                                uint32_t slot) {
    memset(job, 0, sizeof(*job));
    job->layer_index = layer_index;
    job->slot = slot;
#if defined(BITNET_LINUX)
    if (pthread_create(&job->thread, 0, layer_prefetch_main, job) == 0) {
        job->active = 1u;
        return XST_SUCCESS;
    }
#endif
    job->status = emmc_load_layer_weights_tight(layer_index, slot,
                                                &job->elapsed_cycles);
    return job->status;
}

static int layer_prefetch_wait(LayerPrefetchJob *job, InferPerf *perf) {
    if (job == 0) {
        return XST_INVALID_PARAM;
    }
#if defined(BITNET_LINUX)
    if (job->active != 0u) {
        (void)pthread_join(job->thread, 0);
        job->active = 0u;
    }
#endif
    if (perf != 0) {
        perf->emmc_cycles += job->elapsed_cycles;
    }
    return job->status;
}

static int infer_layer_streaming_enabled(void) {
#if defined(BITNET_LINUX)
    const char *value = getenv("BITNET_LAYER_DOUBLE_BUFFER");
    return value != 0 && value[0] == '1';
#else
    return 0;
#endif
}

static int infer_preload_resident_weights(uint32_t first_layer, uint32_t num_layers,
                                          InferPerf *perf) {
    const uint32_t tensors[] = {
        LAYER_TENSOR_Q,
        LAYER_TENSOR_K,
        LAYER_TENSOR_V,
        LAYER_TENSOR_O,
        LAYER_TENSOR_GATE,
        LAYER_TENSOR_UP,
        LAYER_TENSOR_DOWN,
    };
    uint64_t t0;
    uint64_t t1;
    char path[64];

    smoke_status_update(SMOKE_STAGE_INFER_LOAD, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    smoke_status->reserved0 = (uint32_t)(MODEL_U8_RESIDENT_BASE >> 20);
    smoke_status->reserved1 = MODEL_U8_RESIDENT_BYTES >> 20;
    smoke_status_flush();

    xil_printf("Preloading U8 weights to PS DDR: base=0x%08lx bytes=%lu window=L%02lu..L%02lu\r\n",
               (unsigned long)MODEL_U8_RESIDENT_BASE,
               (unsigned long)MODEL_U8_RESIDENT_BYTES,
               (unsigned long)first_layer,
               (unsigned long)(first_layer + num_layers - 1u));

    t0 = perf_now_cycles();
    for (uint32_t layer_offset = 0u; layer_offset < num_layers; layer_offset++) {
        uint32_t layer_index = first_layer + layer_offset;

        if (infer_resident_valid[layer_index] != 0u) {
            perf->cache_hits++;
            continue;
        }

        for (uint32_t i = 0u; i < (sizeof(tensors) / sizeof(tensors[0])); i++) {
            uint32_t tensor_id = tensors[i];
            uint32_t bytes = layer_tensor_weight_bytes(tensor_id);
            uintptr_t dst = layer_resident_tensor_addr(layer_index, tensor_id);
            int status;

            smoke_status->reserved0 = tensor_id;
            smoke_status->reserved1 = layer_index + 1u;
            smoke_status_flush();

            snprintf(path, sizeof(path), "0:/BITNET/L%02lu/%s",
                     (unsigned long)layer_index, layer_tensor_file_name(tensor_id));
            xil_printf("Preload %s -> 0x%08lx (%lu bytes)\r\n",
                       path, (unsigned long)dst, (unsigned long)bytes);
            status = emmc_read_exact(path, dst, bytes);
            if (status != XST_SUCCESS) {
                t1 = perf_now_cycles();
                perf->emmc_cycles += (t1 - t0);
                return status;
            }
        }

        infer_resident_valid[layer_index] = 1u;
        perf->cache_misses++;
    }
    t1 = perf_now_cycles();
    perf->emmc_cycles += (t1 - t0);

    xil_printf("U8 resident preload done: loaded_layers=%lu total_bytes=%lu\r\n",
               (unsigned long)num_layers,
               (unsigned long)(num_layers * LAYER_TIGHT_BYTES));
    return XST_SUCCESS;
}

static int run_projection_checked(uint32_t layer_index, uint32_t tensor_id,
                                  const int8_t *activations, const uint8_t *weights,
                                  uint32_t out_features, uint32_t in_features,
                                  int32_t *outputs, uint32_t flags,
                                  uint32_t *mismatches, uint32_t *first_mismatch) {
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint32_t weight_bytes = (out_features / 4u) * in_features;
    int status;

    smoke_status->reserved0 = tensor_id;
    smoke_status->reserved1 = layer_index + 1u;
    smoke_status_flush();

    Xil_DCacheInvalidateRange((UINTPTR)weights, weight_bytes);
    for (uint32_t n = 0u; n < out_features; n++) {
        outputs[n] = 0;
    }

    status = bitnet_accel_run_gemv_chunked(&AxiDma, (uint32_t *)REAL_TX_BASE, tx_capacity_words,
                                           activations, weights, out_features, in_features,
                                           REAL_CHUNK_OUT, flags, outputs);
    if (status != XST_SUCCESS) {
        smoke_status_update(SMOKE_STAGE_LAYER_RUN, SMOKE_RESULT_FAIL, status);
        return status;
    }

    for (uint32_t n = 0u; n < out_features; n++) {
        int32_t ref = bitnet_accel_reference_dot4_layout(
            activations, weights, n, in_features, flags);
        if (n < STATUS_SAMPLE_N) {
            smoke_status->hw[n] = (uint32_t)outputs[n];
            smoke_status->ref[n] = (uint32_t)ref;
        }
        if (outputs[n] != ref) {
            if (*mismatches == 0u) {
                *first_mismatch = ((layer_index & 0xFFu) << 24) |
                                  ((tensor_id & 0xFFu) << 16) |
                                  (n & 0xFFFFu);
                smoke_status->first_mismatch = *first_mismatch;
            }
            (*mismatches)++;
        }
    }

    smoke_status->mismatches = *mismatches;
    smoke_status_flush();

    if (*mismatches != 0u) {
        smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_FAIL, XST_FAILURE);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static void prefill_batch_buffers_init(PrefillBatchBuffers *bufs) {
    bufs->hidden = (int8_t *)PREFILL_HIDDEN_BASE;
    bufs->norm = (int8_t *)PREFILL_NORM_BASE;
    bufs->attention = (int8_t *)PREFILL_ATT_BASE;
    bufs->ffn = (int8_t *)PREFILL_FFN_BASE;
    bufs->hidden_factors = (float *)PREFILL_HIDDEN_FACTOR_BASE;
    bufs->projection_factors = (float *)PREFILL_PROJECTION_FACTOR_BASE;
    bufs->scratch = (float *)PREFILL_SCRATCH_BASE;
    bufs->projection_stream = (int32_t *)PREFILL_PROJECTION_STREAM_BASE;
    bufs->projection_rows = (int32_t *)PREFILL_PROJECTION_ROWS_BASE;
}

static void unpack_gemm_rows(const int32_t *stream_outputs,
                             int32_t *row_outputs,
                             uint32_t batch_rows,
                             uint32_t out_features) {
    uint32_t output_groups = out_features / 4u;

    for (uint32_t row = 0u; row < batch_rows; row++) {
        int32_t *dst = row_outputs + ((uintptr_t)row * out_features);

        for (uint32_t group = 0u; group < output_groups; group++) {
            const int32_t *src = stream_outputs +
                ((uintptr_t)group * batch_rows * 4u) + (row * 4u);
            uint32_t out_base = group * 4u;

            dst[out_base + 0u] = src[0];
            dst[out_base + 1u] = src[1];
            dst[out_base + 2u] = src[2];
            dst[out_base + 3u] = src[3];
        }
    }
}

#if defined(BITNET_LINUX)
static void unpack_gemm_lane_rows(const int32_t *stream_outputs,
                                  int32_t *row_outputs,
                                  uint32_t batch_rows,
                                  uint32_t lane_out_features,
                                  uint32_t total_out_features,
                                  uint32_t out_base) {
    uint32_t output_groups = lane_out_features / 4u;

    for (uint32_t row = 0u; row < batch_rows; row++) {
        int32_t *dst = row_outputs + ((uintptr_t)row * total_out_features) + out_base;
        for (uint32_t group = 0u; group < output_groups; group++) {
            const int32_t *src = stream_outputs +
                ((uintptr_t)group * batch_rows * 4u) + (row * 4u);
            uint32_t lane_base = group * 4u;
            dst[lane_base + 0u] = src[0];
            dst[lane_base + 1u] = src[1];
            dst[lane_base + 2u] = src[2];
            dst[lane_base + 3u] = src[3];
        }
    }
}

static int run_projection_batch_dual_linux(uint32_t layer_index,
                                           uint32_t tensor_id,
                                           const int8_t *activations,
                                           const uint8_t *weights,
                                           uint32_t batch_rows,
                                           uint32_t out_features,
                                           uint32_t in_features,
                                           uint32_t flags,
                                           int32_t *stream_outputs,
                                           int32_t *row_outputs) {
    const uint32_t lane_out = out_features / 2u;
    const uint32_t lane_capacity_words =
        (REAL_TX_BYTES / 2u) / sizeof(uint32_t);
    uint32_t *lane0_stream = (uint32_t *)REAL_TX_BASE;
    uint32_t *lane1_stream =
        (uint32_t *)(REAL_TX_BASE + (REAL_TX_BYTES / 2u));
    uint32_t lane_words;
    uint32_t lane1_weight_offset;
    uint32_t prefix_words;
    PlddrPacketCacheEntry *cache_entry;
    int status;

    if (!dual_lane_ready || batch_rows == 0u || (out_features < 8u) ||
        ((out_features & 0x7u) != 0u) || row_outputs == 0 ||
        stream_outputs == 0) {
        return BITNET_DUAL_UNAVAILABLE;
    }

    lane1_weight_offset = (lane_out / 4u) * in_features;
    lane_words = bitnet_accel_gemm_stream_words(lane_out, in_features, batch_rows);
    if (lane_words == 0u || lane_words > lane_capacity_words) {
        return BITNET_DUAL_UNAVAILABLE;
    }

    if (layer_index >= MODEL_TOTAL_LAYERS || tensor_id >= 8u) {
        return XST_INVALID_PARAM;
    }
    cache_entry = &plddr_packet_cache[layer_index][tensor_id];
    if (cache_entry->valid != 0u &&
        cache_entry->batch_rows == batch_rows &&
        cache_entry->out_features == out_features &&
        cache_entry->in_features == in_features &&
        cache_entry->lane_words == lane_words) {
        status = bitnet_accel_build_gemm_prefix(
            lane0_stream, lane_capacity_words, activations, batch_rows,
            lane_out, in_features, flags, &prefix_words);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = packet_cache_copy_to(
            cache_entry->lane0_addr, lane0_stream,
            (size_t)prefix_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            return status;
        }
        status = packet_cache_copy_to(
            cache_entry->lane1_addr, lane0_stream,
            (size_t)prefix_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            return status;
        }
    } else {
        status = bitnet_accel_build_gemm_stream(
            lane0_stream, lane_capacity_words, activations, weights, batch_rows,
            lane_out, in_features, flags);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = bitnet_accel_build_gemm_stream(
            lane1_stream, lane_capacity_words, activations,
            weights + lane1_weight_offset, batch_rows, lane_out, in_features,
            flags);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = plddr_packet_cache_alloc(
            lane_words * sizeof(uint32_t),
            &cache_entry->lane0_addr, &cache_entry->lane1_addr);
        if (status != XST_SUCCESS) {
            cache_entry->valid = 0u;
            return BITNET_DUAL_UNAVAILABLE;
        }
        status = packet_cache_copy_to(
            cache_entry->lane0_addr, lane0_stream,
            (size_t)lane_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            cache_entry->valid = 0u;
            return status;
        }
        status = packet_cache_copy_to(
            cache_entry->lane1_addr, lane1_stream,
            (size_t)lane_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            cache_entry->valid = 0u;
            return status;
        }
        cache_entry->batch_rows = batch_rows;
        cache_entry->out_features = out_features;
        cache_entry->in_features = in_features;
        cache_entry->lane_words = lane_words;
        cache_entry->valid = 1u;
    }

    status = bitnet_accel_run_dual_physical(
        &AxiDma, &AxiDmaLane1,
        cache_entry->lane0_addr, lane_words,
        cache_entry->lane1_addr, lane_words,
        stream_outputs, batch_rows * out_features);
    if (status != XST_SUCCESS) {
        return status;
    }

    unpack_gemm_lane_rows(stream_outputs, row_outputs, batch_rows, lane_out,
                          out_features, 0u);
    unpack_gemm_lane_rows(stream_outputs + (batch_rows * lane_out), row_outputs,
                          batch_rows, lane_out, out_features, lane_out);
    return XST_SUCCESS;
}

/* The PLDDR design routes every MM2S master to PL DDR.  Keep a physical
 * single-lane fallback for projection shapes that cannot use the dual split;
 * the legacy PS-DDR bounce path would otherwise raise AXI DMADecErr. */
static int run_projection_batch_single_plddr_linux(
    const int8_t *activations,
    const uint8_t *weights,
    uint32_t batch_rows,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags,
    int32_t *stream_outputs,
    int32_t *row_outputs) {
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint32_t tx_words = bitnet_accel_gemm_stream_words(
        out_features, in_features, batch_rows);
    int status;

    if (tx_words == 0u || tx_words > tx_capacity_words ||
        bitnet_plddr_status() == 0u) {
        return BITNET_DUAL_UNAVAILABLE;
    }
    status = bitnet_accel_build_gemm_stream(
        (uint32_t *)REAL_TX_BASE, tx_capacity_words, activations, weights,
        batch_rows, out_features, in_features, flags);
    if (status != XST_SUCCESS) {
        return status;
    }
    status = bitnet_plddr_copy_to(
        XPAR_PL_DDR4_0_BASEADDR, (const void *)REAL_TX_BASE,
        (size_t)tx_words * sizeof(uint32_t));
    if (status != XST_SUCCESS) {
        return status;
    }
    status = bitnet_accel_run_physical(
        &AxiDma, XPAR_PL_DDR4_0_BASEADDR, tx_words,
        stream_outputs, batch_rows * out_features);
    if (status != XST_SUCCESS) {
        return status;
    }
    unpack_gemm_rows(stream_outputs, row_outputs, batch_rows, out_features);
    return XST_SUCCESS;
}
#endif

static int run_projection_batch_fast(uint32_t layer_index, uint32_t tensor_id,
                                     const int8_t *activations,
                                     const uint8_t *weights,
                                     uint32_t batch_rows,
                                     uint32_t out_features,
                                     uint32_t in_features,
                                     int32_t *stream_outputs,
                                     int32_t *row_outputs,
                                     uint32_t flags,
                                     InferPerf *perf) {
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint64_t accel_t0;
    uint64_t accel_t1;
    uint64_t projection_t1;
    int status;
#if defined(BITNET_LINUX)
    int dual_unpacked = 0;
#endif

    if ((batch_rows == 0u) || (batch_rows > PREFILL_BATCH_MAX) ||
        (out_features > PREFILL_PROJECTION_MAX_OUT) ||
        (stream_outputs == 0) || (row_outputs == 0)) {
        return XST_INVALID_PARAM;
    }

    smoke_status->reserved0 = tensor_id;
    smoke_status->reserved1 = layer_index + 1u;
    smoke_status_flush();

    accel_t0 = perf_now_cycles();
#if defined(BITNET_LINUX)
    status = run_projection_batch_dual_linux(
        layer_index, tensor_id, activations, weights, batch_rows, out_features,
        in_features, flags, stream_outputs, row_outputs);
    if (status == XST_SUCCESS) {
        dual_unpacked = 1;
    }
    if (status == BITNET_DUAL_UNAVAILABLE) {
        status = run_projection_batch_single_plddr_linux(
            activations, weights, batch_rows, out_features, in_features,
            flags, stream_outputs, row_outputs);
        if (status == BITNET_DUAL_UNAVAILABLE) {
            status = bitnet_accel_run_gemm_chunked_direct(
                &AxiDma, (uint32_t *)REAL_TX_BASE, tx_capacity_words,
                activations, weights, batch_rows, out_features, in_features,
                REAL_CHUNK_OUT, flags, stream_outputs, 0u);
        }
    }
#else
    status = bitnet_accel_run_gemm_chunked_direct(
        &AxiDma, (uint32_t *)REAL_TX_BASE, tx_capacity_words,
        activations, weights, batch_rows, out_features, in_features,
        REAL_CHUNK_OUT, flags, stream_outputs, 0u);
#endif
    accel_t1 = perf_now_cycles();
    perf->accel_cycles += (accel_t1 - accel_t0);
    if (status != XST_SUCCESS) {
        smoke_status_update(SMOKE_STAGE_INFER_RUN, SMOKE_RESULT_FAIL, status);
        return status;
    }

#if defined(BITNET_LINUX)
    if (!dual_unpacked) {
        unpack_gemm_rows(stream_outputs, row_outputs, batch_rows, out_features);
    }
#else
    unpack_gemm_rows(stream_outputs, row_outputs, batch_rows, out_features);
#endif
    projection_t1 = perf_now_cycles();
    if ((tensor_id >= LAYER_TENSOR_Q) && (tensor_id <= LAYER_TENSOR_O)) {
        perf->attention_projection_cycles += (projection_t1 - accel_t0);
    } else if ((tensor_id >= LAYER_TENSOR_GATE) &&
               (tensor_id <= LAYER_TENSOR_DOWN)) {
        perf->mlp_projection_cycles += (projection_t1 - accel_t0);
    }
    return XST_SUCCESS;
}

#if defined(BITNET_LINUX)
static int run_projection_dual_linux(uint32_t layer_index,
                                     uint32_t tensor_id,
                                     const int8_t *activations,
                                     const uint8_t *weights,
                                     uint32_t out_features,
                                     uint32_t in_features,
                                     uint32_t flags,
                                     int32_t *outputs) {
    const uint32_t lane_out = out_features / 2u;
    const uint32_t lane_capacity_words =
        (REAL_TX_BYTES / 2u) / sizeof(uint32_t);
    uint32_t *lane0_stream = (uint32_t *)REAL_TX_BASE;
    uint32_t *lane1_stream =
        (uint32_t *)(REAL_TX_BASE + (REAL_TX_BYTES / 2u));
    uint32_t lane0_words;
    uint32_t lane1_words;
    uint32_t lane1_weight_offset;
    uint32_t prefix_words;
    PlddrPacketCacheEntry *cache_entry;
    int status;

    /* Keep the legacy path for odd channel partitions and unsupported model
     * shapes.  The normal BitNet projections are divisible by eight. */
    if (!dual_lane_ready || (out_features < 8u) ||
        ((out_features & 0x7u) != 0u) || (in_features == 0u) ||
        (outputs == 0)) {
        return BITNET_DUAL_UNAVAILABLE;
    }

    lane1_weight_offset = (lane_out / 4u) * in_features;
    lane0_words = bitnet_accel_stream_words(lane_out, in_features);
    lane1_words = lane0_words;
    if (lane0_words == 0u || lane0_words > lane_capacity_words) {
        return BITNET_DUAL_UNAVAILABLE;
    }

    if (layer_index >= MODEL_TOTAL_LAYERS || tensor_id >= 8u) {
        return XST_INVALID_PARAM;
    }
    cache_entry = &plddr_packet_cache[layer_index][tensor_id];
    if (cache_entry->valid != 0u &&
        cache_entry->batch_rows == 1u &&
        cache_entry->out_features == out_features &&
        cache_entry->in_features == in_features &&
        cache_entry->lane_words == lane0_words) {
        status = bitnet_accel_build_prefix(
            lane0_stream, lane_capacity_words, activations,
            lane_out, in_features, flags, &prefix_words);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = packet_cache_copy_to(
            cache_entry->lane0_addr, lane0_stream,
            (size_t)prefix_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            return status;
        }
        status = packet_cache_copy_to(
            cache_entry->lane1_addr, lane0_stream,
            (size_t)prefix_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            return status;
        }
    } else {
        status = bitnet_accel_build_stream(
            lane0_stream, lane_capacity_words, activations, weights,
            lane_out, in_features, flags);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = bitnet_accel_build_stream(
            lane1_stream, lane_capacity_words, activations,
            weights + lane1_weight_offset, lane_out, in_features, flags);
        if (status != XST_SUCCESS) {
            return status;
        }
        status = plddr_packet_cache_alloc(
            lane0_words * sizeof(uint32_t),
            &cache_entry->lane0_addr, &cache_entry->lane1_addr);
        if (status != XST_SUCCESS) {
            cache_entry->valid = 0u;
            return BITNET_DUAL_UNAVAILABLE;
        }
        status = packet_cache_copy_to(
            cache_entry->lane0_addr, lane0_stream,
            (size_t)lane0_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            cache_entry->valid = 0u;
            return status;
        }
        status = packet_cache_copy_to(
            cache_entry->lane1_addr, lane1_stream,
            (size_t)lane1_words * sizeof(uint32_t));
        if (status != XST_SUCCESS) {
            cache_entry->valid = 0u;
            return status;
        }
        cache_entry->batch_rows = 1u;
        cache_entry->out_features = out_features;
        cache_entry->in_features = in_features;
        cache_entry->lane_words = lane0_words;
        cache_entry->valid = 1u;
    }

    return bitnet_accel_run_dual_physical(
        &AxiDma, &AxiDmaLane1,
        cache_entry->lane0_addr, lane0_words,
        cache_entry->lane1_addr, lane1_words,
        outputs, out_features);
}

static int run_projection_single_plddr_linux(
    const int8_t *activations,
    const uint8_t *weights,
    uint32_t out_features,
    uint32_t in_features,
    uint32_t flags,
    int32_t *outputs) {
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint32_t tx_words = bitnet_accel_stream_words(out_features, in_features);
    int status;

    if (tx_words == 0u || tx_words > tx_capacity_words ||
        bitnet_plddr_status() == 0u) {
        return BITNET_DUAL_UNAVAILABLE;
    }
    status = bitnet_accel_build_stream(
        (uint32_t *)REAL_TX_BASE, tx_capacity_words, activations, weights,
        out_features, in_features, flags);
    if (status != XST_SUCCESS) {
        return status;
    }
    status = bitnet_plddr_copy_to(
        XPAR_PL_DDR4_0_BASEADDR, (const void *)REAL_TX_BASE,
        (size_t)tx_words * sizeof(uint32_t));
    if (status != XST_SUCCESS) {
        return status;
    }
    return bitnet_accel_run_physical(
        &AxiDma, XPAR_PL_DDR4_0_BASEADDR, tx_words, outputs, out_features);
}
#endif

static int run_projection_fast(uint32_t layer_index, uint32_t tensor_id,
                               const int8_t *activations, const uint8_t *weights,
                               uint32_t out_features, uint32_t in_features,
                               int32_t *outputs, uint32_t flags,
                               InferPerf *perf) {
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint64_t t0;
    uint64_t t1;
    int status;

    smoke_status->reserved0 = tensor_id;
    smoke_status->reserved1 = layer_index + 1u;
    smoke_status_flush();

    for (uint32_t n = 0u; n < out_features; n++) {
        outputs[n] = 0;
    }

    t0 = perf_now_cycles();
#if defined(BITNET_LINUX)
    status = run_projection_dual_linux(layer_index, tensor_id, activations,
                                       weights, out_features, in_features,
                                       flags, outputs);
    if (status == BITNET_DUAL_UNAVAILABLE) {
        status = run_projection_single_plddr_linux(
            activations, weights, out_features, in_features, flags, outputs);
        if (status == BITNET_DUAL_UNAVAILABLE) {
            status = bitnet_accel_run_gemv_chunked_direct(
                &AxiDma, (uint32_t *)REAL_TX_BASE, tx_capacity_words,
                activations, weights, out_features, in_features,
                REAL_CHUNK_OUT, flags, outputs, 0u);
        }
    }
#else
    status = bitnet_accel_run_gemv_chunked_direct(
        &AxiDma, (uint32_t *)REAL_TX_BASE, tx_capacity_words,
        activations, weights, out_features, in_features,
        REAL_CHUNK_OUT, flags, outputs, 0u);
#endif
    t1 = perf_now_cycles();
    perf->accel_cycles += (t1 - t0);
    if ((tensor_id >= LAYER_TENSOR_Q) && (tensor_id <= LAYER_TENSOR_O)) {
        perf->attention_projection_cycles += (t1 - t0);
    } else if ((tensor_id >= LAYER_TENSOR_GATE) && (tensor_id <= LAYER_TENSOR_DOWN)) {
        perf->mlp_projection_cycles += (t1 - t0);
    }
    if (status != XST_SUCCESS) {
        smoke_status_update(SMOKE_STAGE_INFER_RUN, SMOKE_RESULT_FAIL, status);
        return status;
    }

    return XST_SUCCESS;
}

static int run_layer_chain_step_fast(uint32_t layer_index, uint32_t weight_slot, uint32_t flags,
                                     uint32_t seq_pos,
                                     int8_t *hidden_act, int8_t *att_act, int8_t *ffn_act,
                                     int32_t *q_out, int32_t *k_out, int32_t *v_out,
                                     int32_t *o_out, int32_t *gate_out, int32_t *up_out,
                                     int32_t *down_out, float *hidden_factor,
                                     InferPerf *perf) {
    int8_t *norm_act = (int8_t *)LAYER_Q8_BASE;
    float *numeric_scratch = (float *)LAYER_ATT_FLOAT_BASE;
    float norm_factor;
    float q_factor;
    float k_factor;
    float v_factor;
    float att_factor;
    float gate_factor;
    float up_factor;
    float ffn_factor;
    uint64_t block_t0;
    uint64_t nonlinear_t0;
    uint32_t resident_weight_stream =
        ((weight_slot & LAYER_WEIGHT_REF_RESIDENT) != 0u) ||
        ((weight_slot & LAYER_WEIGHT_REF_DOUBLEBUF) != 0u);
    const int32_t *att_k_out = k_out;
    const int32_t *att_v_out = v_out;
    const int32_t *mlp_up_out = up_out;
    int status;

    smoke_status_update(SMOKE_STAGE_INFER_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    block_t0 = perf_now_cycles();
    nonlinear_t0 = perf_now_cycles();
    norm_factor = bitnet_rmsnorm_i8_dynamic(hidden_act, *hidden_factor, norm_act,
                                           model_aux_input_norm(layer_index),
                                           LAYER_HIDDEN_SIZE, numeric_scratch);
    perf->attention_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;

    if (resident_weight_stream != 0u) {
        status = run_projection_fast(layer_index, LAYER_TENSOR_Q,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_Q_OFFSET),
                                     LAYER_HIDDEN_SIZE + (2u * LAYER_KV_SIZE),
                                     LAYER_HIDDEN_SIZE,
                                     q_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }
        att_k_out = q_out + LAYER_HIDDEN_SIZE;
        att_v_out = att_k_out + LAYER_KV_SIZE;
    } else {
        status = run_projection_fast(layer_index, LAYER_TENSOR_Q,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_Q_OFFSET),
                                     LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE,
                                     q_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }

        status = run_projection_fast(layer_index, LAYER_TENSOR_K,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_K_OFFSET),
                                     LAYER_KV_SIZE, LAYER_HIDDEN_SIZE,
                                     k_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }

        status = run_projection_fast(layer_index, LAYER_TENSOR_V,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_V_OFFSET),
                                     LAYER_KV_SIZE, LAYER_HIDDEN_SIZE,
                                     v_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }
    }
    q_factor = norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_Q);
    k_factor = norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_K);
    v_factor = norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_V);

    status = run_attention_dispatch(layer_index, seq_pos,
                                    q_out, q_factor, att_k_out, k_factor,
                                    att_v_out, v_factor, att_act, &att_factor,
                                    perf);
    if (status != XST_SUCCESS) {
        return status;
    }
    nonlinear_t0 = perf_now_cycles();
    norm_factor = bitnet_rmsnorm_i8_dynamic(att_act, att_factor, norm_act,
                                           model_aux_attn_sub_norm(layer_index),
                                           LAYER_HIDDEN_SIZE, numeric_scratch);
    perf->attention_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;

    status = run_projection_fast(layer_index, LAYER_TENSOR_O,
                                 norm_act,
                                 layer_weight_ptr(weight_slot, LAYER_WEIGHT_O_OFFSET),
                                 LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE,
                                 o_out, flags, perf);
    if (status != XST_SUCCESS) {
        return status;
    }

    residual_add_scaled(hidden_act, hidden_factor, o_out,
                        norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_O),
                        LAYER_HIDDEN_SIZE);
    perf->attention_block_cycles += perf_now_cycles() - block_t0;

    block_t0 = perf_now_cycles();
    nonlinear_t0 = perf_now_cycles();
    norm_factor = bitnet_rmsnorm_i8_dynamic(hidden_act, *hidden_factor, norm_act,
                                           model_aux_post_norm(layer_index),
                                           LAYER_HIDDEN_SIZE, numeric_scratch);
    perf->mlp_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;

    if (resident_weight_stream != 0u) {
        status = run_projection_fast(layer_index, LAYER_TENSOR_GATE,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_GATE_OFFSET),
                                     2u * LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE,
                                     gate_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }
        mlp_up_out = gate_out + LAYER_FFN_SIZE;
    } else {
        status = run_projection_fast(layer_index, LAYER_TENSOR_GATE,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_GATE_OFFSET),
                                     LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE,
                                     gate_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }

        status = run_projection_fast(layer_index, LAYER_TENSOR_UP,
                                     norm_act,
                                     layer_weight_ptr(weight_slot, LAYER_WEIGHT_UP_OFFSET),
                                     LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE,
                                     up_out, flags, perf);
        if (status != XST_SUCCESS) {
            return status;
        }
    }
    gate_factor = norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_GATE);
    up_factor = norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_UP);

    nonlinear_t0 = perf_now_cycles();
    ffn_factor = bitnet_ffn_norm_quantize(
        gate_out, gate_factor, mlp_up_out, up_factor,
        ffn_act, model_aux_ffn_sub_norm(layer_index), LAYER_FFN_SIZE,
        numeric_scratch);
    perf->mlp_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;
    status = run_projection_fast(layer_index, LAYER_TENSOR_DOWN,
                                 ffn_act,
                                 layer_weight_ptr(weight_slot, LAYER_WEIGHT_DOWN_OFFSET),
                                 LAYER_HIDDEN_SIZE, LAYER_FFN_SIZE,
                                 down_out, flags, perf);
    if (status != XST_SUCCESS) {
        return status;
    }

    residual_add_scaled(hidden_act, hidden_factor, down_out,
                        ffn_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_DOWN),
                        LAYER_HIDDEN_SIZE);
    perf->mlp_block_cycles += perf_now_cycles() - block_t0;
    return XST_SUCCESS;
}

static int run_layer_chain_block_fast(uint32_t layer_index,
                                      uint32_t weight_slot,
                                      uint32_t flags,
                                      uint32_t seq_pos_base,
                                      uint32_t batch_rows,
                                      PrefillBatchBuffers *bufs,
                                      InferPerf *perf) {
    const uint32_t qkv_features = LAYER_HIDDEN_SIZE + (2u * LAYER_KV_SIZE);
    const uint32_t gate_up_features = 2u * LAYER_FFN_SIZE;
    uint64_t block_t0;
    uint64_t nonlinear_t0;
    uint32_t resident_weight_stream =
        ((weight_slot & LAYER_WEIGHT_REF_RESIDENT) != 0u) ||
        ((weight_slot & LAYER_WEIGHT_REF_DOUBLEBUF) != 0u);
    int status;

    if ((bufs == 0) || (perf == 0) || (batch_rows == 0u) ||
        (batch_rows > PREFILL_BATCH_MAX) || (resident_weight_stream == 0u) ||
        ((seq_pos_base + batch_rows) > INFER_CONTEXT_MAX_TOKENS)) {
        return XST_INVALID_PARAM;
    }

    smoke_status_update(SMOKE_STAGE_INFER_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    block_t0 = perf_now_cycles();
    nonlinear_t0 = perf_now_cycles();
    for (uint32_t row = 0u; row < batch_rows; row++) {
        const int8_t *hidden = bufs->hidden + ((uintptr_t)row * LAYER_HIDDEN_SIZE);
        int8_t *norm = bufs->norm + ((uintptr_t)row * LAYER_HIDDEN_SIZE);

        bufs->projection_factors[row] = bitnet_rmsnorm_i8_dynamic(
            hidden, bufs->hidden_factors[row], norm,
            model_aux_input_norm(layer_index), LAYER_HIDDEN_SIZE, bufs->scratch);
    }
    perf->attention_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;

    status = run_projection_batch_fast(
        layer_index, LAYER_TENSOR_Q, bufs->norm,
        layer_weight_ptr(weight_slot, LAYER_WEIGHT_Q_OFFSET),
        batch_rows, qkv_features, LAYER_HIDDEN_SIZE,
        bufs->projection_stream, bufs->projection_rows, flags, perf);
    if (status != XST_SUCCESS) {
        return status;
    }

    for (uint32_t row = 0u; row < batch_rows; row++) {
        int32_t *q_out = bufs->projection_rows + ((uintptr_t)row * qkv_features);
        int32_t *k_out = q_out + LAYER_HIDDEN_SIZE;
        int32_t *v_out = k_out + LAYER_KV_SIZE;
        int8_t *att = bufs->attention + ((uintptr_t)row * LAYER_HIDDEN_SIZE);
        int8_t *norm = bufs->norm + ((uintptr_t)row * LAYER_HIDDEN_SIZE);
        float norm_factor = bufs->projection_factors[row];
        float att_factor;

        status = run_attention_dispatch(
            layer_index, seq_pos_base + row,
            q_out, norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_Q),
            k_out, norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_K),
            v_out, norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_V),
            att, &att_factor, perf);
        if (status != XST_SUCCESS) {
            return status;
        }

        nonlinear_t0 = perf_now_cycles();
        bufs->projection_factors[row] = bitnet_rmsnorm_i8_dynamic(
            att, att_factor, norm, model_aux_attn_sub_norm(layer_index),
            LAYER_HIDDEN_SIZE, bufs->scratch);
        perf->attention_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;
    }

    status = run_projection_batch_fast(
        layer_index, LAYER_TENSOR_O, bufs->norm,
        layer_weight_ptr(weight_slot, LAYER_WEIGHT_O_OFFSET),
        batch_rows, LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE,
        bufs->projection_stream, bufs->projection_rows, flags, perf);
    if (status != XST_SUCCESS) {
        return status;
    }
    for (uint32_t row = 0u; row < batch_rows; row++) {
        int8_t *hidden = bufs->hidden + ((uintptr_t)row * LAYER_HIDDEN_SIZE);
        const int32_t *o_out =
            bufs->projection_rows + ((uintptr_t)row * LAYER_HIDDEN_SIZE);

        residual_add_scaled(
            hidden, &bufs->hidden_factors[row], o_out,
            bufs->projection_factors[row] *
                model_aux_weight_scale(layer_index, LAYER_TENSOR_O),
            LAYER_HIDDEN_SIZE);
    }
    perf->attention_block_cycles += perf_now_cycles() - block_t0;

    block_t0 = perf_now_cycles();
    nonlinear_t0 = perf_now_cycles();
    for (uint32_t row = 0u; row < batch_rows; row++) {
        const int8_t *hidden = bufs->hidden + ((uintptr_t)row * LAYER_HIDDEN_SIZE);
        int8_t *norm = bufs->norm + ((uintptr_t)row * LAYER_HIDDEN_SIZE);

        bufs->projection_factors[row] = bitnet_rmsnorm_i8_dynamic(
            hidden, bufs->hidden_factors[row], norm,
            model_aux_post_norm(layer_index), LAYER_HIDDEN_SIZE, bufs->scratch);
    }
    perf->mlp_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;

    status = run_projection_batch_fast(
        layer_index, LAYER_TENSOR_GATE, bufs->norm,
        layer_weight_ptr(weight_slot, LAYER_WEIGHT_GATE_OFFSET),
        batch_rows, gate_up_features, LAYER_HIDDEN_SIZE,
        bufs->projection_stream, bufs->projection_rows, flags, perf);
    if (status != XST_SUCCESS) {
        return status;
    }

    nonlinear_t0 = perf_now_cycles();
    for (uint32_t row = 0u; row < batch_rows; row++) {
        const int32_t *gate =
            bufs->projection_rows + ((uintptr_t)row * gate_up_features);
        const int32_t *up = gate + LAYER_FFN_SIZE;
        int8_t *ffn = bufs->ffn + ((uintptr_t)row * LAYER_FFN_SIZE);
        float norm_factor = bufs->projection_factors[row];

        bufs->projection_factors[row] = bitnet_ffn_norm_quantize(
            gate, norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_GATE),
            up, norm_factor * model_aux_weight_scale(layer_index, LAYER_TENSOR_UP),
            ffn, model_aux_ffn_sub_norm(layer_index), LAYER_FFN_SIZE, bufs->scratch);
    }
    perf->mlp_nonlinear_cycles += perf_now_cycles() - nonlinear_t0;

    status = run_projection_batch_fast(
        layer_index, LAYER_TENSOR_DOWN, bufs->ffn,
        layer_weight_ptr(weight_slot, LAYER_WEIGHT_DOWN_OFFSET),
        batch_rows, LAYER_HIDDEN_SIZE, LAYER_FFN_SIZE,
        bufs->projection_stream, bufs->projection_rows, flags, perf);
    if (status != XST_SUCCESS) {
        return status;
    }
    for (uint32_t row = 0u; row < batch_rows; row++) {
        int8_t *hidden = bufs->hidden + ((uintptr_t)row * LAYER_HIDDEN_SIZE);
        const int32_t *down =
            bufs->projection_rows + ((uintptr_t)row * LAYER_HIDDEN_SIZE);

        residual_add_scaled(
            hidden, &bufs->hidden_factors[row], down,
            bufs->projection_factors[row] *
                model_aux_weight_scale(layer_index, LAYER_TENSOR_DOWN),
            LAYER_HIDDEN_SIZE);
    }
    perf->mlp_block_cycles += perf_now_cycles() - block_t0;
    return XST_SUCCESS;
}

static int run_layer_chain_step(uint32_t layer_index, uint32_t weight_slot, uint32_t flags,
                                int8_t *hidden_act, int8_t *att_act, int8_t *ffn_act,
                                int32_t *q_out, int32_t *k_out, int32_t *v_out,
                                int32_t *o_out, int32_t *gate_out, int32_t *up_out,
                                int32_t *down_out,
                                uint32_t *mismatches, uint32_t *first_mismatch) {
    int status;

    smoke_status_update(SMOKE_STAGE_LAYER_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    status = run_projection_checked(layer_index, LAYER_TENSOR_Q,
                                    hidden_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_Q_OFFSET),
                                    LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE,
                                    q_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = run_projection_checked(layer_index, LAYER_TENSOR_K,
                                    hidden_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_K_OFFSET),
                                    LAYER_KV_SIZE, LAYER_HIDDEN_SIZE,
                                    k_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = run_projection_checked(layer_index, LAYER_TENSOR_V,
                                    hidden_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_V_OFFSET),
                                    LAYER_KV_SIZE, LAYER_HIDDEN_SIZE,
                                    v_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    compress_vector_i32_to_i8(q_out, att_act, LAYER_HIDDEN_SIZE, 256u);
    status = run_projection_checked(layer_index, LAYER_TENSOR_O,
                                    att_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_O_OFFSET),
                                    LAYER_HIDDEN_SIZE, LAYER_HIDDEN_SIZE,
                                    o_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    compress_vector_i32_to_i8(o_out, hidden_act, LAYER_HIDDEN_SIZE, 256u);
    status = run_projection_checked(layer_index, LAYER_TENSOR_GATE,
                                    hidden_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_GATE_OFFSET),
                                    LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE,
                                    gate_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    status = run_projection_checked(layer_index, LAYER_TENSOR_UP,
                                    hidden_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_UP_OFFSET),
                                    LAYER_FFN_SIZE, LAYER_HIDDEN_SIZE,
                                    up_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    make_ffn_activation(gate_out, up_out, ffn_act);
    status = run_projection_checked(layer_index, LAYER_TENSOR_DOWN,
                                    ffn_act,
                                    layer_weight_ptr(weight_slot, LAYER_WEIGHT_DOWN_OFFSET),
                                    LAYER_HIDDEN_SIZE, LAYER_FFN_SIZE,
                                    down_out, flags,
                                    mismatches, first_mismatch);
    if (status != XST_SUCCESS) {
        return status;
    }

    compress_vector_i32_to_i8(down_out, hidden_act, LAYER_HIDDEN_SIZE, 256u);
    return XST_SUCCESS;
}

static int validate_layer_window(uint32_t first_layer, uint32_t num_layers, uint32_t max_layers) {
    if ((num_layers == 0u) || (num_layers > max_layers) ||
        (first_layer >= MODEL_TOTAL_LAYERS) ||
        (num_layers > (MODEL_TOTAL_LAYERS - first_layer))) {
        return XST_INVALID_PARAM;
    }

    return XST_SUCCESS;
}

static uint32_t infer_control_layer_count(const InferControl *ctrl) {
    if (ctrl->num_layers == 0u) {
        return MODEL_TOTAL_LAYERS;
    }
    return ctrl->num_layers;
}

static uint32_t infer_control_new_tokens(const InferControl *ctrl) {
    uint32_t count = ctrl->max_new_tokens;

    if (count == 0u) {
        count = 1u;
    }
    if (count > INFER_MAX_NEW_TOKENS) {
        count = INFER_MAX_NEW_TOKENS;
    }
    return count;
}

static uint32_t infer_control_prompt_addr(const InferControl *ctrl) {
    return (ctrl->prompt_addr == 0u) ? (uint32_t)INFER_PROMPT_BASE : ctrl->prompt_addr;
}

static uint32_t infer_read_uart_line(char *dst, uint32_t max_bytes) {
    uint32_t len = 0u;

    xil_printf("prompt> ");
    while (len < max_bytes) {
        char c = inbyte();

        if ((c == '\r') || (c == '\n')) {
            xil_printf("\r\n");
            break;
        }
        if ((c == '\b') || (c == 0x7F)) {
            if (len != 0u) {
                len--;
                xil_printf("\b \b");
            }
            continue;
        }
        dst[len++] = c;
        outbyte(c);
    }
    dst[len] = '\0';
    return len;
}

static int infer_copy_ddr_text(const InferControl *ctrl, char *dst,
                               uint32_t max_bytes, uint32_t *byte_count) {
    uintptr_t prompt_addr = (uintptr_t)infer_control_prompt_addr(ctrl);
    uint32_t length = ctrl->prompt_length;
    const volatile uint8_t *src = (const volatile uint8_t *)prompt_addr;

    if ((length == 0u) || (length > max_bytes)) {
        smoke_status_update(SMOKE_STAGE_INFER_INPUT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    Xil_DCacheInvalidateRange((UINTPTR)prompt_addr, length);
    for (uint32_t i = 0u; i < length; i++) {
        dst[i] = (char)src[i];
    }
    dst[length] = '\0';
    *byte_count = length;
    return XST_SUCCESS;
}

static int infer_copy_ddr_tokens(const InferControl *ctrl, uint32_t *tokens,
                                 uint32_t max_tokens, uint32_t *token_count) {
    uintptr_t prompt_addr = (uintptr_t)infer_control_prompt_addr(ctrl);
    uint32_t byte_count = ctrl->prompt_length;
    uint32_t count = byte_count / sizeof(uint32_t);
    const volatile uint32_t *src = (const volatile uint32_t *)prompt_addr;

    if ((byte_count == 0u) || ((byte_count & 0x3u) != 0u) || (count > max_tokens)) {
        smoke_status_update(SMOKE_STAGE_INFER_INPUT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    Xil_DCacheInvalidateRange((UINTPTR)prompt_addr, byte_count);
    for (uint32_t i = 0u; i < count; i++) {
        tokens[i] = src[i];
    }
    *token_count = count;
    return XST_SUCCESS;
}

static int infer_tokenize_input(const InferControl *ctrl, uint32_t *tokens,
                                uint32_t max_tokens, uint32_t *token_count) {
    uint32_t byte_count = 0u;
    int status;

    smoke_status_update(SMOKE_STAGE_INFER_INPUT, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    if (ctrl->mode == INFER_MODE_DDR_TOKEN_IDS) {
        return infer_copy_ddr_tokens(ctrl, tokens, max_tokens, token_count);
    }

    if (ctrl->mode == INFER_MODE_UART_TEXT) {
        byte_count = infer_read_uart_line(infer_prompt_text, INFER_PROMPT_MAX_BYTES);
    } else if (ctrl->mode == INFER_MODE_DDR_TEXT) {
        status = infer_copy_ddr_text(ctrl, infer_prompt_text,
                                     INFER_PROMPT_MAX_BYTES, &byte_count);
        if (status != XST_SUCCESS) {
            return status;
        }
    } else {
        smoke_status_update(SMOKE_STAGE_INFER_INPUT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    status = tokenizer_encode_text(infer_prompt_text, byte_count,
                                   tokens, max_tokens, token_count);
    if (status == XST_SUCCESS) {
        xil_printf("PS tokenizer produced %lu token ids\r\n", (unsigned long)*token_count);
    }
    return status;
}

static void infer_make_hidden_from_tokens(int8_t *hidden_act, const uint32_t *tokens,
                                          uint32_t token_count, uint32_t seed) {
    uint32_t state = seed ^ 0x9E3779B9u ^ (token_count * 131u);

    if ((token_count != 0u) &&
        (model_load_embedding_token_i8(tokens[token_count - 1u], hidden_act, 0) == XST_SUCCESS)) {
        xil_printf("Embedding row loaded for token_id=%lu\r\n",
                   (unsigned long)tokens[token_count - 1u]);
        return;
    }

    for (uint32_t i = 0u; i < token_count; i++) {
        state ^= tokens[i] + 0x9E3779B9u + (state << 6) + (state >> 2);
    }

    for (uint32_t i = 0u; i < LAYER_HIDDEN_SIZE; i++) {
        int32_t value;

        state = (state * 1664525u) + 1013904223u + (i * 17u);
        value = (int32_t)((state >> 24) % 15u) - 7;
        if (value == 0) {
            value = 1;
        }
        hidden_act[i] = (int8_t)value;
    }
}

static uint32_t infer_pseudo_sample_token(const int8_t *hidden_act, uint32_t token_count,
                                          uint32_t temperature_milli, uint32_t top_k,
                                          uint32_t *pseudo_score) {
    uint32_t score = 2166136261u ^ token_count ^ (temperature_milli << 1) ^ (top_k << 9);

    for (uint32_t i = 0u; i < LAYER_HIDDEN_SIZE; i++) {
        score ^= (uint8_t)hidden_act[i];
        score *= 16777619u;
    }

    *pseudo_score = score;
    return score % MODEL_VOCAB_SIZE;
}

static void infer_output_text_reset(void) {
    volatile InferOutputText *out = (volatile InferOutputText *)INFER_OUTPUT_TEXT_BASE;

    out->magic = INFER_OUTPUT_TEXT_MAGIC;
    out->version = INFER_OUTPUT_TEXT_VERSION;
    out->byte_count = 0u;
    out->token_count = 0u;
    Xil_DCacheFlushRange((UINTPTR)INFER_OUTPUT_TEXT_BASE, sizeof(InferOutputText));
}

static int tokenizer_decode_token_bytes(uint32_t token_id, const uint8_t **bytes,
                                        uint32_t *byte_count, uint32_t *flags) {
    const TokenizerDecodeEntry *entry;

    if ((bytes == 0) || (byte_count == 0) || (flags == 0) ||
        (token_id >= MODEL_VOCAB_SIZE)) {
        return XST_INVALID_PARAM;
    }

    if (tokenizer_decode_load_from_emmc() != XST_SUCCESS) {
        return XST_FAILURE;
    }

    entry = &tokenizer_decode_entries[token_id];
    if ((entry->byte_count == 0u) ||
        (entry->byte_offset < tokenizer_decode_header->payload_offset) ||
        ((uint32_t)entry->byte_offset + (uint32_t)entry->byte_count >
         tokenizer_decode_header->total_bytes)) {
        return XST_FAILURE;
    }

    *bytes = (const uint8_t *)(TOKENIZER_DECODE_BASE + entry->byte_offset);
    *byte_count = (uint32_t)entry->byte_count;
    *flags = (uint32_t)entry->flags;
    return XST_SUCCESS;
}

static void infer_output_append_raw(const uint8_t *bytes, uint32_t byte_count) {
    volatile InferOutputText *out = (volatile InferOutputText *)INFER_OUTPUT_TEXT_BASE;
    uint32_t available;
    uint32_t copy_bytes;

    if ((bytes == 0) || (byte_count == 0u) ||
        (out->magic != INFER_OUTPUT_TEXT_MAGIC) ||
        (out->version != INFER_OUTPUT_TEXT_VERSION)) {
        return;
    }
    if (out->byte_count >= INFER_OUTPUT_TEXT_MAX_BYTES) {
        return;
    }

    available = INFER_OUTPUT_TEXT_MAX_BYTES - out->byte_count;
    copy_bytes = (byte_count > available) ? available : byte_count;
    for (uint32_t i = 0u; i < copy_bytes; i++) {
        out->bytes[out->byte_count + i] = bytes[i];
    }
    out->byte_count += copy_bytes;
    Xil_DCacheFlushRange((UINTPTR)INFER_OUTPUT_TEXT_BASE, sizeof(InferOutputText));
}

static void infer_output_append_token(uint32_t token_id, uint32_t sample_index) {
    const uint8_t *bytes = 0;
    uint32_t byte_count = 0u;
    uint32_t flags = 0u;
    volatile InferOutputText *out = (volatile InferOutputText *)INFER_OUTPUT_TEXT_BASE;
    int status;

    if ((token_id == TOKENIZER_BOS_ID) || (token_id == TOKENIZER_EOS_ID) ||
        (token_id == TOKENIZER_EOT_ID)) {
        xil_printf("text[%lu] token=%lu <special>\r\n",
                   (unsigned long)sample_index, (unsigned long)token_id);
        return;
    }

    status = tokenizer_decode_token_bytes(token_id, &bytes, &byte_count, &flags);
    if (status != XST_SUCCESS) {
        xil_printf("text[%lu] token=%lu <decode-miss>\r\n",
                   (unsigned long)sample_index, (unsigned long)token_id);
        return;
    }

    infer_output_append_raw(bytes, byte_count);
    out->token_count++;
    Xil_DCacheFlushRange((UINTPTR)INFER_OUTPUT_TEXT_BASE, sizeof(InferOutputText));

    xil_printf("text[%lu] token=%lu bytes=%lu: ",
               (unsigned long)sample_index,
               (unsigned long)token_id,
               (unsigned long)byte_count);
    for (uint32_t i = 0u; i < byte_count; i++) {
        outbyte((char)bytes[i]);
    }
    xil_printf("\r\n");
}

static uint32_t model_embed_rows_in_shard(uint32_t shard) {
    uint32_t row_start = shard * MODEL_EMBED_ROWS_PER_SHARD;

    if (row_start >= MODEL_VOCAB_SIZE) {
        return 0u;
    }
    if ((MODEL_VOCAB_SIZE - row_start) > MODEL_EMBED_ROWS_PER_SHARD) {
        return MODEL_EMBED_ROWS_PER_SHARD;
    }
    return MODEL_VOCAB_SIZE - row_start;
}

static int lm_head_i8_preload_resident(InferPerf *perf) {
    char path[32];
    uint64_t t0;
    uint64_t t1;

    if (lm_head_i8_resident_ready != 0u) {
        return XST_SUCCESS;
    }

    xil_printf("Preloading int8 lm_head to PS DDR: base=0x%08lx bytes=%lu\r\n",
               (unsigned long)LM_HEAD_I8_RESIDENT_BASE,
               (unsigned long)LM_HEAD_I8_TOTAL_BYTES);

    t0 = perf_now_cycles();
    for (uint32_t shard = 0u; shard < MODEL_EMBED_SHARD_COUNT; shard++) {
        uint32_t rows = model_embed_rows_in_shard(shard);
        uint32_t weight_bytes = rows * LM_HEAD_I8_ROW_BYTES;
        uint32_t scale_bytes = rows * sizeof(float);
        uint32_t bytes = weight_bytes + scale_bytes;
        uint32_t row_start = shard * MODEL_EMBED_ROWS_PER_SHARD;
        uintptr_t dst = LM_HEAD_I8_RESIDENT_BASE +
                        ((uintptr_t)row_start * (uintptr_t)LM_HEAD_I8_ROW_BYTES);
        uintptr_t scale_dst = LM_HEAD_SCALE_RESIDENT_BASE +
                              ((uintptr_t)row_start * sizeof(float));
        int status;

        if ((rows == 0u) || (bytes > LM_HEAD_PACKED_SHARD_MAX_BYTES) ||
            (bytes > LM_HEAD_SHARD_MAX_BYTES)) {
            lm_head_i8_resident_ready = 0u;
            return XST_INVALID_PARAM;
        }

        snprintf(path, sizeof(path), "0:/BITNET/LMH/H%02lu.BIN",
                 (unsigned long)shard);
        xil_printf("Preload %s -> 0x%08lx (%lu bytes)\r\n",
                   path, (unsigned long)dst, (unsigned long)bytes);
        status = emmc_read_exact(path, LM_HEAD_SHARD_BASE, bytes);
        if (status != XST_SUCCESS) {
            t1 = perf_now_cycles();
            if (perf != 0) {
                perf->emmc_cycles += (t1 - t0);
            }
            lm_head_i8_resident_ready = 0u;
            xil_printf("int8 lm_head resident preload skipped: status=%d\r\n", status);
            return status;
        }
        memcpy((void *)dst, (const void *)LM_HEAD_SHARD_BASE, weight_bytes);
        memcpy((void *)scale_dst,
               (const void *)(LM_HEAD_SHARD_BASE + weight_bytes), scale_bytes);
    }

    Xil_DCacheFlushRange((UINTPTR)LM_HEAD_I8_RESIDENT_BASE, LM_HEAD_I8_TOTAL_BYTES);
    Xil_DCacheFlushRange((UINTPTR)LM_HEAD_SCALE_RESIDENT_BASE, LM_HEAD_SCALE_TOTAL_BYTES);
    t1 = perf_now_cycles();
    if (perf != 0) {
        perf->emmc_cycles += (t1 - t0);
    }
    lm_head_i8_resident_ready = 1u;
    xil_printf("int8 lm_head resident preload done: shards=%lu bytes=%lu\r\n",
               (unsigned long)MODEL_EMBED_SHARD_COUNT,
               (unsigned long)LM_HEAD_I8_TOTAL_BYTES);
    return XST_SUCCESS;
}

static uint32_t infer_top_k_limit(uint32_t top_k) {
    if (top_k == 0u) {
        return 1u;
    }
    if (top_k > INFER_TOPK_MAX) {
        return INFER_TOPK_MAX;
    }
    return top_k;
}

static void lm_head_topk_reset(uint32_t top_k) {
    for (uint32_t i = 0u; i < top_k; i++) {
        lm_head_top_tokens[i] = 0u;
        lm_head_top_scores[i] = (int32_t)0x80000000u;
        lm_head_top_scores_float[i] = -3.402823466e+38f;
    }
}

static void lm_head_topk_insert_float(uint32_t token_id, float score, uint32_t top_k) {
    uint32_t insert_at = top_k;

    for (uint32_t i = 0u; i < top_k; i++) {
        if (score > lm_head_top_scores_float[i]) {
            insert_at = i;
            break;
        }
    }
    if (insert_at >= top_k) {
        return;
    }
    for (uint32_t i = top_k - 1u; i > insert_at; i--) {
        lm_head_top_scores_float[i] = lm_head_top_scores_float[i - 1u];
        lm_head_top_scores[i] = lm_head_top_scores[i - 1u];
        lm_head_top_tokens[i] = lm_head_top_tokens[i - 1u];
    }
    lm_head_top_scores_float[insert_at] = score;
    lm_head_top_scores[insert_at] = (int32_t)(score * 256.0f);
    lm_head_top_tokens[insert_at] = token_id;
}

static uint32_t lm_head_sample_rank(const int8_t *final_act, uint32_t token_count,
                                    uint32_t temperature_milli, uint32_t top_k) {
    uint32_t hash = 2166136261u ^ token_count ^ (temperature_milli << 7) ^ top_k;

    if ((temperature_milli == 0u) || (top_k <= 1u)) {
        return 0u;
    }

    for (uint32_t i = 0u; i < LAYER_HIDDEN_SIZE; i++) {
        hash ^= (uint8_t)final_act[i];
        hash *= 16777619u;
    }
    return hash % top_k;
}

static int lm_head_load_embedding_shard_to_ddr(uint32_t shard, uint32_t rows,
                                               InferPerf *perf) {
    char path[32];
    uint32_t bytes = rows * MODEL_EMBED_ROW_BYTES;
    uint64_t t0;
    uint64_t t1;
    int status;

    if ((rows == 0u) || (bytes > LM_HEAD_SHARD_MAX_BYTES) ||
        (bytes > MODEL_EMBED_SHARD_MAX_BYTES)) {
        return XST_INVALID_PARAM;
    }

    snprintf(path, sizeof(path), "0:/BITNET/EMB/E%02lu.BIN", (unsigned long)shard);
    t0 = perf_now_cycles();
    status = emmc_read_exact(path, LM_HEAD_SHARD_BASE, bytes);
    t1 = perf_now_cycles();
    if (perf != 0) {
        perf->emmc_cycles += (t1 - t0);
    }
    if (status != XST_SUCCESS) {
        return status;
    }
    Xil_DCacheInvalidateRange((UINTPTR)LM_HEAD_SHARD_BASE, bytes);
    return XST_SUCCESS;
}

static int infer_tied_lm_head_sample_token(const int8_t *hidden_act, float hidden_factor,
                                           uint32_t token_count,
                                           uint32_t temperature_milli, uint32_t top_k,
                                           uint32_t *sampled_token, uint32_t *score_word,
                                           InferPerf *perf) {
    int8_t *final_act = (int8_t *)LAYER_Q8_BASE;
    uint32_t active_top_k = infer_top_k_limit(top_k);
    uint32_t selected_rank;
    uint64_t t0;
    uint64_t t1;

    if ((model_aux_available == 0u) || (sampled_token == 0) || (score_word == 0)) {
        return XST_FAILURE;
    }

    t0 = perf_now_cycles();
    (void)bitnet_rmsnorm_i8_dynamic(hidden_act, hidden_factor, final_act,
                                    model_aux_final_norm(), LAYER_HIDDEN_SIZE,
                                    (float *)LAYER_ATT_FLOAT_BASE);
    lm_head_topk_reset(active_top_k);

    if (lm_head_i8_resident_ready != 0u) {
        if (bitnet_lm_head_topk_i8(
                final_act,
                (const int8_t *)LM_HEAD_I8_RESIDENT_BASE,
                (const float *)LM_HEAD_SCALE_RESIDENT_BASE,
                MODEL_VOCAB_SIZE, LAYER_HIDDEN_SIZE, active_top_k,
                lm_head_top_tokens, lm_head_top_scores_float) != 0) {
            return XST_FAILURE;
        }
        for (uint32_t i = 0u; i < active_top_k; i++) {
            lm_head_top_scores[i] = (int32_t)(lm_head_top_scores_float[i] * 256.0f);
        }
    } else {
        for (uint32_t shard = 0u; shard < MODEL_EMBED_SHARD_COUNT; shard++) {
            uint32_t rows = model_embed_rows_in_shard(shard);
            int status;

            status = lm_head_load_embedding_shard_to_ddr(shard, rows, perf);
            if (status != XST_SUCCESS) {
                t1 = perf_now_cycles();
                if (perf != 0) {
                    perf->lm_head_cycles += (t1 - t0);
                }
                return status;
            }

            for (uint32_t row = 0u; row < rows; row++) {
                uint32_t token_id = (shard * MODEL_EMBED_ROWS_PER_SHARD) + row;
                const uint16_t *row_bf16 =
                    (const uint16_t *)(LM_HEAD_SHARD_BASE +
                                       ((uintptr_t)row * MODEL_EMBED_ROW_BYTES));
                float score = 0.0f;

                for (uint32_t i = 0u; i < LAYER_HIDDEN_SIZE; i++) {
                    score += ((float)final_act[i]) * bf16_to_float(row_bf16[i]);
                }
                lm_head_topk_insert_float(token_id, score, active_top_k);
            }
        }
    }

    selected_rank = lm_head_sample_rank(final_act, token_count, temperature_milli, active_top_k);
    t1 = perf_now_cycles();
    if (perf != 0) {
        perf->lm_head_cycles += (t1 - t0);
    }

    *sampled_token = lm_head_top_tokens[selected_rank];
    *score_word = (uint32_t)lm_head_top_scores[selected_rank];
    return XST_SUCCESS;
}

static int run_layer_chain_test(const LayerChainControl *ctrl) {
    int8_t *hidden_act = (int8_t *)LAYER_HIDDEN_ACT_BASE;
    int8_t *att_act = (int8_t *)LAYER_ATT_ACT_BASE;
    int8_t *ffn_act = (int8_t *)LAYER_FFN_ACT_BASE;
    int32_t *q_out = (int32_t *)LAYER_Q_OUT_BASE;
    int32_t *k_out = (int32_t *)LAYER_K_OUT_BASE;
    int32_t *v_out = (int32_t *)LAYER_V_OUT_BASE;
    int32_t *o_out = (int32_t *)LAYER_O_OUT_BASE;
    int32_t *gate_out = (int32_t *)LAYER_GATE_OUT_BASE;
    int32_t *up_out = (int32_t *)LAYER_UP_OUT_BASE;
    int32_t *down_out = (int32_t *)LAYER_DOWN_OUT_BASE;
    uint32_t mismatches = 0u;
    uint32_t first_mismatch = 0u;
    int status;

    smoke_status_set_dims(LAYER_HIDDEN_SIZE, ctrl->num_layers);
    smoke_status_update(SMOKE_STAGE_LAYER_DETECT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    status = validate_layer_window(ctrl->first_layer, ctrl->num_layers, LAYER_MAX_PRELOAD_LAYERS);
    if (status != XST_SUCCESS) {
        smoke_status_update(SMOKE_STAGE_LAYER_DETECT, SMOKE_RESULT_FAIL, status);
        return status;
    }

    smoke_status->tx_words = layer_chain_total_words(ctrl->num_layers);
    smoke_status->reserved0 = REAL_CHUNK_OUT;
    smoke_status->reserved1 = ctrl->num_layers;
    smoke_status_flush();

    status = init_dma(SMOKE_STAGE_DMA_INIT);
    if (status != XST_SUCCESS) {
        return status;
    }
#if defined(BITNET_LINUX)
    plddr_packet_cache_reset();
#endif
    smoke_status_update(SMOKE_STAGE_DMA_INIT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    make_real_activation(hidden_act, LAYER_HIDDEN_SIZE, ctrl->activation_seed);

    for (uint32_t layer_slot = 0u; layer_slot < ctrl->num_layers; layer_slot++) {
        uint32_t layer_index = ctrl->first_layer + layer_slot;

        status = run_layer_chain_step(layer_index, layer_slot, ctrl->flags,
                                      hidden_act, att_act, ffn_act,
                                      q_out, k_out, v_out, o_out, gate_out, up_out, down_out,
                                      &mismatches, &first_mismatch);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    smoke_status_update(SMOKE_STAGE_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    return XST_SUCCESS;
}

static int run_emmc_layer_chain_test(const EmmcLayerChainControl *ctrl) {
    int8_t *hidden_act = (int8_t *)LAYER_HIDDEN_ACT_BASE;
    int8_t *att_act = (int8_t *)LAYER_ATT_ACT_BASE;
    int8_t *ffn_act = (int8_t *)LAYER_FFN_ACT_BASE;
    int32_t *q_out = (int32_t *)LAYER_Q_OUT_BASE;
    int32_t *k_out = (int32_t *)LAYER_K_OUT_BASE;
    int32_t *v_out = (int32_t *)LAYER_V_OUT_BASE;
    int32_t *o_out = (int32_t *)LAYER_O_OUT_BASE;
    int32_t *gate_out = (int32_t *)LAYER_GATE_OUT_BASE;
    int32_t *up_out = (int32_t *)LAYER_UP_OUT_BASE;
    int32_t *down_out = (int32_t *)LAYER_DOWN_OUT_BASE;
    uint32_t mismatches = 0u;
    uint32_t first_mismatch = 0u;
    int status;

    smoke_status_set_dims(LAYER_HIDDEN_SIZE, ctrl->num_layers);
    smoke_status_update(SMOKE_STAGE_EMMC_DETECT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    status = validate_layer_window(ctrl->first_layer, ctrl->num_layers, EMMC_MAX_LAYERS);
    if (status != XST_SUCCESS) {
        smoke_status_update(SMOKE_STAGE_EMMC_DETECT, SMOKE_RESULT_FAIL, status);
        return status;
    }

    smoke_status->tx_words = layer_chain_total_words(ctrl->num_layers);
    smoke_status->reserved0 = REAL_CHUNK_OUT;
    smoke_status->reserved1 = ctrl->num_layers;
    smoke_status_flush();

    status = emmc_mount();
    if (status != XST_SUCCESS) {
        return status;
    }

    status = init_dma(SMOKE_STAGE_DMA_INIT);
    if (status != XST_SUCCESS) {
        return status;
    }
    smoke_status_update(SMOKE_STAGE_DMA_INIT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    make_real_activation(hidden_act, LAYER_HIDDEN_SIZE, ctrl->activation_seed);

    for (uint32_t layer_offset = 0u; layer_offset < ctrl->num_layers; layer_offset++) {
        uint32_t layer_index = ctrl->first_layer + layer_offset;

        status = emmc_load_layer_weights(layer_index, 0u);
        if (status != XST_SUCCESS) {
            return status;
        }

        smoke_status_update(SMOKE_STAGE_EMMC_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);
        status = run_layer_chain_step(layer_index, 0u, ctrl->flags,
                                      hidden_act, att_act, ffn_act,
                                      q_out, k_out, v_out, o_out, gate_out, up_out, down_out,
                                      &mismatches, &first_mismatch);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    smoke_status_update(SMOKE_STAGE_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    return XST_SUCCESS;
}

static int run_infer_transformer_token_streaming(const InferControl *ctrl,
                                                 const uint32_t *tokens,
                                                 uint32_t fallback_token_count,
                                                 uint32_t token_id,
                                                 uint32_t seq_pos,
                                                 uint32_t num_layers,
                                                 InferBuffers *bufs,
                                                 InferPerf *perf);

static int run_infer_transformer_token(const InferControl *ctrl, const uint32_t *tokens,
                                       uint32_t fallback_token_count, uint32_t token_id,
                                       uint32_t seq_pos, uint32_t num_layers,
                                       InferBuffers *bufs, InferPerf *perf) {
    int status;

    if (infer_layer_streaming != 0u) {
        return run_infer_transformer_token_streaming(
            ctrl, tokens, fallback_token_count, token_id, seq_pos,
            num_layers, bufs, perf);
    }

    if ((seq_pos >= INFER_CONTEXT_MAX_TOKENS) || (bufs == 0)) {
        smoke_status_update(SMOKE_STAGE_INFER_RUN, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    status = model_load_embedding_token_i8(token_id, bufs->hidden_act,
                                           &bufs->hidden_factor);
    if (status != XST_SUCCESS) {
        infer_make_hidden_from_tokens(bufs->hidden_act, tokens, fallback_token_count,
                                      ctrl->activation_seed + seq_pos);
        bufs->hidden_factor = 1.0f;
    }

    for (uint32_t layer_offset = 0u; layer_offset < num_layers; layer_offset++) {
        uint32_t layer_index = ctrl->first_layer + layer_offset;
        uint32_t weight_ref = LAYER_WEIGHT_REF_RESIDENT | layer_index;
        uint64_t layer_t0;

        if (infer_resident_valid[layer_index] == 0u) {
            smoke_status_update(SMOKE_STAGE_INFER_LOAD, SMOKE_RESULT_FAIL, XST_FAILURE);
            return XST_FAILURE;
        }
        perf->cache_hits++;

        layer_t0 = perf_now_cycles();
        status = run_layer_chain_step_fast(layer_index, weight_ref, ctrl->flags,
                                           seq_pos, bufs->hidden_act, bufs->att_act,
                                           bufs->ffn_act, bufs->q_out, bufs->k_out,
                                           bufs->v_out, bufs->o_out, bufs->gate_out,
                                           bufs->up_out, bufs->down_out,
                                           &bufs->hidden_factor, perf);
        perf->layer_forward_cycles += perf_now_cycles() - layer_t0;
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    return XST_SUCCESS;
}

static int run_infer_transformer_token_streaming(const InferControl *ctrl,
                                                 const uint32_t *tokens,
                                                 uint32_t fallback_token_count,
                                                 uint32_t token_id,
                                                 uint32_t seq_pos,
                                                 uint32_t num_layers,
                                                 InferBuffers *bufs,
                                                 InferPerf *perf) {
    LayerPrefetchJob jobs[2];
    uint32_t current_slot = 0u;
    int status;

    if ((num_layers == 0u) || (num_layers > MODEL_TOTAL_LAYERS)) {
        return XST_INVALID_PARAM;
    }
    status = model_load_embedding_token_i8(token_id, bufs->hidden_act,
                                           &bufs->hidden_factor);
    if (status != XST_SUCCESS) {
        infer_make_hidden_from_tokens(bufs->hidden_act, tokens, fallback_token_count,
                                      ctrl->activation_seed + seq_pos);
        bufs->hidden_factor = 1.0f;
    }

#if defined(BITNET_LINUX)
    /*
     * The first streaming decode pass populates one batch-1 packet per
     * layer/tensor in PL-DDR.  On later tokens the raw PS-DDR slots are no
     * longer needed: run_layer_chain_step_fast() rebuilds only the tiny
     * activation prefix on a cache hit.  This removes a full layer's eMMC
     * transfer from every subsequent decode token while preserving the
     * two-slot prefetch pipeline for cache warm-up.
     */
    if (plddr_decode_cache_all_ready(ctrl->first_layer, num_layers)) {
        for (uint32_t layer_offset = 0u; layer_offset < num_layers;
             layer_offset++) {
            uint32_t layer_index = ctrl->first_layer + layer_offset;
            uint32_t weight_ref = LAYER_WEIGHT_REF_DOUBLEBUF;
            uint64_t layer_t0 = perf_now_cycles();

            status = run_layer_chain_step_fast(
                layer_index, weight_ref, ctrl->flags, seq_pos,
                bufs->hidden_act, bufs->att_act, bufs->ffn_act,
                bufs->q_out, bufs->k_out, bufs->v_out, bufs->o_out,
                bufs->gate_out, bufs->up_out, bufs->down_out,
                &bufs->hidden_factor, perf);
            perf->layer_forward_cycles += perf_now_cycles() - layer_t0;
            if (status != XST_SUCCESS) {
                return status;
            }
        }
        return XST_SUCCESS;
    }
#endif

    /* Fill layer 0 synchronously, then keep the other slot busy with L+1. */
    status = emmc_load_layer_weights_tight(ctrl->first_layer, current_slot, 0);
    if (status != XST_SUCCESS) {
        return status;
    }
    memset(jobs, 0, sizeof(jobs));
    if (num_layers > 1u) {
        status = layer_prefetch_start(&jobs[1], ctrl->first_layer + 1u, 1u);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    for (uint32_t layer_offset = 0u; layer_offset < num_layers; layer_offset++) {
        uint32_t layer_index = ctrl->first_layer + layer_offset;
        uint32_t weight_ref = LAYER_WEIGHT_REF_DOUBLEBUF | current_slot;
        uint64_t layer_t0 = perf_now_cycles();

        if (layer_offset != 0u) {
            status = layer_prefetch_wait(&jobs[current_slot], perf);
            if (status != XST_SUCCESS) {
                return status;
            }
        }
        status = run_layer_chain_step_fast(layer_index, weight_ref, ctrl->flags,
                                           seq_pos, bufs->hidden_act, bufs->att_act,
                                           bufs->ffn_act, bufs->q_out, bufs->k_out,
                                           bufs->v_out, bufs->o_out, bufs->gate_out,
                                           bufs->up_out, bufs->down_out,
                                           &bufs->hidden_factor, perf);
        perf->layer_forward_cycles += perf_now_cycles() - layer_t0;
        if (status != XST_SUCCESS) {
            return status;
        }

        /* The slot just consumed is free for L+2 while L+1 is computing. */
        if ((layer_offset + 2u) < num_layers) {
            uint32_t refill_slot = current_slot;
            status = layer_prefetch_start(
                &jobs[refill_slot], ctrl->first_layer + layer_offset + 2u,
                refill_slot);
            if (status != XST_SUCCESS) {
                return status;
            }
        }
        current_slot ^= 1u;
    }
    return XST_SUCCESS;
}

static int run_layer_chain_block_streaming(const InferControl *ctrl,
                                           uint32_t seq_pos_base,
                                           uint32_t batch_rows,
                                           uint32_t num_layers,
                                           PrefillBatchBuffers *bufs,
                                           InferPerf *perf) {
    LayerPrefetchJob jobs[2];
    uint32_t current_slot = 0u;
    int status;

    if ((ctrl == 0) || (bufs == 0) || (perf == 0) ||
        (batch_rows == 0u) || (batch_rows > PREFILL_BATCH_MAX) ||
        (num_layers == 0u)) {
        return XST_INVALID_PARAM;
    }
    status = emmc_load_layer_weights_tight(ctrl->first_layer, current_slot, 0);
    if (status != XST_SUCCESS) {
        return status;
    }
    memset(jobs, 0, sizeof(jobs));
    if (num_layers > 1u) {
        status = layer_prefetch_start(&jobs[1], ctrl->first_layer + 1u, 1u);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    for (uint32_t layer_offset = 0u; layer_offset < num_layers; layer_offset++) {
        uint32_t layer_index = ctrl->first_layer + layer_offset;
        uint32_t weight_ref = LAYER_WEIGHT_REF_DOUBLEBUF | current_slot;
        uint64_t layer_t0 = perf_now_cycles();

        if (layer_offset != 0u) {
            status = layer_prefetch_wait(&jobs[current_slot], perf);
            if (status != XST_SUCCESS) {
                return status;
            }
        }
        status = run_layer_chain_block_fast(
            layer_index, weight_ref, ctrl->flags, seq_pos_base, batch_rows,
            bufs, perf);
        perf->layer_forward_cycles += perf_now_cycles() - layer_t0;
        if (status != XST_SUCCESS) {
            return status;
        }
        if ((layer_offset + 2u) < num_layers) {
            uint32_t refill_slot = current_slot;
            status = layer_prefetch_start(
                &jobs[refill_slot], ctrl->first_layer + layer_offset + 2u,
                refill_slot);
            if (status != XST_SUCCESS) {
                return status;
            }
        }
        current_slot ^= 1u;
    }
    return XST_SUCCESS;
}

static int run_prefill_tokens(const InferControl *ctrl, const uint32_t *tokens,
                              uint32_t prompt_token_count, uint32_t num_layers,
                              InferBuffers *bufs, InferPerf *perf) {
    uint32_t configured_batch = ctrl->prefill_batch_rows;
    uint64_t t0;
    uint64_t t1;
    int status = XST_SUCCESS;

    if ((prompt_token_count == 0u) || (bufs == 0) || (perf == 0)) {
        return XST_INVALID_PARAM;
    }
    if (configured_batch == 0u) {
        configured_batch = BITNET_INFER_PREFILL_BATCH_DEFAULT;
    }
    if (configured_batch > PREFILL_BATCH_MAX) {
        return XST_INVALID_PARAM;
    }

    xil_printf("PREFILL_START prompt_tokens=%lu batch_rows=%lu\r\n",
               (unsigned long)prompt_token_count,
               (unsigned long)configured_batch);
    t0 = perf_now_cycles();

    if (configured_batch == 1u) {
        for (uint32_t pos = 0u; pos < prompt_token_count; pos++) {
            status = run_infer_transformer_token(ctrl, tokens, pos + 1u, tokens[pos],
                                                 pos, num_layers, bufs, perf);
            if (status != XST_SUCCESS) {
                break;
            }
        }
    } else {
        PrefillBatchBuffers batch_bufs;

        prefill_batch_buffers_init(&batch_bufs);
        for (uint32_t block_base = 0u;
             block_base < prompt_token_count && status == XST_SUCCESS;
             block_base += configured_batch) {
            uint32_t remaining = prompt_token_count - block_base;
            uint32_t batch_rows =
                (remaining > configured_batch) ? configured_batch : remaining;

#if defined(BITNET_LINUX)
            /* A different tail batch changes the packed packet layout.  Reuse
             * the full-batch cache for every complete block, then evict it
             * once before the tail instead of allocating a second 521 MiB
             * weight image in the 1 GiB PL DDR. */
            if ((block_base != 0u) && (batch_rows != configured_batch)) {
                plddr_packet_cache_reset();
            }
#endif

            for (uint32_t row = 0u; row < batch_rows; row++) {
                uint32_t pos = block_base + row;
                int8_t *hidden =
                    batch_bufs.hidden + ((uintptr_t)row * LAYER_HIDDEN_SIZE);

                status = model_load_embedding_token_i8(
                    tokens[pos], hidden, &batch_bufs.hidden_factors[row]);
                if (status != XST_SUCCESS) {
                    infer_make_hidden_from_tokens(hidden, tokens, pos + 1u,
                                                  ctrl->activation_seed + pos);
                    batch_bufs.hidden_factors[row] = 1.0f;
                    status = XST_SUCCESS;
                }
            }

            if (infer_layer_streaming != 0u) {
                status = run_layer_chain_block_streaming(
                    ctrl, block_base, batch_rows, num_layers, &batch_bufs, perf);
            } else {
                for (uint32_t layer_offset = 0u;
                     layer_offset < num_layers && status == XST_SUCCESS;
                     layer_offset++) {
                    uint32_t layer_index = ctrl->first_layer + layer_offset;
                    uint32_t weight_ref = LAYER_WEIGHT_REF_RESIDENT | layer_index;
                    uint64_t layer_t0;

                    if (infer_resident_valid[layer_index] == 0u) {
                        smoke_status_update(
                            SMOKE_STAGE_INFER_LOAD, SMOKE_RESULT_FAIL, XST_FAILURE);
                        status = XST_FAILURE;
                        break;
                    }
                    perf->cache_hits += batch_rows;
                    layer_t0 = perf_now_cycles();
                    status = run_layer_chain_block_fast(
                        layer_index, weight_ref, ctrl->flags, block_base,
                        batch_rows, &batch_bufs, perf);
                    perf->layer_forward_cycles += perf_now_cycles() - layer_t0;
                }
            }

            if ((status == XST_SUCCESS) &&
                ((block_base + batch_rows) == prompt_token_count)) {
                uint32_t last_row = batch_rows - 1u;

                memcpy(bufs->hidden_act,
                       batch_bufs.hidden +
                           ((uintptr_t)last_row * LAYER_HIDDEN_SIZE),
                       LAYER_HIDDEN_SIZE);
                bufs->hidden_factor = batch_bufs.hidden_factors[last_row];
            }
        }
    }
    t1 = perf_now_cycles();
    perf->prefill_cycles += (t1 - t0);

#if defined(BITNET_LINUX)
    /* Decode uses a one-row packet layout.  Release the prefill cache now so
     * the first decode token can populate its resident packet set and all
     * following tokens only update activation prefixes. */
    plddr_packet_cache_reset();
#endif

    if (status == XST_SUCCESS) {
        xil_printf("PREFILL_DONE context=%lu\r\n", (unsigned long)prompt_token_count);
    }
    return status;
}

static int run_decode_one(const InferControl *ctrl, const uint32_t *tokens,
                          uint32_t token_count, uint32_t seq_pos, uint32_t num_layers,
                          InferBuffers *bufs, InferPerf *perf) {
    uint64_t t0;
    uint64_t t1;
    int status;

    t0 = perf_now_cycles();
    /* Decode always has one row.  Keep this path free of the prefill batch
     * setup, per-layer smoke bookkeeping and redundant argument normalization. */
    if (infer_layer_streaming != 0u) {
        status = run_infer_transformer_token_streaming(
            ctrl, tokens, token_count, tokens[seq_pos], seq_pos,
            num_layers, bufs, perf);
    } else {
        status = model_load_embedding_token_i8(tokens[seq_pos], bufs->hidden_act,
                                                &bufs->hidden_factor);
        if (status != XST_SUCCESS) {
            infer_make_hidden_from_tokens(bufs->hidden_act, tokens, token_count,
                                          ctrl->activation_seed + seq_pos);
            bufs->hidden_factor = 1.0f;
            status = XST_SUCCESS;
        }
        for (uint32_t layer_offset = 0u;
             layer_offset < num_layers && status == XST_SUCCESS;
             layer_offset++) {
            uint32_t layer_index = ctrl->first_layer + layer_offset;
            uint32_t weight_ref = LAYER_WEIGHT_REF_RESIDENT | layer_index;

            if (infer_resident_valid[layer_index] == 0u) {
                status = XST_FAILURE;
                break;
            }
            status = run_layer_chain_step_fast(
                layer_index, weight_ref, ctrl->flags, seq_pos,
                bufs->hidden_act, bufs->att_act, bufs->ffn_act,
                bufs->q_out, bufs->k_out, bufs->v_out, bufs->o_out,
                bufs->gate_out, bufs->up_out, bufs->down_out,
                &bufs->hidden_factor, perf);
        }
    }
    t1 = perf_now_cycles();
    perf->decode_cycles += (t1 - t0);
    return status;
}

static int run_infer_entry(const InferControl *ctrl) {
    int8_t *hidden_act = (int8_t *)LAYER_HIDDEN_ACT_BASE;
    int8_t *att_act = (int8_t *)LAYER_ATT_ACT_BASE;
    int8_t *ffn_act = (int8_t *)LAYER_FFN_ACT_BASE;
    int32_t *q_out = (int32_t *)LAYER_Q_OUT_BASE;
    int32_t *k_out = (int32_t *)LAYER_K_OUT_BASE;
    int32_t *v_out = (int32_t *)LAYER_V_OUT_BASE;
    int32_t *o_out = (int32_t *)LAYER_O_OUT_BASE;
    int32_t *gate_out = (int32_t *)LAYER_GATE_OUT_BASE;
    int32_t *up_out = (int32_t *)LAYER_UP_OUT_BASE;
    int32_t *down_out = (int32_t *)LAYER_DOWN_OUT_BASE;
    uint32_t *tokens = (uint32_t *)INFER_TOKEN_IDS_BASE;
    uint32_t token_count = 0u;
    uint32_t prompt_token_count = 0u;
    uint32_t generated_count = 0u;
    uint32_t last_token = 0u;
    uint32_t pseudo_score = 0u;
    uint32_t num_layers = infer_control_layer_count(ctrl);
    uint32_t max_new_tokens = infer_control_new_tokens(ctrl);
    InferBuffers bufs;
    InferPerf perf;
    uint64_t total_start;
    uint64_t total_end;
    uint64_t decode_e2e_cycles = 0u;
    uint64_t decode_warm_cycles = 0u;
    uint32_t decode_e2e_tokens = 0u;
    uint32_t decode_warm_tokens = 0u;
    int status;

    infer_layer_streaming = (uint8_t)infer_layer_streaming_enabled();

    perf.cache_hits = 0u;
    perf.cache_misses = 0u;
    perf.emmc_cycles = 0u;
    perf.accel_cycles = 0u;
    perf.total_cycles = 0u;
    perf.prefill_cycles = 0u;
    perf.decode_cycles = 0u;
    perf.lm_head_cycles = 0u;
    perf.attention_block_cycles = 0u;
    perf.mlp_block_cycles = 0u;
    perf.attention_projection_cycles = 0u;
    perf.attention_core_cycles = 0u;
    perf.attention_nonlinear_cycles = 0u;
    perf.mlp_projection_cycles = 0u;
    perf.mlp_nonlinear_cycles = 0u;
    perf.layer_forward_cycles = 0u;

    bufs.hidden_act = hidden_act;
    bufs.att_act = att_act;
    bufs.ffn_act = ffn_act;
    bufs.q_out = q_out;
    bufs.k_out = k_out;
    bufs.v_out = v_out;
    bufs.o_out = o_out;
    bufs.gate_out = gate_out;
    bufs.up_out = up_out;
    bufs.down_out = down_out;
    bufs.hidden_factor = 1.0f;

    smoke_status_set_dims(LAYER_HIDDEN_SIZE, num_layers);
    smoke_status_update(SMOKE_STAGE_INFER_DETECT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    if ((ctrl->version != INFER_CONTROL_VERSION) ||
        (validate_layer_window(ctrl->first_layer, num_layers, EMMC_MAX_LAYERS) != XST_SUCCESS)) {
        smoke_status_update(SMOKE_STAGE_INFER_DETECT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    xil_printf("BitNet AXU3EGB PS-side inference entry\r\n");
    xil_printf("Path: UART/DDR text -> PS tokenizer -> PS DDR -> AXI DMA -> PL -> PS DDR\r\n");
    xil_printf("Weight mode: %s\r\n",
               (infer_layer_streaming != 0u) ?
                   "two-slot PS DDR streaming with background prefetch" :
                   "all-layer PS DDR resident map");
    xil_printf("Runtime path: embedding -> RMSNorm/RoPE/KV/GQA softmax -> PL GEMV -> final norm -> full-vocab tied lm_head top-k.\r\n");
    xil_printf("Context limit: model=%lu tokens kv_cache=%lu MiB\r\n",
               (unsigned long)INFER_CONTEXT_MAX_TOKENS,
               (unsigned long)(KV_CACHE_TOTAL_BYTES >> 20));

    status = emmc_mount();
    if (status != XST_SUCCESS) {
        return status;
    }
    (void)model_aux_load_from_emmc();
    (void)model_rope_load_from_emmc();
    (void)tokenizer_decode_load_from_emmc();
    infer_output_text_reset();

    status = infer_tokenize_input(ctrl, tokens,
                                  INFER_CONTEXT_MAX_TOKENS,
                                  &token_count);
    if (status != XST_SUCCESS) {
        return status;
    }
    if (token_count == 0u) {
        smoke_status_update(SMOKE_STAGE_INFER_TOKENIZE, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }
    prompt_token_count = token_count;
    if (max_new_tokens > (INFER_CONTEXT_MAX_TOKENS - prompt_token_count)) {
        max_new_tokens = INFER_CONTEXT_MAX_TOKENS - prompt_token_count;
    }

    smoke_status->tx_words = layer_chain_total_words(num_layers) *
                             (prompt_token_count + max_new_tokens);
    smoke_status->reserved0 = REAL_CHUNK_OUT;
    smoke_status->reserved1 = num_layers;
    smoke_status_flush();
    xil_printf("Token budget: prompt=%lu max_new=%lu total_limit=%lu\r\n",
               (unsigned long)prompt_token_count,
               (unsigned long)max_new_tokens,
               (unsigned long)INFER_CONTEXT_MAX_TOKENS);

    status = init_dma(SMOKE_STAGE_DMA_INIT);
    if (status != XST_SUCCESS) {
        return status;
    }
#if defined(BITNET_LINUX)
    plddr_packet_cache_reset();
#endif
    smoke_status_update(SMOKE_STAGE_DMA_INIT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    total_start = perf_now_cycles();

    if (infer_layer_streaming == 0u) {
        status = infer_preload_resident_weights(ctrl->first_layer, num_layers, &perf);
        if (status != XST_SUCCESS) {
            return status;
        }
    }

    status = lm_head_i8_preload_resident(&perf);
    if (status != XST_SUCCESS) {
        lm_head_i8_resident_ready = 0u;
        xil_printf("Using BF16 tied lm_head fallback scan.\r\n");
    }

    kv_cache_clear();

    status = run_prefill_tokens(ctrl, tokens, prompt_token_count, num_layers, &bufs, &perf);
    if (status != XST_SUCCESS) {
        return status;
    }

    for (uint32_t gen = 0u; gen < max_new_tokens; gen++) {
        uint64_t decode_token_start = 0u;

        if (gen != 0u) {
            uint32_t seq_pos = token_count - 1u;

            decode_token_start = perf_now_cycles();
            status = run_decode_one(ctrl, tokens, token_count, seq_pos,
                                    num_layers, &bufs, &perf);
            if (status != XST_SUCCESS) {
                return status;
            }
        }

        smoke_status_update(SMOKE_STAGE_INFER_LOGITS, SMOKE_RESULT_RUNNING, XST_SUCCESS);
        status = infer_tied_lm_head_sample_token(hidden_act, bufs.hidden_factor,
                                                 token_count,
                                                 ctrl->temperature_milli,
                                                 ctrl->top_k,
                                                 &last_token,
                                                 &pseudo_score,
                                                 &perf);
        if (status != XST_SUCCESS) {
            last_token = infer_pseudo_sample_token(hidden_act, token_count,
                                                   ctrl->temperature_milli,
                                                   ctrl->top_k,
                                                   &pseudo_score);
        }
        if (gen != 0u) {
            uint64_t decode_token_cycles =
                perf_now_cycles() - decode_token_start;

            decode_e2e_cycles += decode_token_cycles;
            decode_e2e_tokens++;
            /* gen==1 performs the lazy PS-DDR -> PL-DDR packet residency
             * fill.  Excluding it yields the requested steady-state decode
             * rate after all weights are already resident in PL DDR. */
            if (gen > 1u) {
                decode_warm_cycles += decode_token_cycles;
                decode_warm_tokens++;
            }
        }
        generated_count++;
        infer_output_append_token(last_token, gen);
        xil_printf("sample[%lu] token_id=%lu score=0x%08lx context=%lu phase=%s\r\n",
                   (unsigned long)gen,
                   (unsigned long)last_token,
                   (unsigned long)pseudo_score,
                   (unsigned long)token_count,
                   (gen == 0u) ? "prefill" : "decode");

        if (token_count < INFER_CONTEXT_MAX_TOKENS) {
            tokens[token_count++] = last_token;
        }
        infer_perf_publish(&perf, token_count, generated_count, last_token, pseudo_score);

        if ((last_token == TOKENIZER_EOS_ID) || (last_token == TOKENIZER_EOT_ID)) {
            break;
        }
    }
    total_end = perf_now_cycles();
    perf.total_cycles = total_end - total_start;
    infer_perf_publish(&perf, token_count, generated_count, last_token, pseudo_score);

    xil_printf("INFER_CHAIN_DONE prompt_tokens=%lu tokens=%lu generated=%lu cache_hits=%lu cache_misses=%lu\r\n",
               (unsigned long)prompt_token_count,
               (unsigned long)token_count,
               (unsigned long)generated_count,
               (unsigned long)perf.cache_hits,
               (unsigned long)perf.cache_misses);
    xil_printf("TIMING_US emmc=%lu accel=%lu prefill=%lu decode=%lu lm_head=%lu total=%lu\r\n",
               (unsigned long)cycles_to_us32(perf.emmc_cycles),
               (unsigned long)cycles_to_us32(perf.accel_cycles),
               (unsigned long)cycles_to_us32(perf.prefill_cycles),
               (unsigned long)cycles_to_us32(perf.decode_cycles),
               (unsigned long)cycles_to_us32(perf.lm_head_cycles),
               (unsigned long)cycles_to_us32(perf.total_cycles));
    {
        uint32_t prefill_rate = token_rate_milli(
            prompt_token_count, perf.prefill_cycles);
        uint32_t decode_rate = token_rate_milli(
            decode_e2e_tokens, decode_e2e_cycles);
        uint32_t warm_rate = token_rate_milli(
            decode_warm_tokens, decode_warm_cycles);

        xil_printf("PREFILL_PERF tokens=%lu elapsed_us=%lu tok_s=%lu.%03lu weight_load_excluded=1\r\n",
                   (unsigned long)prompt_token_count,
                   (unsigned long)cycles_to_us32(perf.prefill_cycles),
                   (unsigned long)(prefill_rate / 1000u),
                   (unsigned long)(prefill_rate % 1000u));
        xil_printf("DECODE_PERF tokens=%lu elapsed_us=%lu tok_s=%lu.%03lu lm_head_included=1\r\n",
                   (unsigned long)decode_e2e_tokens,
                   (unsigned long)cycles_to_us32(decode_e2e_cycles),
                   (unsigned long)(decode_rate / 1000u),
                   (unsigned long)(decode_rate % 1000u));
        xil_printf("DECODE_WARM_PERF tokens=%lu elapsed_us=%lu tok_s=%lu.%03lu weights_resident=1 lm_head_included=1\r\n",
                   (unsigned long)decode_warm_tokens,
                   (unsigned long)cycles_to_us32(decode_warm_cycles),
                   (unsigned long)(warm_rate / 1000u),
                   (unsigned long)(warm_rate % 1000u));
    }
    xil_printf("BLOCK_TIMING_US attention=%lu mlp=%lu attn_proj=%lu attn_core=%lu attn_nonlin=%lu mlp_proj=%lu mlp_nonlin=%lu layer=%lu\r\n",
               (unsigned long)cycles_to_us32(perf.attention_block_cycles),
               (unsigned long)cycles_to_us32(perf.mlp_block_cycles),
               (unsigned long)cycles_to_us32(perf.attention_projection_cycles),
               (unsigned long)cycles_to_us32(perf.attention_core_cycles),
               (unsigned long)cycles_to_us32(perf.attention_nonlinear_cycles),
               (unsigned long)cycles_to_us32(perf.mlp_projection_cycles),
               (unsigned long)cycles_to_us32(perf.mlp_nonlinear_cycles),
               (unsigned long)cycles_to_us32(perf.layer_forward_cycles));

    smoke_status_update(SMOKE_STAGE_INFER_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    return XST_SUCCESS;
}

static int run_real_tensor_test(const RealTensorControl *ctrl) {
    uint32_t expected_weight_bytes;
    uint32_t tx_words;
    uint32_t tx_capacity_words = REAL_TX_BYTES / sizeof(uint32_t);
    uint32_t chunk_out_features = REAL_CHUNK_OUT;
    uint32_t max_chunk_words;
    uint32_t mismatches = 0u;
    uint32_t out_features = ctrl->out_features;
    uint32_t in_features = ctrl->in_features;
    uint32_t *tx_words_ptr = (uint32_t *)REAL_TX_BASE;
    const uint8_t *packed_weights = (const uint8_t *)REAL_WEIGHT_BASE;
    int8_t *activations = (int8_t *)REAL_ACT_BASE;
    int32_t *rx_outputs = (int32_t *)REAL_OUTPUT_BASE;
    int status;

    smoke_status_set_dims(out_features, in_features);
    smoke_status_update(SMOKE_STAGE_REAL_DETECT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    if ((out_features == 0u) || ((out_features & 0x3u) != 0u) ||
        (in_features == 0u) || (in_features > REAL_MAX_K)) {
        smoke_status_update(SMOKE_STAGE_REAL_DETECT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    expected_weight_bytes = (out_features / 4u) * in_features;
    if (ctrl->weight_bytes != expected_weight_bytes) {
        smoke_status_update(SMOKE_STAGE_REAL_DETECT, SMOKE_RESULT_FAIL, XST_INVALID_PARAM);
        return XST_INVALID_PARAM;
    }

    if (chunk_out_features > out_features) {
        chunk_out_features = out_features;
    }
    max_chunk_words = bitnet_accel_stream_words(chunk_out_features, in_features);
    if (max_chunk_words > tx_capacity_words) {
        smoke_status_update(SMOKE_STAGE_REAL_DETECT, SMOKE_RESULT_FAIL, XST_BUFFER_TOO_SMALL);
        return XST_BUFFER_TOO_SMALL;
    }
    tx_words = bitnet_accel_chunked_stream_words(out_features, in_features, chunk_out_features);
    smoke_status->tx_words = tx_words;
    smoke_status->reserved0 = chunk_out_features;
    smoke_status->reserved1 = bitnet_accel_chunk_count(out_features, chunk_out_features);
    smoke_status_flush();

    status = init_dma(SMOKE_STAGE_DMA_INIT);
    if (status != XST_SUCCESS) {
        return status;
    }
    smoke_status_update(SMOKE_STAGE_DMA_INIT, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    Xil_DCacheInvalidateRange((UINTPTR)REAL_WEIGHT_BASE, expected_weight_bytes);
    make_real_activation(activations, in_features, ctrl->activation_seed);

    for (uint32_t n = 0u; n < out_features; n++) {
        rx_outputs[n] = 0;
    }

    smoke_status_update(SMOKE_STAGE_STREAM_BUILD, SMOKE_RESULT_RUNNING, XST_SUCCESS);

    smoke_status_record_dma();
    smoke_status_update(SMOKE_STAGE_DMA_RUN, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    status = bitnet_accel_run_gemv_chunked(&AxiDma, tx_words_ptr, tx_capacity_words,
                                           activations, packed_weights,
                                           out_features, in_features,
                                           chunk_out_features, ctrl->flags,
                                           rx_outputs);
    smoke_status_record_dma();
    if (status != XST_SUCCESS) {
        smoke_status_update(SMOKE_STAGE_DMA_RUN, SMOKE_RESULT_FAIL, status);
        return status;
    }

    smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_RUNNING, XST_SUCCESS);
    for (uint32_t n = 0u; n < out_features; n++) {
        int32_t ref = bitnet_accel_reference_dot4_layout(
            activations, packed_weights, n, in_features, ctrl->flags);
        if (n < STATUS_SAMPLE_N) {
            smoke_status->hw[n] = (uint32_t)rx_outputs[n];
            smoke_status->ref[n] = (uint32_t)ref;
        }
        if (rx_outputs[n] != ref) {
            if (mismatches == 0u) {
                smoke_status->first_mismatch = n;
            }
            mismatches++;
        }
        smoke_status->mismatches = mismatches;
        smoke_status_flush();
    }

    if (mismatches != 0u) {
        smoke_status_update(SMOKE_STAGE_COMPARE, SMOKE_RESULT_FAIL, XST_FAILURE);
        return XST_FAILURE;
    }

    smoke_status_update(SMOKE_STAGE_DONE, SMOKE_RESULT_PASS, XST_SUCCESS);
    return XST_SUCCESS;
}

int bitnet_runtime_dispatch(void) {
    EmmcInstallControl install_ctrl;
    InferControl infer_ctrl;
    EmmcLayerChainControl emmc_ctrl;
    LayerChainControl layer_ctrl;
    RealTensorControl real_ctrl;

    smoke_status_clear();

    if (emmc_install_control_loaded(&install_ctrl)) {
        xil_printf("BitNet AXU3EGB eMMC installer\r\n");
        return run_emmc_install(&install_ctrl);
    }

    if (infer_control_loaded(&infer_ctrl)) {
        return run_infer_entry(&infer_ctrl);
    }

    if (emmc_layer_chain_control_loaded(&emmc_ctrl)) {
        xil_printf("BitNet AXU3EGB eMMC layer-chain test\r\n");
        return run_emmc_layer_chain_test(&emmc_ctrl);
    }

    if (layer_chain_control_loaded(&layer_ctrl)) {
        xil_printf("BitNet AXU3EGB layer-chain test\r\n");
        return run_layer_chain_test(&layer_ctrl);
    }

    if (real_tensor_control_loaded(&real_ctrl)) {
        xil_printf("BitNet AXU3EGB real tensor test\r\n");
        return run_real_tensor_test(&real_ctrl);
    }

    return run_smoke_test();
}
