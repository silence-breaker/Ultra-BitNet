#ifndef ULTRA_BITNET_H
#define ULTRA_BITNET_H
#include <stddef.h>
#include <stdint.h>

#define BITNET_IDENTITY UINT32_C(0x48425432)
#define BITNET_CONTROL_BASE UINT64_C(0x80000000)
#define BITNET_FEATURES 2560u
#define BITNET_GROUPS 640u
#define BITNET_SOURCE_OFFSET UINT64_C(0x40000000)
#define BITNET_STAGE_BYTES UINT64_C(104693760)
#define BITNET_BANK0_BASE UINT64_C(0x850000000)
#define BITNET_SOURCE_PHYSICAL UINT64_C(0x840000000)

enum bitnet_register {
    BN_IDENTITY=0x00, BN_SUBMIT=0x04, BN_RESULT_ACK=0x08, BN_IRQ_ENABLE=0x0c,
    BN_SESSION_ID=0x10, BN_IMAGE_EPOCH=0x14, BN_ROUND_FLAGS=0x18,
    BN_POSITION_BASE=0x1c, BN_TOKEN_COUNT=0x20, BN_PS_SOURCE_BASE_LOW=0x24,
    BN_PS_SOURCE_BASE_HIGH=0x28, BN_PS_SOURCE_BYTES_LOW=0x2c,
    BN_PS_SOURCE_BYTES_HIGH=0x30, BN_STATUS=0x34, BN_HIDDEN_GROUP=0x38,
    BN_HIDDEN_REQUEST_ID=0x3c, BN_RESULT_STATUS=0x40, BN_RESULT_SESSION_ID=0x44,
    BN_RESULT_REQUEST_ID=0x48, BN_RESULT_TOKEN_ID=0x4c, BN_RESULT_FAULT=0x50,
    BN_RESULT_POSITION=0x54, BN_SHELL_FAULT_LOW=0x58, BN_SHELL_FAULT_HIGH=0x5c,
    BN_HIDDEN_DATA0=0x80, BN_HIDDEN_LAST=0xa0, BN_HIDDEN_PUSH=0xa4,
    BN_HIDDEN_PIO_STATUS=0xa8
};
enum bitnet_error {
    BN_OK=0, BN_INVALID=-1, BN_WRONG_DEVICE=-2, BN_BUSY=-3, BN_TIMEOUT=-4,
    BN_HARDWARE_FAULT=-5, BN_RESULT_MISMATCH=-6, BN_RESET_REQUIRED=-7
};
/* Callbacks perform ordered 32-bit device accesses. One caller owns a device. */
struct bitnet_io {
    void *context;
    uint32_t (*read32)(void *, uint32_t);
    void (*write32)(void *, uint32_t, uint32_t);
    uint64_t (*now_ms)(void *);
    void (*wait)(void *);
};
struct bitnet_device { struct bitnet_io io; int reset_required; };
struct bitnet_round {
    uint32_t session_id, image_epoch, mode, last_prompt, new_session;
    uint32_t position, tokens;
    uint64_t source_offset, source_bytes;
};
struct bitnet_result {
    uint32_t session_id, request_id, token_id, next_position;
    uint32_t run_epoch, fault_code, success, token_valid, valid;
    uint64_t shell_faults;
};
int bitnet_init(struct bitnet_device *, const struct bitnet_io *);
int bitnet_probe(struct bitnet_device *);
int bitnet_validate_round(const struct bitnet_round *);
int bitnet_run_round(struct bitnet_device *, const struct bitnet_round *,
                     const int32_t *hidden, size_t hidden_values,
                     struct bitnet_result *, uint32_t timeout_ms);
const char *bitnet_error_string(int);
#endif
