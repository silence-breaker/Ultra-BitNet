#include "bitnet.h"
#include <string.h>

#define STATUS_FAULT ((1u<<5)|(1u<<6)|(1u<<7))
#define STATUS_BUSY ((1u<<0)|(1u<<2)|(1u<<3)|(1u<<8)|(1u<<9))
static uint32_t rd(struct bitnet_device *d, uint32_t r) { return d->io.read32(d->io.context,r); }
static void wr(struct bitnet_device *d, uint32_t r, uint32_t v) { d->io.write32(d->io.context,r,v); }
int bitnet_init(struct bitnet_device *d, const struct bitnet_io *io) {
    if (!d || !io || !io->read32 || !io->write32 || !io->now_ms || !io->wait) return BN_INVALID;
    memset(d,0,sizeof(*d)); d->io=*io; return BN_OK;
}
int bitnet_probe(struct bitnet_device *d) {
    return d && d->io.read32 ? (rd(d,BN_IDENTITY)==BITNET_IDENTITY ? BN_OK : BN_WRONG_DEVICE) : BN_INVALID;
}
int bitnet_validate_round(const struct bitnet_round *r) {
    if (!r || r->mode>1 || r->image_epoch>255 || r->new_session>1 || r->last_prompt>1 ||
        r->tokens<1 || r->tokens>2 || r->position>4096-r->tokens ||
        (r->mode==1 && (r->tokens!=1 || r->last_prompt)) ||
        (r->new_session && (r->mode!=0 || r->position!=0))) return BN_INVALID;
    /* Board-reserved staging window, expressed relative to the HP0 AXI base. */
    if (r->source_offset<BITNET_SOURCE_OFFSET || r->source_offset>=UINT64_C(0x50000000) ||
        (r->source_offset&4095) || r->source_bytes<BITNET_STAGE_BYTES ||
        r->source_bytes>UINT64_C(0x50000000)-r->source_offset) return BN_INVALID;
    return BN_OK;
}
static int expired(struct bitnet_device *d, uint64_t start, uint32_t timeout) {
    return d->io.now_ms(d->io.context)-start>=timeout;
}
static int fail(struct bitnet_device *d,int error) { d->reset_required=1; return error; }
static int take_result(struct bitnet_device *d, struct bitnet_result *r) {
    uint32_t s=rd(d,BN_RESULT_STATUS);
    if (!(s&1)) return 0;
    r->valid=1; r->success=(s>>1)&1; r->token_valid=(s>>2)&1; r->run_epoch=(s>>8)&255;
    r->session_id=rd(d,BN_RESULT_SESSION_ID); r->request_id=rd(d,BN_RESULT_REQUEST_ID);
    r->token_id=rd(d,BN_RESULT_TOKEN_ID); r->next_position=rd(d,BN_RESULT_POSITION);
    r->fault_code=rd(d,BN_RESULT_FAULT)&65535;
    r->shell_faults=(uint64_t)rd(d,BN_SHELL_FAULT_LOW)|((uint64_t)rd(d,BN_SHELL_FAULT_HIGH)<<32);
    /* ACK only the captured result. No late-result read follows a timeout. */
    wr(d,BN_RESULT_ACK,1);
    return 1;
}
static int finish(struct bitnet_device *d,const struct bitnet_round *round,
                  struct bitnet_result *result,uint32_t request,int have_request,int sent_all) {
    if (result->session_id!=round->session_id || (have_request && result->request_id!=request))
        return fail(d,BN_RESULT_MISMATCH);
    if (!result->success || result->fault_code || result->shell_faults) return fail(d,BN_HARDWARE_FAULT);
    if (!sent_all || result->next_position!=round->position+round->tokens ||
        result->token_valid!=(round->mode==1 || round->last_prompt) ||
        (result->token_valid && result->token_id>=128256)) return fail(d,BN_RESULT_MISMATCH);
    return BN_OK;
}
int bitnet_run_round(struct bitnet_device *d,const struct bitnet_round *r,
                     const int32_t *hidden,size_t count,struct bitnet_result *result,uint32_t timeout) {
    uint64_t start; uint32_t status,request=0; int rc,have_request=0;
    if (!d || !d->io.read32 || !result || !hidden || !timeout || bitnet_validate_round(r)!=BN_OK ||
        count!=(size_t)r->tokens*BITNET_FEATURES) return BN_INVALID;
    memset(result,0,sizeof(*result));
    if (d->reset_required) return BN_RESET_REQUIRED;
    rc=bitnet_probe(d); if (rc!=BN_OK) return rc;
    status=rd(d,BN_STATUS);
    if (status&STATUS_FAULT) return fail(d,BN_HARDWARE_FAULT);
    if (status&STATUS_BUSY) return BN_BUSY;
    if (!r->new_session && !(status&(1u<<4))) return BN_INVALID;
    start=d->io.now_ms(d->io.context);
    wr(d,BN_IRQ_ENABLE,0); /* This API polls; a level interrupt is not needed. */
    wr(d,BN_SESSION_ID,r->session_id); wr(d,BN_IMAGE_EPOCH,r->image_epoch);
    wr(d,BN_ROUND_FLAGS,r->mode|(r->last_prompt<<1)|(r->new_session<<2));
    wr(d,BN_POSITION_BASE,r->position); wr(d,BN_TOKEN_COUNT,r->tokens);
    wr(d,BN_PS_SOURCE_BASE_LOW,(uint32_t)r->source_offset);
    wr(d,BN_PS_SOURCE_BASE_HIGH,(uint32_t)(r->source_offset>>32));
    wr(d,BN_PS_SOURCE_BYTES_LOW,(uint32_t)r->source_bytes);
    wr(d,BN_PS_SOURCE_BYTES_HIGH,(uint32_t)(r->source_bytes>>32));
    wr(d,BN_SUBMIT,1);
    for (uint32_t group=0;group<BITNET_GROUPS;group++) {
        for (;;) {
            if (expired(d,start,timeout)) return fail(d,BN_TIMEOUT);
            if (take_result(d,result)) return finish(d,r,result,request,have_request,0);
            status=rd(d,BN_STATUS);
            if (status&STATUS_FAULT) return fail(d,BN_HARDWARE_FAULT);
            uint32_t pio=rd(d,BN_HIDDEN_PIO_STATUS);
            if (pio&2) return fail(d,BN_HARDWARE_FAULT);
            if ((status&2) && (pio&5)==4) break;
            d->io.wait(d->io.context);
        }
        uint32_t current=rd(d,BN_HIDDEN_REQUEST_ID);
        if (!have_request) { request=current; have_request=1; }
        if (current!=request || rd(d,BN_HIDDEN_GROUP)!=group) return fail(d,BN_RESULT_MISMATCH);
        for (uint32_t lane=0;lane<2;lane++) for (uint32_t shard=0;shard<4;shard++) {
            uint32_t value=lane<r->tokens ? (uint32_t)hidden[lane*BITNET_FEATURES+group*4+shard] : 0;
            wr(d,BN_HIDDEN_DATA0+(lane*4+shard)*4,value);
        }
        wr(d,BN_HIDDEN_LAST,group==BITNET_GROUPS-1);
        wr(d,BN_HIDDEN_PUSH,1);
    }
    for (;;) {
        if (expired(d,start,timeout)) return fail(d,BN_TIMEOUT);
        if (take_result(d,result)) return finish(d,r,result,request,have_request,1);
        if (rd(d,BN_STATUS)&STATUS_FAULT) return fail(d,BN_HARDWARE_FAULT);
        if (rd(d,BN_HIDDEN_PIO_STATUS)&2) return fail(d,BN_HARDWARE_FAULT);
        d->io.wait(d->io.context);
    }
}
const char *bitnet_error_string(int error) {
    switch(error) {
    case BN_OK:return "success"; case BN_INVALID:return "invalid descriptor or input";
    case BN_WRONG_DEVICE:return "board identity mismatch"; case BN_BUSY:return "device busy or result pending";
    case BN_TIMEOUT:return "round timed out; inspect/reset hardware before retry";
    case BN_HARDWARE_FAULT:return "hardware reported a fault";
    case BN_RESULT_MISMATCH:return "completion or hidden-input identity mismatch";
    case BN_RESET_REQUIRED:return "device recovery required after failed round";
    default:return "unknown error";
    }
}
