#include "bitnet.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct mock {
    uint32_t reg[64],writes,pushes,acks,pending,finishing;
    uint64_t time;
    int stall,overrun,fatal,early_failure,wrong_session,wrong_request,bad_group,late;
    int32_t expected[BITNET_FEATURES*2];
};
static void completion(struct mock *m) {
    m->reg[BN_RESULT_STATUS/4]=1|(!m->early_failure<<1)|(((m->reg[BN_ROUND_FLAGS/4]&3)!=0)<<2)|(9<<8);
    m->reg[BN_RESULT_SESSION_ID/4]=m->reg[BN_SESSION_ID/4]+m->wrong_session;
    m->reg[BN_RESULT_REQUEST_ID/4]=77+m->wrong_request;
    m->reg[BN_RESULT_POSITION/4]=m->reg[BN_POSITION_BASE/4]+m->reg[BN_TOKEN_COUNT/4];
    m->reg[BN_RESULT_TOKEN_ID/4]=1234;
    m->reg[BN_RESULT_FAULT/4]=m->early_failure ? 0x12 : 0;
    m->reg[BN_STATUS/4]=(1<<3)|(1<<4);
}
static uint32_t read_reg(void *ctx,uint32_t r) {
    struct mock *m=ctx; return m->reg[r/4];
}
static void write_reg(void *ctx,uint32_t r,uint32_t v) {
    struct mock *m=ctx; m->writes++; m->reg[r/4]=v;
    if (r==BN_SUBMIT) {
        m->reg[BN_STATUS/4]=m->fatal ? (1<<6) : (1|2);
        m->reg[BN_HIDDEN_PIO_STATUS/4]=m->overrun ? 6 : 4;
        m->reg[BN_HIDDEN_REQUEST_ID/4]=77;
        m->reg[BN_HIDDEN_GROUP/4]=m->bad_group;
        if (m->early_failure) completion(m);
    } else if (r==BN_HIDDEN_PUSH) {
        assert(m->pending==0); assert(m->pushes<640);
        for (unsigned lane=0;lane<2;lane++) for (unsigned shard=0;shard<4;shard++) {
            uint32_t expected=lane<m->reg[BN_TOKEN_COUNT/4] ?
                (uint32_t)m->expected[lane*BITNET_FEATURES+m->pushes*4+shard] : 0;
            assert(m->reg[(BN_HIDDEN_DATA0+(lane*4+shard)*4)/4]==expected);
        }
        assert(m->reg[BN_HIDDEN_LAST/4]==(m->pushes==639));
        m->pushes++; m->pending=2; m->reg[BN_HIDDEN_PIO_STATUS/4]=1;
    } else if (r==BN_RESULT_ACK) { m->acks++; m->reg[BN_RESULT_STATUS/4]=0; }
}
static uint64_t now(void *ctx) {
    struct mock *m=ctx;
    if (m->late && m->time>=3) completion(m);
    return m->time;
}
static void wait_mock(void *ctx) {
    struct mock *m=ctx; m->time++;
    if (m->stall) return;
    if (m->pending && --m->pending==0) {
        m->reg[BN_HIDDEN_PIO_STATUS/4]=4; m->reg[BN_HIDDEN_GROUP/4]=m->pushes;
        if (m->pushes==640) { m->finishing=3; m->reg[BN_STATUS/4]=1; }
    } else if (m->finishing && --m->finishing==0) completion(m);
}
static void setup(struct mock *m,struct bitnet_device *d) {
    memset(m,0,sizeof(*m)); m->reg[BN_IDENTITY/4]=BITNET_IDENTITY;
    for (unsigned i=0;i<BITNET_FEATURES*2;i++) m->expected[i]=(int32_t)i-2600;
    struct bitnet_io io={m,read_reg,write_reg,now,wait_mock}; assert(bitnet_init(d,&io)==BN_OK);
}
static struct bitnet_round descriptor(void) {
    struct bitnet_round r={17,1,0,1,1,0,2,BITNET_SOURCE_OFFSET,BITNET_STAGE_BYTES}; return r;
}
int main(void) {
    struct mock m; struct bitnet_device d; struct bitnet_round r; struct bitnet_result result;
    setup(&m,&d); r=descriptor();
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,10000)==BN_OK);
    assert(m.pushes==640 && m.acks==1 && result.token_id==1234 && result.request_id==77 && result.run_epoch==9);
    assert(m.reg[BN_PS_SOURCE_BASE_LOW/4]==0x40000000 && m.reg[BN_PS_SOURCE_BASE_HIGH/4]==0);
    setup(&m,&d); r=descriptor(); r.tokens=1; r.last_prompt=0;
    assert(bitnet_run_round(&d,&r,m.expected,2560,&result,10000)==BN_OK && !result.token_valid);
    setup(&m,&d); r=descriptor(); r.tokens=1; r.mode=1; r.new_session=0; r.last_prompt=0; r.position=2;
    m.reg[BN_STATUS/4]=1<<4;
    assert(bitnet_run_round(&d,&r,m.expected,2560,&result,10000)==BN_OK && result.token_valid);
    setup(&m,&d); r=descriptor(); r.source_offset=BITNET_SOURCE_PHYSICAL;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_INVALID && m.writes==0);
    r=descriptor(); r.source_bytes=UINT64_MAX; assert(bitnet_validate_round(&r)==BN_INVALID);
    r=descriptor(); r.source_offset++; assert(bitnet_validate_round(&r)==BN_INVALID);
    r=descriptor(); r.position=4095; assert(bitnet_validate_round(&r)==BN_INVALID);
    r=descriptor(); r.mode=1; assert(bitnet_validate_round(&r)==BN_INVALID);
    r=descriptor(); r.image_epoch=256; assert(bitnet_validate_round(&r)==BN_INVALID);
    r=descriptor(); assert(bitnet_run_round(&d,&r,NULL,5120,&result,100)==BN_INVALID && m.writes==0);
    assert(bitnet_run_round(&d,&r,m.expected,1,&result,100)==BN_INVALID && m.writes==0);
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,0)==BN_INVALID && m.writes==0);
    m.reg[BN_STATUS/4]=1; assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_BUSY && !m.writes);
    m.reg[BN_STATUS/4]=1<<3; assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_BUSY && !m.acks);
    m.reg[BN_IDENTITY/4]=0; assert(bitnet_probe(&d)==BN_WRONG_DEVICE);
    setup(&m,&d); r=descriptor(); m.stall=1; m.late=1;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,3)==BN_TIMEOUT);
    assert(!result.valid && !m.acks && m.reg[BN_RESULT_STATUS/4]&1);
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,10)==BN_RESET_REQUIRED);
    setup(&m,&d); m.early_failure=1; r=descriptor();
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_HARDWARE_FAULT && m.acks==1 && !m.pushes);
    setup(&m,&d); m.overrun=1;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_HARDWARE_FAULT && !m.acks);
    setup(&m,&d); m.fatal=1;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_HARDWARE_FAULT);
    setup(&m,&d); m.wrong_session=1;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,10000)==BN_RESULT_MISMATCH && m.acks==1);
    setup(&m,&d); m.wrong_request=1;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,10000)==BN_RESULT_MISMATCH);
    setup(&m,&d); m.bad_group=1;
    assert(bitnet_run_round(&d,&r,m.expected,5120,&result,100)==BN_RESULT_MISMATCH && !m.pushes);
    puts("Driver scenarios passed: TP2/prefill/decode, 640 beats, backpressure, invalid descriptors, faults, identities, late-result timeout.");
    return 0;
}
