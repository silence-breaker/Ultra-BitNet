#define _POSIX_C_SOURCE 200809L
#include "bitnet_linux.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    puts("Usage: bitnetctl probe|status|round [options]\n"
         "  --uio /dev/uio0             UIO map 0 (default)\n"
         "  --mem /dev/mem --base ADDR  MMIO device/file and byte offset\n"
         "  --hidden FILE              lane-major signed little-endian Q16.16\n"
         "  --session N --epoch N --mode prefill|decode --position N --tokens 1|2\n"
         "  --new-session --last-prompt --timeout-ms N\n"
         "  --source-offset N          HP0-relative offset (default 0x40000000)\n"
         "  --source-bytes N           allocation bytes (default 104693760)\n"
         "Defaults: session=1 epoch=1 prefill position=0 tokens=1 timeout=60000ms\n"
         "probe/status do not write registers. round does not load model banks.");
}
static int number(const char *text,uint64_t *value) {
    char *end; if (!text || !*text || *text=='-' || *text=='+') return -1;
    errno=0; unsigned long long v=strtoull(text,&end,0);
    if (errno || *end) return -1;
    *value=(uint64_t)v; return 0;
}
static int read_hidden(const char *path,int32_t *values,size_t count) {
    FILE *f=fopen(path,"rb"); if (!f) return -1;
    for (size_t i=0;i<count;i++) {
        unsigned char b[4]; if (fread(b,1,4,f)!=4) { fclose(f); errno=EINVAL; return -1; }
        uint32_t v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
        memcpy(values+i,&v,sizeof(v));
    }
    int extra=fgetc(f),failed=ferror(f); fclose(f);
    if (extra!=EOF || failed) { errno=EINVAL; return -1; } return 0;
}
int main(int argc,char **argv) {
    if (argc<2 || !strcmp(argv[1],"--help")) { usage(); return argc<2 ? 2 : 0; }
    const char *command=argv[1],*device="/dev/uio0",*hidden_path=NULL;
    if (strcmp(command,"probe") && strcmp(command,"status") && strcmp(command,"round")) { usage(); return 2; }
    uint64_t base=0; uint32_t timeout=60000; int backend=0,base_set=0;
    struct bitnet_round r={1,1,0,0,0,0,1,BITNET_SOURCE_OFFSET,BITNET_STAGE_BYTES};
    for (int i=2;i<argc;i++) {
        const char *key=argv[i];
        if (!strcmp(key,"--new-session")) { r.new_session=1; continue; }
        if (!strcmp(key,"--last-prompt")) { r.last_prompt=1; continue; }
        if (i+1==argc) { fprintf(stderr,"Missing value for %s\n",key); return 2; }
        const char *value=argv[++i];
        if (!strcmp(key,"--uio") || !strcmp(key,"--mem")) {
            if (backend) { fputs("Select only one MMIO backend\n",stderr); return 2; }
            backend=!strcmp(key,"--uio") ? 1 : 2; device=value; continue;
        }
        if (!strcmp(key,"--hidden")) { hidden_path=value; continue; }
        if (!strcmp(key,"--mode")) {
            if (strcmp(value,"prefill") && strcmp(value,"decode")) { fputs("Invalid mode\n",stderr); return 2; }
            r.mode=!strcmp(value,"decode"); continue;
        }
        uint64_t v; if (number(value,&v)) { fprintf(stderr,"Invalid number: %s\n",value); return 2; }
        if (!strcmp(key,"--base")) { base=v; base_set=1; }
        else if (!strcmp(key,"--source-offset")) r.source_offset=v;
        else if (!strcmp(key,"--source-bytes")) r.source_bytes=v;
        else {
            if (v>UINT32_MAX) { fputs("Option exceeds 32-bit range\n",stderr); return 2; }
            if (!strcmp(key,"--session")) r.session_id=(uint32_t)v;
            else if (!strcmp(key,"--epoch")) r.image_epoch=(uint32_t)v;
            else if (!strcmp(key,"--position")) r.position=(uint32_t)v;
            else if (!strcmp(key,"--tokens")) r.tokens=(uint32_t)v;
            else if (!strcmp(key,"--timeout-ms")) timeout=(uint32_t)v;
            else { fprintf(stderr,"Unknown option: %s\n",key); return 2; }
        }
    }
    if (backend!=2 && base_set) { fputs("--base requires --mem\n",stderr); return 2; }
    if (backend==2 && !base_set) base=BITNET_CONTROL_BASE;
    int32_t hidden[BITNET_FEATURES*2];
    if (!strcmp(command,"round")) {
        if (!timeout || bitnet_validate_round(&r)!=BN_OK || !hidden_path) { fputs("Invalid round descriptor or missing --hidden\n",stderr); return 2; }
        if (read_hidden(hidden_path,hidden,r.tokens*BITNET_FEATURES)) { perror("hidden input"); return 2; }
    }
    struct bitnet_linux mapping; struct bitnet_io io; struct bitnet_device board;
    if (bitnet_linux_open(&mapping,device,base,&io)) { perror("MMIO mapping/lock"); return 1; }
    bitnet_init(&board,&io); int rc=bitnet_probe(&board);
    if (rc!=BN_OK) { fprintf(stderr,"%s\n",bitnet_error_string(rc)); bitnet_linux_close(&mapping); return 1; }
    if (!strcmp(command,"probe")) printf("{\"identity\":\"0x%08x\",\"compatible\":true}\n",BITNET_IDENTITY);
    else if (!strcmp(command,"status")) {
        uint32_t s=io.read32(io.context,BN_STATUS),p=io.read32(io.context,BN_HIDDEN_PIO_STATUS);
        printf("{\"status\":%u,\"pio_status\":%u,\"hidden_group\":%u,\"result_status\":%u}\n",
               s,p,io.read32(io.context,BN_HIDDEN_GROUP),io.read32(io.context,BN_RESULT_STATUS));
    } else {
        struct bitnet_result result={0};
        rc=bitnet_run_round(&board,&r,hidden,r.tokens*BITNET_FEATURES,&result,timeout);
        printf("{\"rc\":%d,\"result_valid\":%u,\"session_id\":%u,\"request_id\":%u,"
               "\"run_epoch\":%u,\"success\":%u,\"token_valid\":%u,\"token_id\":%u,"
               "\"next_position\":%u,\"fault_code\":%u,\"shell_fault_bits\":\"0x%016" PRIx64 "\"}\n",
               rc,result.valid,result.session_id,result.request_id,result.run_epoch,result.success,
               result.token_valid,result.token_id,result.next_position,result.fault_code,result.shell_faults);
        if (rc!=BN_OK) fprintf(stderr,"%s\n",bitnet_error_string(rc));
    }
    bitnet_linux_close(&mapping); return rc==BN_OK ? 0 : 1;
}
