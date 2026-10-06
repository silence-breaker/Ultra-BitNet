#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "bitnet_linux.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void barrier(void) {
#if defined(__aarch64__)
    __asm__ volatile("dmb osh" ::: "memory");
#elif defined(__arm__)
    __asm__ volatile("dmb" ::: "memory");
#else
    __sync_synchronize();
#endif
}
static uint32_t read32(void *context,uint32_t offset) {
    struct bitnet_linux *m=context;
    barrier(); uint32_t value=m->registers[offset/4]; barrier(); return value;
}
static void write32(void *context,uint32_t offset,uint32_t value) {
    struct bitnet_linux *m=context;
    barrier(); m->registers[offset/4]=value; barrier();
}
static uint64_t now_ms(void *unused) {
    struct timespec ts; (void)unused;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (uint64_t)ts.tv_sec*1000+(uint64_t)ts.tv_nsec/1000000;
}
static void pause_poll(void *unused) {
    struct timespec ts={0,50000}; (void)unused; nanosleep(&ts,NULL);
}
int bitnet_linux_open(struct bitnet_linux *m,const char *path,uint64_t offset,struct bitnet_io *io) {
    struct stat st; long page=sysconf(_SC_PAGESIZE);
    if (!m || !path || !io || page<=0 || offset%4 || offset>INT64_MAX-(uint64_t)page) { errno=EINVAL; return -1; }
    memset(m,0,sizeof(*m)); m->fd=-1;
    uint64_t aligned=offset-offset%(uint64_t)page; size_t delta=(size_t)(offset-aligned);
    m->length=((delta+0x100+(size_t)page-1)/(size_t)page)*(size_t)page;
    m->fd=open(path,O_RDWR|O_SYNC|O_CLOEXEC);
    if (m->fd<0) return -1;
    if (flock(m->fd,LOCK_EX|LOCK_NB)<0 || fstat(m->fd,&st)<0) goto failed;
    if (S_ISREG(st.st_mode) && (uint64_t)st.st_size<aligned+m->length) { errno=EINVAL; goto failed; }
    m->mapping=mmap(NULL,m->length,PROT_READ|PROT_WRITE,MAP_SHARED,m->fd,(off_t)aligned);
    if (m->mapping==MAP_FAILED) { m->mapping=NULL; goto failed; }
    m->registers=(volatile uint32_t *)((unsigned char *)m->mapping+delta);
    *io=(struct bitnet_io){m,read32,write32,now_ms,pause_poll}; return 0;
failed: { int saved=errno; bitnet_linux_close(m); errno=saved; return -1; }
}
void bitnet_linux_close(struct bitnet_linux *m) {
    if (!m) return;
    if (m->mapping) munmap(m->mapping,m->length);
    if (m->fd>=0) close(m->fd);
    m->mapping=NULL; m->registers=NULL; m->fd=-1;
}
