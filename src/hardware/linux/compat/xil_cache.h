#ifndef BITNET_LINUX_XIL_CACHE_H
#define BITNET_LINUX_XIL_CACHE_H

#include <stddef.h>
#include <stdint.h>

static inline void Xil_DCacheFlushRange(uintptr_t address, size_t length) {
    (void)address;
    (void)length;
    __sync_synchronize();
}

static inline void Xil_DCacheInvalidateRange(uintptr_t address, size_t length) {
    (void)address;
    (void)length;
    __sync_synchronize();
}

#endif
