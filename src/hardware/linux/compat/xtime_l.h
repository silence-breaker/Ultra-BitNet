#ifndef BITNET_LINUX_XTIME_L_H
#define BITNET_LINUX_XTIME_L_H

#include <stdint.h>
#include <time.h>

typedef uint64_t XTime;
#define COUNTS_PER_SECOND 1000000000ull

static inline void XTime_GetTime(XTime *value) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    *value = ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

#endif
