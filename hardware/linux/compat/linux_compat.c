#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ff.h"
#include "xaxidma.h"
#include "xaxidma_hw.h"
#include "xparameters.h"
#include "xstatus.h"

#define BITNET_ANON_BASE 0x08000000ull
#define BITNET_ANON_END 0x70000000ull
#define BITNET_DMA_RESERVED_BASE 0x70000000ull
#define BITNET_DMA_RESERVED_BYTES 0x08000000ull
#define BITNET_DMA_DIRECT_VIRT 0x60000000ull
#define BITNET_DMA_DIRECT_BYTES 0x02000000ull
#define BITNET_DMA_PREFILL_VIRT 0x64a00000ull
#define BITNET_DMA_PREFILL_BYTES 0x00400000ull
#define BITNET_DMA_PREFILL_OFFSET 0x01000000ull
#define BITNET_CMA_SCAN_BASE 0x40000000ull
#define BITNET_CMA_SCAN_BYTES 0x10000000ull
#define BITNET_CMA_PAGE_BYTES 0x00001000ull
#define BITNET_DMA_TX_PHYS 0x72000000ull
#define BITNET_DMA_TX_BYTES 0x00800000ull
#define BITNET_DMA_RX_PHYS 0x72800000ull
#define BITNET_DMA_RX_BYTES 0x00800000ull
#define BITNET_DMA_REG_BYTES 0x00010000ull
#define BITNET_DMA_MAX_TRANSFER_BYTES 0x007fffffu

/*
 * Keep local definitions so the application also builds with the Vitis 2020.1
 * sysroot, whose userspace headers predate dma-heap.
 */
struct bitnet_dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

struct bitnet_dma_buf_sync {
    uint64_t flags;
};

#define BITNET_DMA_HEAP_IOCTL_ALLOC \
    _IOWR('H', 0x0, struct bitnet_dma_heap_allocation_data)
#define BITNET_DMA_BUF_IOCTL_SYNC \
    _IOW('b', 0x0, struct bitnet_dma_buf_sync)
#define BITNET_DMA_BUF_SYNC_READ  (1ull << 0)
#define BITNET_DMA_BUF_SYNC_WRITE (2ull << 0)
#define BITNET_DMA_BUF_SYNC_END   (1ull << 2)
#define BITNET_DMA_BUF_SYNC_RW \
    (BITNET_DMA_BUF_SYNC_READ | BITNET_DMA_BUF_SYNC_WRITE)

static int devmem_fd = -1;
static int direct_dma_heap_fd = -1;
static int direct_dma_buf_fd = -1;
static char model_root_path[PATH_MAX] = "/opt/bitnet";
static void *rx_destination;
static uint32_t rx_length;
static int rx_copy_pending;
static int rx_direct_pending;
static int direct_dma_enabled;
static int direct_dma_cpu_access_active;
static int direct_dma_mapped;
static int direct_dma_prefill_mapped;
static int plddr_enabled;
static int plddr_mapped;
static int plddr_fast_copy;
static int psddr_cache_active;
static void *psddr_cache_mapping;
static int dma0_regs_mapped;
static int dma1_regs_mapped;
static uint64_t direct_dma_phys_base;
static uint64_t cma_scan_base = BITNET_CMA_SCAN_BASE;
static uint64_t cma_scan_bytes = BITNET_CMA_SCAN_BYTES;
static uint64_t direct_tx_bytes;
static uint64_t bounce_tx_bytes;
static uint64_t direct_rx_bytes;
static uint64_t bounce_rx_bytes;
static uint64_t physical_tx_bytes;
static uint64_t physical_tx_transfers;
static uint64_t tx0_submit_ns;
static uint64_t tx1_submit_ns;
static uint64_t tx0_dma_elapsed_ns;
static uint64_t tx1_dma_elapsed_ns;
static uint64_t tx0_dma_transfers;
static uint64_t tx1_dma_transfers;
static uint64_t tx_busy_polls;
static uint64_t rx_busy_polls;
static uint64_t rx_submit_ns;
static uint64_t rx_dma_elapsed_ns;
static uint64_t rx_dma_transfers;
static char cma_trace_instance[PATH_MAX];
static int cma_trace_active;

static uint64_t monotonic_ns(void) {
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        return 0u;
    }
    return ((uint64_t)now.tv_sec * UINT64_C(1000000000)) +
           (uint64_t)now.tv_nsec;
}

static void *map_fixed(void *address, size_t length, int flags, int fd, off_t offset) {
    void *mapped = mmap(address, length, PROT_READ | PROT_WRITE,
                        flags | MAP_FIXED, fd, offset);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap(%p, 0x%zx) failed: %s\n", address, length,
                strerror(errno));
        return NULL;
    }
    return mapped;
}

static int direct_dma_sync(uint64_t flags) {
    struct bitnet_dma_buf_sync sync = { flags };

    if (direct_dma_buf_fd < 0 ||
        ioctl(direct_dma_buf_fd, BITNET_DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        perror("DMA_BUF_IOCTL_SYNC");
        return -1;
    }
    return 0;
}

static int direct_dma_begin_cpu_access(void) {
    if (direct_dma_cpu_access_active) {
        return 0;
    }
    if (direct_dma_sync(BITNET_DMA_BUF_SYNC_RW) != 0) {
        return -1;
    }
    direct_dma_cpu_access_active = 1;
    return 0;
}

static int direct_dma_end_cpu_access(void) {
    if (!direct_dma_cpu_access_active) {
        return 0;
    }
    if (direct_dma_sync(BITNET_DMA_BUF_SYNC_RW |
                        BITNET_DMA_BUF_SYNC_END) != 0) {
        return -1;
    }
    direct_dma_cpu_access_active = 0;
    return 0;
}

static int marker_matches(const volatile uint32_t *candidate,
                          const uint32_t marker[4]) {
    return candidate[0] == marker[0] && candidate[1] == marker[1] &&
           candidate[2] == marker[2] && candidate[3] == marker[3];
}

static void configure_cma_scan_range(void) {
    char command_line[4096];
    unsigned long long megabytes;
    unsigned long long base;
    FILE *file = fopen("/proc/cmdline", "r");
    char *argument;

    if (!file) {
        return;
    }
    if (!fgets(command_line, sizeof(command_line), file)) {
        fclose(file);
        return;
    }
    fclose(file);
    argument = strstr(command_line, "cma=");
    if (!argument ||
        sscanf(argument, "cma=%lluM@%llx", &megabytes, &base) != 2) {
        return;
    }
    if (megabytes == 0u || megabytes > (UINT64_MAX >> 20) ||
        (megabytes << 20) < BITNET_DMA_DIRECT_BYTES) {
        return;
    }
    cma_scan_base = (uint64_t)base;
    cma_scan_bytes = (uint64_t)megabytes << 20;
}

static int physical_range_is_dma_backed(uint64_t base, uint64_t bytes) {
    const uint64_t range_bases[2] = {
        cma_scan_base, BITNET_DMA_RESERVED_BASE
    };
    const uint64_t range_bytes[2] = {
        cma_scan_bytes, BITNET_DMA_RESERVED_BYTES
    };

    for (unsigned i = 0u; i < 2u; ++i) {
        if (base >= range_bases[i] && bytes <= range_bytes[i] &&
            base - range_bases[i] <= range_bytes[i] - bytes) {
            return 1;
        }
    }
    return 0;
}

static int write_text_file(const char *path, const char *text) {
    size_t length = strlen(text);
    size_t completed = 0u;
    int fd = open(path, O_WRONLY | O_CLOEXEC);

    if (fd < 0) {
        return -1;
    }
    while (completed < length) {
        ssize_t bytes = write(fd, text + completed, length - completed);

        if (bytes <= 0) {
            close(fd);
            return -1;
        }
        completed += (size_t)bytes;
    }
    close(fd);
    return 0;
}

static void stop_cma_allocation_trace(void) {
    char path[PATH_MAX];

    if (!cma_trace_active) {
        return;
    }
    if (snprintf(path, sizeof(path), "%s/tracing_on", cma_trace_instance) <
        (int)sizeof(path)) {
        (void)write_text_file(path, "0\n");
    }
    if (snprintf(path, sizeof(path),
                 "%s/events/cma/cma_alloc_finish/enable",
                 cma_trace_instance) < (int)sizeof(path)) {
        (void)write_text_file(path, "0\n");
    }
    (void)rmdir(cma_trace_instance);
    cma_trace_instance[0] = '\0';
    cma_trace_active = 0;
}

/*
 * dma-buf intentionally hides its physical address from userspace.  This
 * appliance already requires root for /dev/mem, and its kernel exposes the
 * cma_alloc_finish tracepoint.  A private tracefs instance filtered to this
 * PID provides the allocator's PFN without cache aliases or hard-coded CMA
 * offsets.  The instance exists only around the allocation ioctl.
 */
static int start_cma_allocation_trace(void) {
    static const char *trace_roots[] = {
        "/sys/kernel/tracing", "/sys/kernel/debug/tracing"
    };
    char path[PATH_MAX];
    char pid[32];
    const char *trace_root = NULL;

    for (unsigned i = 0u; i < sizeof(trace_roots) / sizeof(trace_roots[0]); ++i) {
        if (access(trace_roots[i], W_OK) == 0) {
            trace_root = trace_roots[i];
            break;
        }
    }
    if (!trace_root ||
        snprintf(cma_trace_instance, sizeof(cma_trace_instance),
                 "%s/instances/bitnet-%ld", trace_root, (long)getpid()) >=
            (int)sizeof(cma_trace_instance) ||
        mkdir(cma_trace_instance, 0700) != 0) {
        cma_trace_instance[0] = '\0';
        return -1;
    }
    cma_trace_active = 1;
    if (snprintf(path, sizeof(path), "%s/tracing_on", cma_trace_instance) >=
            (int)sizeof(path) ||
        write_text_file(path, "0\n") != 0 ||
        snprintf(path, sizeof(path), "%s/trace", cma_trace_instance) >=
            (int)sizeof(path) ||
        write_text_file(path, "\n") != 0 ||
        snprintf(pid, sizeof(pid), "%ld\n", (long)getpid()) >=
            (int)sizeof(pid) ||
        snprintf(path, sizeof(path), "%s/set_event_pid", cma_trace_instance) >=
            (int)sizeof(path) ||
        write_text_file(path, pid) != 0 ||
        snprintf(path, sizeof(path),
                 "%s/events/cma/cma_alloc_finish/enable",
                 cma_trace_instance) >= (int)sizeof(path) ||
        write_text_file(path, "1\n") != 0 ||
        snprintf(path, sizeof(path), "%s/tracing_on", cma_trace_instance) >=
            (int)sizeof(path) ||
        write_text_file(path, "1\n") != 0) {
        stop_cma_allocation_trace();
        return -1;
    }
    return 0;
}

static int finish_cma_allocation_trace(void) {
    char path[PATH_MAX];
    char line[1024];
    unsigned long long pfn = 0u;
    unsigned long long count = 0u;
    int error_number = -1;
    int found = 0;
    FILE *trace = NULL;
    const uint64_t expected_pages = BITNET_DMA_DIRECT_BYTES /
        (uint64_t)sysconf(_SC_PAGESIZE);

    if (!cma_trace_active || expected_pages == 0u) {
        return -1;
    }
    if (snprintf(path, sizeof(path), "%s/tracing_on", cma_trace_instance) <
        (int)sizeof(path)) {
        (void)write_text_file(path, "0\n");
    }
    if (snprintf(path, sizeof(path), "%s/trace", cma_trace_instance) >=
        (int)sizeof(path)) {
        stop_cma_allocation_trace();
        return -1;
    }
    trace = fopen(path, "r");
    if (trace) {
        while (fgets(line, sizeof(line), trace)) {
            char *event = strstr(line, "cma_alloc_finish:");

            if (event &&
                sscanf(event,
                       "cma_alloc_finish: name=%*s pfn=0x%llx page=%*s "
                       "count=%llu align=%*u errorno=%d",
                       &pfn, &count, &error_number) == 3 &&
                error_number == 0 && count == expected_pages) {
                found = 1;
                break;
            }
        }
        fclose(trace);
    }
    stop_cma_allocation_trace();
    if (!found) {
        return -1;
    }
    direct_dma_phys_base = pfn * (uint64_t)sysconf(_SC_PAGESIZE);
    if (!physical_range_is_dma_backed(direct_dma_phys_base,
                                      BITNET_DMA_DIRECT_BYTES)) {
        direct_dma_phys_base = 0u;
        return -1;
    }
    fprintf(stderr, "DMA_COMPAT cma_trace_phys=0x%llx heap_bytes=0x%llx\n",
            (unsigned long long)direct_dma_phys_base,
            (unsigned long long)BITNET_DMA_DIRECT_BYTES);
    return 0;
}

/*
 * Resolve the cacheable dma-buf mapping through the process page tables.
 * This avoids creating a second /dev/mem alias solely to search for marker
 * words; such aliases can observe stale cache lines on arm64 and made direct
 * DMA availability depend on cache eviction timing.  The runtime executes as
 * root on the appliance image, so CAP_SYS_ADMIN exposes PFNs in pagemap.
 */
static int discover_direct_dma_physical_base_pagemap(void) {
    const uintptr_t virtual_base = (uintptr_t)BITNET_DMA_DIRECT_VIRT;
    const long system_page_bytes = sysconf(_SC_PAGESIZE);
    uint64_t *entries = NULL;
    size_t page_bytes;
    size_t page_count;
    size_t entries_bytes;
    size_t completed = 0u;
    uint64_t first_pfn = 0u;
    int pagemap_fd = -1;
    int result = -1;

    if (system_page_bytes <= 0) {
        perror("sysconf(_SC_PAGESIZE)");
        return -1;
    }
    page_bytes = (size_t)system_page_bytes;
    if ((virtual_base % page_bytes) != 0u ||
        (BITNET_DMA_DIRECT_BYTES % page_bytes) != 0u) {
        return -1;
    }
    page_count = (size_t)(BITNET_DMA_DIRECT_BYTES / page_bytes);
    if (page_count == 0u || page_count > SIZE_MAX / sizeof(*entries)) {
        return -1;
    }
    entries_bytes = page_count * sizeof(*entries);
    entries = malloc(entries_bytes);
    if (!entries) {
        perror("malloc pagemap entries");
        return -1;
    }

    /* Fault every dma-buf page in before reading its pagemap entry. */
    for (size_t i = 0u; i < page_count; ++i) {
        volatile const uint8_t *page =
            (volatile const uint8_t *)(virtual_base + i * page_bytes);
        (void)*page;
    }

    pagemap_fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (pagemap_fd < 0) {
        perror("open /proc/self/pagemap");
        goto done;
    }
    while (completed < entries_bytes) {
        uint64_t first_page = (uint64_t)virtual_base / page_bytes;
        off_t offset = (off_t)(first_page * sizeof(*entries) + completed);
        ssize_t bytes = pread(pagemap_fd, (uint8_t *)entries + completed,
                              entries_bytes - completed, offset);

        if (bytes <= 0) {
            if (bytes < 0) {
                perror("pread /proc/self/pagemap");
            } else {
                fprintf(stderr, "short /proc/self/pagemap read at byte %zu\n",
                        completed);
            }
            goto done;
        }
        completed += (size_t)bytes;
    }

    for (size_t i = 0u; i < page_count; ++i) {
        const uint64_t entry = entries[i];
        const uint64_t present = UINT64_C(1) << 63;
        const uint64_t swapped = UINT64_C(1) << 62;
        const uint64_t pfn_mask = (UINT64_C(1) << 55) - 1u;
        const uint64_t pfn = entry & pfn_mask;

        if ((entry & present) == 0u || (entry & swapped) != 0u || pfn == 0u) {
            fprintf(stderr, "pagemap entry unavailable at page %zu: 0x%llx\n",
                    i, (unsigned long long)entry);
            goto done;
        }
        if (i == 0u) {
            first_pfn = pfn;
        } else if (pfn != first_pfn + i) {
            fprintf(stderr, "dma-heap pagemap is not physically contiguous "
                            "at page %zu\n", i);
            goto done;
        }
    }

    direct_dma_phys_base = first_pfn * page_bytes;
    if (!physical_range_is_dma_backed(direct_dma_phys_base,
                                      BITNET_DMA_DIRECT_BYTES)) {
        fprintf(stderr, "dma-heap pagemap physical range 0x%llx+0x%llx "
                        "is outside CMA/reserved memory\n",
                (unsigned long long)direct_dma_phys_base,
                (unsigned long long)BITNET_DMA_DIRECT_BYTES);
        direct_dma_phys_base = 0u;
        goto done;
    }
    fprintf(stderr, "DMA_COMPAT pagemap_phys=0x%llx heap_bytes=0x%llx\n",
            (unsigned long long)direct_dma_phys_base,
            (unsigned long long)BITNET_DMA_DIRECT_BYTES);
    result = 0;

done:
    if (pagemap_fd >= 0) {
        close(pagemap_fd);
    }
    free(entries);
    return result;
}

/*
 * The dma-buf API deliberately does not expose a physical address to
 * userspace, while the legacy standalone AXI DMA API requires one.  The
 * reserved heap is backed by the board's physically contiguous CMA region.
 * Locate this experimental allocation once using two cache-synchronised page
 * markers and verify that its first and last pages have the expected spacing.
 */
static int discover_direct_dma_physical_base(void) {
    static const uint32_t first_marker[4] = {
        0x31484e42u, 0x6d444d43u, 0xa55a19e7u, 0x83c42f10u
    };
    static const uint32_t last_marker[4] = {
        0x32484e42u, 0x6d444d43u, 0x5aa5e718u, 0x7c3bd0efu
    };
    uint8_t *direct = (uint8_t *)(uintptr_t)BITNET_DMA_DIRECT_VIRT;
    volatile uint8_t *scan;
    uint64_t found = UINT64_MAX;
    unsigned matches = 0u;
    const uint64_t scan_bases[2] = {
        cma_scan_base, BITNET_DMA_RESERVED_BASE
    };
    const uint64_t scan_sizes[2] = {
        cma_scan_bytes, BITNET_DMA_RESERVED_BYTES
    };

    if (direct_dma_begin_cpu_access() != 0) {
        return -1;
    }
    memcpy(direct, first_marker, sizeof(first_marker));
    memcpy(direct + BITNET_DMA_DIRECT_BYTES - BITNET_CMA_PAGE_BYTES,
           last_marker, sizeof(last_marker));
    if (direct_dma_end_cpu_access() != 0) {
        return -1;
    }

    for (unsigned range = 0u; range < 2u; range++) {
        uint64_t range_base = scan_bases[range];
        uint64_t range_bytes = scan_sizes[range];

        if (range_bytes < BITNET_DMA_DIRECT_BYTES ||
            (range != 0u && range_base == scan_bases[0] &&
             range_bytes == scan_sizes[0])) {
            continue;
        }
        scan = mmap(NULL, (size_t)range_bytes, PROT_READ, MAP_SHARED,
                    devmem_fd, (off_t)range_base);
        if (scan == MAP_FAILED) {
            continue;
        }
        for (uint64_t offset = 0u;
             offset + BITNET_DMA_DIRECT_BYTES <= range_bytes;
             offset += BITNET_CMA_PAGE_BYTES) {
            const volatile uint32_t *first =
                (const volatile uint32_t *)(scan + offset);
            const volatile uint32_t *last = (const volatile uint32_t *)(
                scan + offset + BITNET_DMA_DIRECT_BYTES - BITNET_CMA_PAGE_BYTES);

            if (marker_matches(first, first_marker) &&
                marker_matches(last, last_marker)) {
                found = range_base + offset;
                ++matches;
            }
        }
        munmap((void *)scan, (size_t)range_bytes);
    }

    if (matches != 1u) {
        fprintf(stderr, "could not uniquely locate dma-heap allocation "
                        "in CMA/reserved ranges (matches=%u)\n", matches);
        return -1;
    }
    direct_dma_phys_base = found;

    if (direct_dma_begin_cpu_access() != 0) {
        return -1;
    }
    memset(direct, 0, sizeof(first_marker));
    memset(direct + BITNET_DMA_DIRECT_BYTES - BITNET_CMA_PAGE_BYTES,
           0, sizeof(last_marker));
    if (direct_dma_end_cpu_access() != 0) {
        return -1;
    }
    fprintf(stderr, "DMA_COMPAT heap_phys=0x%llx heap_bytes=0x%llx\n",
            (unsigned long long)direct_dma_phys_base,
            (unsigned long long)BITNET_DMA_DIRECT_BYTES);
    return 0;
}

static int allocate_direct_dma_heap(void) {
    struct bitnet_dma_heap_allocation_data allocation = {0};
    int trace_started;

    direct_dma_heap_fd = open("/dev/dma_heap/reserved", O_RDWR | O_CLOEXEC);
    if (direct_dma_heap_fd < 0) {
        perror("open /dev/dma_heap/reserved");
        return -1;
    }
    allocation.len = BITNET_DMA_DIRECT_BYTES;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    trace_started = start_cma_allocation_trace() == 0;
    if (ioctl(direct_dma_heap_fd, BITNET_DMA_HEAP_IOCTL_ALLOC,
              &allocation) != 0) {
        if (trace_started) {
            stop_cma_allocation_trace();
        }
        perror("DMA_HEAP_IOCTL_ALLOC");
        return -1;
    }
    direct_dma_buf_fd = (int)allocation.fd;
    if (!map_fixed((void *)(uintptr_t)BITNET_DMA_DIRECT_VIRT,
                   BITNET_DMA_DIRECT_BYTES, MAP_SHARED,
                   direct_dma_buf_fd, 0)) {
        return -1;
    }
    direct_dma_mapped = 1;
    if (!map_fixed((void *)(uintptr_t)BITNET_DMA_PREFILL_VIRT,
                   BITNET_DMA_PREFILL_BYTES, MAP_SHARED,
                   direct_dma_buf_fd, BITNET_DMA_PREFILL_OFFSET)) {
        return -1;
    }
    direct_dma_prefill_mapped = 1;
    if (trace_started && finish_cma_allocation_trace() == 0) {
        return 0;
    }
    if (trace_started) {
        fprintf(stderr, "DMA_COMPAT CMA trace lookup unavailable; "
                        "falling back to pagemap\n");
    }
    if (discover_direct_dma_physical_base_pagemap() == 0) {
        return 0;
    }
    fprintf(stderr, "DMA_COMPAT pagemap lookup unavailable; "
                    "falling back to marker scan\n");
    return discover_direct_dma_physical_base();
}

static void release_direct_dma_heap(void) {
    if (direct_dma_buf_fd >= 0) {
        if (direct_dma_cpu_access_active) {
            (void)direct_dma_end_cpu_access();
        }
        if (direct_dma_prefill_mapped) {
            munmap((void *)(uintptr_t)BITNET_DMA_PREFILL_VIRT,
                   BITNET_DMA_PREFILL_BYTES);
        }
        if (direct_dma_mapped) {
            munmap((void *)(uintptr_t)BITNET_DMA_DIRECT_VIRT,
                   BITNET_DMA_DIRECT_BYTES);
        }
        close(direct_dma_buf_fd);
        direct_dma_buf_fd = -1;
    }
    if (direct_dma_heap_fd >= 0) {
        close(direct_dma_heap_fd);
        direct_dma_heap_fd = -1;
    }
    direct_dma_phys_base = 0u;
    direct_dma_cpu_access_active = 0;
    direct_dma_mapped = 0;
    direct_dma_prefill_mapped = 0;
}

int bitnet_linux_runtime_init(const char *model_root) {
    size_t anon_bytes = (size_t)(BITNET_ANON_END - BITNET_ANON_BASE);
    const char *direct_dma = getenv("BITNET_DIRECT_DMA");
    const char *plddr = getenv("BITNET_PLDDR_ENABLE");
    const char *plddr_copy_mode = getenv("BITNET_PLDDR_MEMCPY");
    const char *split_memory = getenv("BITNET_DUAL_MEMORY_SPLIT");

    if (model_root && model_root[0]) {
        if (snprintf(model_root_path, sizeof(model_root_path), "%s", model_root) >=
            (int)sizeof(model_root_path)) {
            return -1;
        }
    }
    if (!map_fixed((void *)(uintptr_t)BITNET_ANON_BASE, anon_bytes,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)) {
        return -1;
    }
    /* Weight and LM-head scans are long sequential walks.  Requesting THP
     * reduces page-walk pressure without changing the fixed-address ABI; the
     * kernel simply ignores the hint when huge pages are unavailable. */
    (void)madvise((void *)(uintptr_t)BITNET_ANON_BASE, anon_bytes,
                  MADV_HUGEPAGE);

    devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) {
        perror("open /dev/mem");
        return -1;
    }
    psddr_cache_active = split_memory != NULL && split_memory[0] == '1';
    if (psddr_cache_active) {
        psddr_cache_mapping = mmap(NULL, (size_t)XPAR_PS_DDR_CACHE_SIZE,
                                   PROT_READ | PROT_WRITE, MAP_SHARED,
                                   devmem_fd,
                                   (off_t)XPAR_PS_DDR_CACHE_BASEADDR);
        if (psddr_cache_mapping == MAP_FAILED) {
            psddr_cache_mapping = NULL;
            perror("mmap PS DDR packet cache");
            return -1;
        }
        fprintf(stderr,
                "DMA_COMPAT split_memory=1 psddr_cache=0x%llx+0x%llx\n",
                (unsigned long long)XPAR_PS_DDR_CACHE_BASEADDR,
                (unsigned long long)XPAR_PS_DDR_CACHE_SIZE);
    }
    if (!map_fixed((void *)(uintptr_t)BITNET_DMA_RESERVED_BASE,
                   BITNET_DMA_RESERVED_BYTES, MAP_SHARED, devmem_fd,
                   BITNET_DMA_RESERVED_BASE)) {
        return -1;
    }
    /* Keep the conservative bounce-buffer path as the default.  The direct
     * dma-heap mapping is board/kernel dependent; on AXU3EGB it can expose
     * cache-coherency differences between runs.  Opt in explicitly with
     * BITNET_DIRECT_DMA=1 after a board-specific coherency regression. */
    if (!direct_dma) {
        direct_dma = getenv("BITNET_DIRECT_DMA_TEST");
    }
    direct_dma_enabled = direct_dma != NULL && strcmp(direct_dma, "0") != 0;
    if (direct_dma_enabled) {
        configure_cma_scan_range();
    }
    if (direct_dma_enabled && allocate_direct_dma_heap() != 0) {
        int restore_direct_range = direct_dma_mapped;
        int restore_prefill_range = direct_dma_prefill_mapped;

        fprintf(stderr, "DMA_COMPAT direct DMA unavailable; using bounce buffers\n");
        release_direct_dma_heap();
        if (restore_direct_range &&
            !map_fixed((void *)(uintptr_t)BITNET_DMA_DIRECT_VIRT,
                       BITNET_DMA_DIRECT_BYTES,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)) {
            return -1;
        }
        if (restore_prefill_range &&
            !map_fixed((void *)(uintptr_t)BITNET_DMA_PREFILL_VIRT,
                       BITNET_DMA_PREFILL_BYTES,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)) {
            return -1;
        }
        direct_dma_enabled = 0;
    }
    /* The shipped hardware is the PL-DDR dual-lane design.  Keep it enabled
     * by default so the production command cannot silently fall back to the
     * legacy path; BITNET_PLDDR_ENABLE=0 is the explicit compatibility mode. */
    plddr_enabled = plddr == NULL || strcmp(plddr, "0") != 0;
    /* Paired/SIMD libc stores passed repeated 1-layer and full-model board
     * regressions.  Keep an explicit zero-valued escape hatch for boards
     * whose /dev/mem mapping requires conservative 32-bit accesses. */
    plddr_fast_copy = plddr_copy_mode == NULL ||
                      strcmp(plddr_copy_mode, "0") != 0;
    if (plddr_enabled) {
        if (!map_fixed((void *)(uintptr_t)XPAR_PL_DDR4_0_BASEADDR,
                       (size_t)XPAR_PL_DDR4_0_SIZE, MAP_SHARED, devmem_fd,
                       (off_t)XPAR_PL_DDR4_0_BASEADDR) ||
            !map_fixed((void *)(uintptr_t)XPAR_PLDDR_STATUS_GPIO_BASEADDR,
                       BITNET_DMA_REG_BYTES, MAP_SHARED, devmem_fd,
                       XPAR_PLDDR_STATUS_GPIO_BASEADDR)) {
            return -1;
        }
        plddr_mapped = 1;
    }
    if (!map_fixed((void *)(uintptr_t)XPAR_AXI_DMA_0_BASEADDR,
                   BITNET_DMA_REG_BYTES, MAP_SHARED, devmem_fd,
                   XPAR_AXI_DMA_0_BASEADDR)) {
        return -1;
    }
    dma0_regs_mapped = 1;
    /* DMA1 only exists in the PL-DDR dual-lane hardware variant.  Mapping a
     * physical hole succeeds on /dev/mem, but the first register access can
     * raise an asynchronous external abort.  Do not touch this aperture when
     * running the stable single-lane bitstream. */
    if (plddr_enabled) {
        if (!map_fixed((void *)(uintptr_t)XPAR_AXI_DMA_1_BASEADDR,
                       BITNET_DMA_REG_BYTES, MAP_SHARED, devmem_fd,
                       XPAR_AXI_DMA_1_BASEADDR)) {
            return -1;
        }
        dma1_regs_mapped = 1;
    }
    return 0;
}

void bitnet_linux_runtime_close(void) {
    fprintf(stderr,
            "DMA_COMPAT direct_tx_bytes=%llu bounce_tx_bytes=%llu "
            "direct_rx_bytes=%llu bounce_rx_bytes=%llu "
            "physical_tx_bytes=%llu physical_tx_transfers=%llu "
            "tx0_dma_us=%llu tx0_transfers=%llu "
            "tx1_dma_us=%llu tx1_transfers=%llu "
            "rx_dma_us=%llu rx_transfers=%llu "
            "tx_polls=%llu rx_polls=%llu\n",
            (unsigned long long)direct_tx_bytes,
            (unsigned long long)bounce_tx_bytes,
            (unsigned long long)direct_rx_bytes,
            (unsigned long long)bounce_rx_bytes,
            (unsigned long long)physical_tx_bytes,
            (unsigned long long)physical_tx_transfers,
            (unsigned long long)(tx0_dma_elapsed_ns / UINT64_C(1000)),
            (unsigned long long)tx0_dma_transfers,
            (unsigned long long)(tx1_dma_elapsed_ns / UINT64_C(1000)),
            (unsigned long long)tx1_dma_transfers,
            (unsigned long long)(rx_dma_elapsed_ns / UINT64_C(1000)),
            (unsigned long long)rx_dma_transfers,
            (unsigned long long)tx_busy_polls,
            (unsigned long long)rx_busy_polls);
    release_direct_dma_heap();
    if (psddr_cache_mapping != NULL) {
        munmap(psddr_cache_mapping, (size_t)XPAR_PS_DDR_CACHE_SIZE);
        psddr_cache_mapping = NULL;
        psddr_cache_active = 0;
    }
    if (plddr_mapped) {
        munmap((void *)(uintptr_t)XPAR_PLDDR_STATUS_GPIO_BASEADDR,
               BITNET_DMA_REG_BYTES);
        munmap((void *)(uintptr_t)XPAR_PL_DDR4_0_BASEADDR,
               (size_t)XPAR_PL_DDR4_0_SIZE);
        plddr_mapped = 0;
    }
    if (dma1_regs_mapped) {
        munmap((void *)(uintptr_t)XPAR_AXI_DMA_1_BASEADDR,
               BITNET_DMA_REG_BYTES);
        dma1_regs_mapped = 0;
    }
    if (dma0_regs_mapped) {
        munmap((void *)(uintptr_t)XPAR_AXI_DMA_0_BASEADDR,
               BITNET_DMA_REG_BYTES);
        dma0_regs_mapped = 0;
    }
    if (devmem_fd >= 0) {
        close(devmem_fd);
        devmem_fd = -1;
    }
}

const char *bitnet_linux_model_root(void) {
    return model_root_path;
}

void *bitnet_plddr_pointer(uint64_t physical, size_t length) {
    uint64_t offset;

    if (!plddr_mapped || physical < XPAR_PL_DDR4_0_BASEADDR) {
        return NULL;
    }
    offset = physical - XPAR_PL_DDR4_0_BASEADDR;
    if (length > XPAR_PL_DDR4_0_SIZE ||
        offset > XPAR_PL_DDR4_0_SIZE - length) {
        return NULL;
    }
    return (void *)(uintptr_t)physical;
}

int bitnet_plddr_copy_to(uint64_t physical, const void *source, size_t length) {
    void *destination;
    const uint32_t *source_words = (const uint32_t *)source;
    volatile uint32_t *destination_words;

    if (!source || (physical & 0x3u) != 0u || (length & 0x3u) != 0u) {
        return XST_INVALID_PARAM;
    }
    destination = bitnet_plddr_pointer(physical, length);
    if (!destination) {
        return XST_FAILURE;
    }
    if (plddr_fast_copy) {
        /* Optional fast path: let libc issue paired/SIMD stores.  The
         * default remains the conservative word-at-a-time path until the
         * board-specific /dev/mem mapping has passed the A/B regression. */
        memcpy(destination, source, length);
    } else {
        destination_words = (volatile uint32_t *)destination;
        for (size_t i = 0; i < length / sizeof(uint32_t); ++i) {
            destination_words[i] = source_words[i];
        }
    }
    __sync_synchronize();
    return XST_SUCCESS;
}

uint32_t bitnet_plddr_status(void) {
    volatile uint32_t *gpio_data =
        (volatile uint32_t *)(uintptr_t)XPAR_PLDDR_STATUS_GPIO_BASEADDR;

    if (!plddr_mapped) {
        return 0u;
    }
    __sync_synchronize();
    return *gpio_data;
}

int bitnet_psddr_cache_enabled(void) {
    return psddr_cache_active && psddr_cache_mapping != NULL;
}

void *bitnet_psddr_cache_pointer(uint64_t physical, size_t length) {
    uint64_t offset;

    if (!bitnet_psddr_cache_enabled() ||
        physical < XPAR_PS_DDR_CACHE_BASEADDR) {
        return NULL;
    }
    offset = physical - XPAR_PS_DDR_CACHE_BASEADDR;
    if (length > XPAR_PS_DDR_CACHE_SIZE ||
        offset > XPAR_PS_DDR_CACHE_SIZE - length) {
        return NULL;
    }
    return (uint8_t *)psddr_cache_mapping + offset;
}

int bitnet_psddr_cache_copy_to(uint64_t physical, const void *source,
                               size_t length) {
    void *destination;

    if (!source || (physical & 0x3u) != 0u || (length & 0x3u) != 0u) {
        return XST_INVALID_PARAM;
    }
    destination = bitnet_psddr_cache_pointer(physical, length);
    if (!destination) {
        return XST_FAILURE;
    }
    memcpy(destination, source, length);
    __sync_synchronize();
    return XST_SUCCESS;
}

static inline volatile uint32_t *reg_pointer(uintptr_t base, uint32_t offset) {
    return (volatile uint32_t *)(base + offset);
}

/*
 * /dev/mem maps the reserved DMA bounce buffers with a device memory type on
 * arm64.  libc memcpy may issue unaligned 64-bit or SIMD accesses, which raise
 * SIGBUS (BUS_ADRALN) on that mapping.  Keep every access to the device-mapped
 * side explicitly volatile and 32-bit wide.  DMA payload addresses and sizes
 * are required to be word-aligned by the accelerator protocol.
 */
static int dma_payload_is_word_aligned(uintptr_t address, uint32_t length) {
    return ((address | (uintptr_t)length) & (sizeof(uint32_t) - 1u)) == 0u;
}

static int address_range_is_within(uintptr_t address, uint32_t length,
                                   uintptr_t base, uint64_t bytes) {
    uint64_t offset;

    if (address < base) {
        return 0;
    }
    offset = (uint64_t)(address - base);
    return offset <= bytes && (uint64_t)length <= (bytes - offset);
}

static int direct_dma_physical_address(uintptr_t address, uint32_t length,
                                       uint64_t *physical) {
    if (!direct_dma_enabled) {
        return 0;
    }
    if (address_range_is_within(address, length, BITNET_DMA_DIRECT_VIRT,
                                BITNET_DMA_DIRECT_BYTES)) {
        *physical = direct_dma_phys_base +
            (uint64_t)(address - BITNET_DMA_DIRECT_VIRT);
        return 1;
    }
    if (address_range_is_within(address, length, BITNET_DMA_PREFILL_VIRT,
                                BITNET_DMA_PREFILL_BYTES)) {
        *physical = direct_dma_phys_base + BITNET_DMA_PREFILL_OFFSET +
            (uint64_t)(address - BITNET_DMA_PREFILL_VIRT);
        return 1;
    }
    return 0;
}

static void copy_words_to_dma(uintptr_t dma_address, uintptr_t source_address,
                              uint32_t length) {
    volatile uint32_t *destination =
        (volatile uint32_t *)(uintptr_t)dma_address;
    const uint32_t *source = (const uint32_t *)(uintptr_t)source_address;
    uint32_t word_count = length / (uint32_t)sizeof(uint32_t);

    for (uint32_t i = 0; i < word_count; ++i) {
        destination[i] = source[i];
    }
}

static void copy_words_from_dma(uintptr_t destination_address,
                                uintptr_t dma_address, uint32_t length) {
    uint32_t *destination = (uint32_t *)(uintptr_t)destination_address;
    const volatile uint32_t *source =
        (const volatile uint32_t *)(uintptr_t)dma_address;
    uint32_t word_count = length / (uint32_t)sizeof(uint32_t);

    for (uint32_t i = 0; i < word_count; ++i) {
        destination[i] = source[i];
    }
}

uint32_t XAxiDma_ReadReg(uintptr_t base, uint32_t offset) {
    uint32_t value = *reg_pointer(base, offset);
    __sync_synchronize();
    return value;
}

void XAxiDma_WriteReg(uintptr_t base, uint32_t offset, uint32_t value) {
    *reg_pointer(base, offset) = value;
    __sync_synchronize();
}

XAxiDma_Config *XAxiDma_LookupConfig(uint16_t device_id) {
    static XAxiDma_Config configs[] = {
        {XPAR_AXI_DMA_0_DEVICE_ID, XPAR_AXI_DMA_0_BASEADDR, 0, 1},
        {XPAR_AXI_DMA_1_DEVICE_ID, XPAR_AXI_DMA_1_BASEADDR, 0, 0},
    };
    if (device_id >= (sizeof(configs) / sizeof(configs[0]))) {
        return NULL;
    }
    if (device_id == XPAR_AXI_DMA_1_DEVICE_ID && !dma1_regs_mapped) {
        return NULL;
    }
    return &configs[device_id];
}

static int reset_channel(uintptr_t channel_base) {
    XAxiDma_WriteReg(channel_base, XAXIDMA_CR_OFFSET, XAXIDMA_CR_RESET_MASK);
    for (unsigned i = 0; i < 1000000u; ++i) {
        if ((XAxiDma_ReadReg(channel_base, XAXIDMA_CR_OFFSET) &
             XAXIDMA_CR_RESET_MASK) == 0u) {
            XAxiDma_WriteReg(channel_base, XAXIDMA_CR_OFFSET,
                             XAXIDMA_CR_RUNSTOP_MASK);
            return XST_SUCCESS;
        }
    }
    return XST_FAILURE;
}

int XAxiDma_CfgInitialize(XAxiDma *instance, XAxiDma_Config *config) {
    if (!instance || !config || devmem_fd < 0) {
        return XST_INVALID_PARAM;
    }
    instance->RegBase = config->BaseAddr;
    instance->HasS2mm = config->HasS2mm;
    if (reset_channel(instance->RegBase + XAXIDMA_TX_OFFSET) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (instance->HasS2mm &&
        reset_channel(instance->RegBase + XAXIDMA_RX_OFFSET) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    /*
     * AXI DMA reset is core-wide even when it is requested through one
     * channel's control register.  Resetting RX above therefore clears the
     * RUNSTOP bit that reset_channel() just set for TX.  Start both channels
     * only after the final reset has completed.
     */
    XAxiDma_WriteReg(instance->RegBase + XAXIDMA_TX_OFFSET,
                     XAXIDMA_CR_OFFSET, XAXIDMA_CR_RUNSTOP_MASK);
    if (instance->HasS2mm) {
        XAxiDma_WriteReg(instance->RegBase + XAXIDMA_RX_OFFSET,
                         XAXIDMA_CR_OFFSET, XAXIDMA_CR_RUNSTOP_MASK);
    }
    return XST_SUCCESS;
}

int XAxiDma_HasSg(XAxiDma *instance) {
    (void)instance;
    return 0;
}

int XAxiDma_SimpleTransfer(XAxiDma *instance, uintptr_t address,
                           uint32_t length, int direction) {
    uintptr_t channel;
    uint64_t physical;

    if (!instance || !address || !length ||
        !dma_payload_is_word_aligned(address, length)) {
        return XST_INVALID_PARAM;
    }
    if (direction == XAXIDMA_DMA_TO_DEVICE) {
        if (length > BITNET_DMA_MAX_TRANSFER_BYTES) {
            return XST_BUFFER_TOO_SMALL;
        }
        if (!direct_dma_physical_address(address, length, &physical)) {
            if (length > BITNET_DMA_TX_BYTES) {
                return XST_BUFFER_TOO_SMALL;
            }
            copy_words_to_dma(BITNET_DMA_TX_PHYS, address, length);
            physical = BITNET_DMA_TX_PHYS;
            bounce_tx_bytes += length;
        } else {
            direct_tx_bytes += length;
        }
        channel = instance->RegBase + XAXIDMA_TX_OFFSET;
        __sync_synchronize();
        XAxiDma_WriteReg(channel, XAXIDMA_SRCADDR_OFFSET, (uint32_t)physical);
        XAxiDma_WriteReg(channel, XAXIDMA_SRCADDR_MSB_OFFSET,
                         (uint32_t)(physical >> 32));
    } else {
        if (length > BITNET_DMA_RX_BYTES ||
            length > BITNET_DMA_MAX_TRANSFER_BYTES) {
            return XST_BUFFER_TOO_SMALL;
        }
        if (direct_dma_physical_address(address, length, &physical)) {
            if (direct_dma_end_cpu_access() != 0) {
                return XST_FAILURE;
            }
            rx_direct_pending = 1;
            rx_copy_pending = 0;
            direct_rx_bytes += length;
        } else {
            physical = BITNET_DMA_RX_PHYS;
            rx_direct_pending = 0;
            rx_copy_pending = 1;
            rx_destination = (void *)address;
            rx_length = length;
            bounce_rx_bytes += length;
        }
        channel = instance->RegBase + XAXIDMA_RX_OFFSET;
        rx_submit_ns = monotonic_ns();
        XAxiDma_WriteReg(channel, XAXIDMA_DESTADDR_OFFSET, (uint32_t)physical);
        XAxiDma_WriteReg(channel, XAXIDMA_DESTADDR_MSB_OFFSET,
                         (uint32_t)(physical >> 32));
    }
    XAxiDma_WriteReg(channel, XAXIDMA_CR_OFFSET,
                     XAxiDma_ReadReg(channel, XAXIDMA_CR_OFFSET) |
                     XAXIDMA_CR_RUNSTOP_MASK);
    XAxiDma_WriteReg(channel, XAXIDMA_BUFFLEN_OFFSET, length);
    return XST_SUCCESS;
}

int XAxiDma_SimpleTransferPhysical(XAxiDma *instance, uint64_t physical,
                                   uint32_t length, int direction) {
    uintptr_t channel;

    if (!instance || !physical || !length ||
        !dma_payload_is_word_aligned((uintptr_t)physical, length) ||
        length > BITNET_DMA_MAX_TRANSFER_BYTES) {
        return XST_INVALID_PARAM;
    }
    if (direction == XAXIDMA_DEVICE_TO_DMA) {
        if (!instance->HasS2mm || length > BITNET_DMA_RX_BYTES) {
            return XST_INVALID_PARAM;
        }
        channel = instance->RegBase + XAXIDMA_RX_OFFSET;
        XAxiDma_WriteReg(channel, XAXIDMA_DESTADDR_OFFSET,
                         (uint32_t)physical);
        XAxiDma_WriteReg(channel, XAXIDMA_DESTADDR_MSB_OFFSET,
                         (uint32_t)(physical >> 32));
    } else {
        channel = instance->RegBase + XAXIDMA_TX_OFFSET;
        physical_tx_bytes += length;
        ++physical_tx_transfers;
        if (instance->RegBase == XPAR_AXI_DMA_0_BASEADDR) {
            tx0_submit_ns = monotonic_ns();
        } else if (instance->RegBase == XPAR_AXI_DMA_1_BASEADDR) {
            tx1_submit_ns = monotonic_ns();
        }
        XAxiDma_WriteReg(channel, XAXIDMA_SRCADDR_OFFSET,
                         (uint32_t)physical);
        XAxiDma_WriteReg(channel, XAXIDMA_SRCADDR_MSB_OFFSET,
                         (uint32_t)(physical >> 32));
    }
    XAxiDma_WriteReg(channel, XAXIDMA_CR_OFFSET,
                     XAxiDma_ReadReg(channel, XAXIDMA_CR_OFFSET) |
                     XAXIDMA_CR_RUNSTOP_MASK);
    XAxiDma_WriteReg(channel, XAXIDMA_BUFFLEN_OFFSET, length);
    return XST_SUCCESS;
}

int XAxiDma_SimpleTransferConcat(XAxiDma *instance,
                                 uintptr_t first_address, uint32_t first_length,
                                 uintptr_t second_address, uint32_t second_length) {
    uintptr_t channel;
    uint32_t total_length;

    if (!instance || !first_address || !second_address ||
        !first_length || !second_length ||
        !dma_payload_is_word_aligned(first_address, first_length) ||
        !dma_payload_is_word_aligned(second_address, second_length) ||
        first_length > BITNET_DMA_TX_BYTES ||
        second_length > (BITNET_DMA_TX_BYTES - first_length) ||
        (first_length + second_length) > BITNET_DMA_MAX_TRANSFER_BYTES) {
        return XST_INVALID_PARAM;
    }

    total_length = first_length + second_length;
    copy_words_to_dma(BITNET_DMA_TX_PHYS, first_address, first_length);
    copy_words_to_dma(BITNET_DMA_TX_PHYS + first_length,
                      second_address, second_length);
    __sync_synchronize();

    channel = instance->RegBase + XAXIDMA_TX_OFFSET;
    XAxiDma_WriteReg(channel, XAXIDMA_CR_OFFSET,
                     XAxiDma_ReadReg(channel, XAXIDMA_CR_OFFSET) |
                     XAXIDMA_CR_RUNSTOP_MASK);
    XAxiDma_WriteReg(channel, XAXIDMA_SRCADDR_OFFSET,
                     (uint32_t)BITNET_DMA_TX_PHYS);
    XAxiDma_WriteReg(channel, XAXIDMA_SRCADDR_MSB_OFFSET,
                     (uint32_t)(BITNET_DMA_TX_PHYS >> 32));
    XAxiDma_WriteReg(channel, XAXIDMA_BUFFLEN_OFFSET, total_length);
    return XST_SUCCESS;
}

int XAxiDma_Busy(XAxiDma *instance, int direction) {
    uintptr_t channel = instance->RegBase +
        ((direction == XAXIDMA_DMA_TO_DEVICE) ? XAXIDMA_TX_OFFSET : XAXIDMA_RX_OFFSET);
    uint32_t status = XAxiDma_ReadReg(channel, XAXIDMA_SR_OFFSET);
    int busy = ((status & XAXIDMA_IDLE_MASK) == 0u);

    if (direction == XAXIDMA_DMA_TO_DEVICE) {
        ++tx_busy_polls;
    } else {
        ++rx_busy_polls;
    }

    if (!busy && direction == XAXIDMA_DEVICE_TO_DMA &&
        (rx_direct_pending || rx_copy_pending) && rx_submit_ns != 0u) {
        uint64_t completed_ns = monotonic_ns();

        if (completed_ns >= rx_submit_ns) {
            rx_dma_elapsed_ns += completed_ns - rx_submit_ns;
        }
        ++rx_dma_transfers;
        rx_submit_ns = 0u;
    }

    if (!busy && direction == XAXIDMA_DMA_TO_DEVICE) {
        uint64_t submitted_ns = 0u;
        if (instance->RegBase == XPAR_AXI_DMA_0_BASEADDR) {
            submitted_ns = tx0_submit_ns;
        } else if (instance->RegBase == XPAR_AXI_DMA_1_BASEADDR) {
            submitted_ns = tx1_submit_ns;
        }
        if (submitted_ns != 0u) {
            uint64_t completed_ns = monotonic_ns();
            if (completed_ns >= submitted_ns) {
                if (instance->RegBase == XPAR_AXI_DMA_0_BASEADDR) {
                    tx0_dma_elapsed_ns += completed_ns - submitted_ns;
                    ++tx0_dma_transfers;
                } else if (instance->RegBase == XPAR_AXI_DMA_1_BASEADDR) {
                    tx1_dma_elapsed_ns += completed_ns - submitted_ns;
                    ++tx1_dma_transfers;
                }
            }
            if (instance->RegBase == XPAR_AXI_DMA_0_BASEADDR) {
                tx0_submit_ns = 0u;
            } else if (instance->RegBase == XPAR_AXI_DMA_1_BASEADDR) {
                tx1_submit_ns = 0u;
            }
        }
    }

    if (!busy && direction == XAXIDMA_DEVICE_TO_DMA && rx_direct_pending) {
        __sync_synchronize();
        if (direct_dma_begin_cpu_access() != 0) {
            return busy;
        }
        /* The destination is the caller's output buffer itself.  No CPU
         * bounce/copy is needed; cache ownership is the only hand-off. */
        rx_direct_pending = 0;
    } else if (!busy && direction == XAXIDMA_DEVICE_TO_DMA && rx_copy_pending) {
        __sync_synchronize();
        copy_words_from_dma((uintptr_t)rx_destination,
                            BITNET_DMA_RX_PHYS, rx_length);
        rx_copy_pending = 0;
    }
    return busy;
}

static int translate_path(const char *fat_path, char *output, size_t output_size) {
    const char *suffix = fat_path;
    if (strncmp(suffix, "0:/", 3) == 0) {
        suffix += 2;
    }
    return snprintf(output, output_size, "%s%s", model_root_path, suffix) <
           (int)output_size ? 0 : -1;
}

FRESULT f_mount(FATFS *fs, const char *path, int mount_now) {
    (void)fs; (void)path; (void)mount_now;
    return FR_OK;
}

FRESULT f_open(FIL *file, const char *path, unsigned mode) {
    char translated[PATH_MAX];
    const char *open_mode = (mode & FA_CREATE_ALWAYS) ? "wb" :
                            ((mode & FA_WRITE) ? "r+b" : "rb");
    if (translate_path(path, translated, sizeof(translated)) != 0) {
        return FR_NO_FILE;
    }
    file->fp = fopen(translated, open_mode);
    if (!file->fp) {
        return FR_NO_FILE;
    }
    if (fseeko(file->fp, 0, SEEK_END) != 0) {
        fclose(file->fp); file->fp = NULL; return FR_DISK_ERR;
    }
    file->size = (FSIZE_t)ftello(file->fp);
    fseeko(file->fp, 0, SEEK_SET);
    return FR_OK;
}

FRESULT f_close(FIL *file) {
    int result = file->fp ? fclose(file->fp) : 0;
    file->fp = NULL;
    return result == 0 ? FR_OK : FR_DISK_ERR;
}

FRESULT f_read(FIL *file, void *buffer, UINT bytes, UINT *read_bytes) {
    uint64_t direct_physical;

    if (direct_dma_physical_address((uintptr_t)buffer, bytes,
                                    &direct_physical)) {
        size_t count;

        if (direct_dma_begin_cpu_access() != 0) {
            return FR_DISK_ERR;
        }
        count = fread(buffer, 1, bytes, file->fp);
        if (direct_dma_end_cpu_access() != 0) {
            return FR_DISK_ERR;
        }
        if (read_bytes) *read_bytes = (UINT)count;
        return ferror(file->fp) ? FR_DISK_ERR : FR_OK;
    }

    size_t count = fread(buffer, 1, bytes, file->fp);
    if (read_bytes) *read_bytes = (UINT)count;
    return ferror(file->fp) ? FR_DISK_ERR : FR_OK;
}

FRESULT f_write(FIL *file, const void *buffer, UINT bytes, UINT *written_bytes) {
    size_t count = fwrite(buffer, 1, bytes, file->fp);
    if (written_bytes) *written_bytes = (UINT)count;
    return count == bytes ? FR_OK : FR_DISK_ERR;
}

FRESULT f_lseek(FIL *file, FSIZE_t offset) {
    return fseeko(file->fp, (off_t)offset, SEEK_SET) == 0 ? FR_OK : FR_DISK_ERR;
}

FRESULT f_mkdir(const char *path) {
    char translated[PATH_MAX];
    if (translate_path(path, translated, sizeof(translated)) != 0) return FR_DISK_ERR;
    if (mkdir(translated, 0755) == 0) return FR_OK;
    return errno == EEXIST ? FR_EXIST : FR_DISK_ERR;
}

FRESULT f_mkfs(const char *path, unsigned option, unsigned allocation_unit,
               void *work, UINT work_bytes) {
    (void)path; (void)option; (void)allocation_unit; (void)work; (void)work_bytes;
    return FR_DISK_ERR;
}

char inbyte(void) {
    int value = getchar();
    return value == EOF ? '\0' : (char)value;
}

void outbyte(char value) {
    putchar((unsigned char)value);
    fflush(stdout);
}
