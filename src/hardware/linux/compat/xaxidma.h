#ifndef BITNET_LINUX_XAXIDMA_H
#define BITNET_LINUX_XAXIDMA_H

#include <stddef.h>
#include <stdint.h>
#include "xil_types.h"
#include "xstatus.h"

#define XAXIDMA_DMA_TO_DEVICE 0
#define XAXIDMA_DEVICE_TO_DMA 1

typedef struct {
    uint16_t DeviceId;
    uintptr_t BaseAddr;
    int HasSg;
    int HasS2mm;
} XAxiDma_Config;

typedef struct {
    uintptr_t RegBase;
    int HasS2mm;
} XAxiDma;

int bitnet_linux_runtime_init(const char *model_root);
void bitnet_linux_runtime_close(void);
const char *bitnet_linux_model_root(void);
void *bitnet_plddr_pointer(uint64_t physical, size_t length);
int bitnet_plddr_copy_to(uint64_t physical, const void *source, size_t length);
uint32_t bitnet_plddr_status(void);
int bitnet_psddr_cache_enabled(void);
void *bitnet_psddr_cache_pointer(uint64_t physical, size_t length);
int bitnet_psddr_cache_copy_to(uint64_t physical, const void *source,
                               size_t length);

XAxiDma_Config *XAxiDma_LookupConfig(uint16_t device_id);
int XAxiDma_CfgInitialize(XAxiDma *instance, XAxiDma_Config *config);
int XAxiDma_HasSg(XAxiDma *instance);
int XAxiDma_SimpleTransfer(XAxiDma *instance, uintptr_t address,
                           uint32_t length, int direction);
/* Candidate dual-lane path: submit an already DMA-visible physical address.
 * This is required for buffers allocated from the PL DDR4 aperture. */
int XAxiDma_SimpleTransferPhysical(XAxiDma *instance, uint64_t physical,
                                   uint32_t length, int direction);
int XAxiDma_SimpleTransferConcat(XAxiDma *instance,
                                 uintptr_t first_address, uint32_t first_length,
                                 uintptr_t second_address, uint32_t second_length);
int XAxiDma_Busy(XAxiDma *instance, int direction);
uint32_t XAxiDma_ReadReg(uintptr_t base, uint32_t offset);
void XAxiDma_WriteReg(uintptr_t base, uint32_t offset, uint32_t value);

#endif
