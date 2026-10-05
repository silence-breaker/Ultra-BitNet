#ifndef ULTRA_BITNET_LINUX_H
#define ULTRA_BITNET_LINUX_H
#include "bitnet.h"
struct bitnet_linux { int fd; void *mapping; size_t length; volatile uint32_t *registers; };
int bitnet_linux_open(struct bitnet_linux *,const char *path,uint64_t offset,struct bitnet_io *);
void bitnet_linux_close(struct bitnet_linux *);
#endif
