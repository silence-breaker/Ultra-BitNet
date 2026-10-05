#ifndef BITNET_LINUX_XIL_PRINTF_H
#define BITNET_LINUX_XIL_PRINTF_H

#include <stdio.h>
#define xil_printf(...) printf(__VA_ARGS__)

char inbyte(void);
void outbyte(char value);

#endif
