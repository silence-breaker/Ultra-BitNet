#ifndef BITNET_PROMPT_SOURCE_H
#define BITNET_PROMPT_SOURCE_H

#include <stddef.h>
#include <stdio.h>

int bitnet_prompt_read_file(const char *path, char *buffer, size_t capacity,
                            size_t *length);
int bitnet_prompt_read_line(FILE *stream, char *buffer, size_t capacity,
                            size_t *length);

#endif
