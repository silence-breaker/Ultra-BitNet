#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "prompt_source.h"

static void trim_line_ending(char *buffer, size_t *length) {
    while (*length != 0u &&
           (buffer[*length - 1u] == '\n' || buffer[*length - 1u] == '\r')) {
        buffer[--(*length)] = '\0';
    }
}

int bitnet_prompt_read_file(const char *path, char *buffer, size_t capacity,
                            size_t *length) {
    FILE *file;
    size_t count;

    if (!path || !buffer || capacity < 2u || !length) {
        return -1;
    }
    file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "无法打开 prompt 文件 %s: %s\n", path, strerror(errno));
        return -1;
    }
    count = fread(buffer, 1u, capacity, file);
    if (ferror(file)) {
        fprintf(stderr, "读取 prompt 文件 %s 失败\n", path);
        fclose(file);
        return -1;
    }
    if (count == capacity && fgetc(file) != EOF) {
        fprintf(stderr, "prompt 文件超过 %lu 字节\n", (unsigned long)capacity);
        fclose(file);
        return -1;
    }
    fclose(file);
    buffer[count] = '\0';
    *length = count;
    trim_line_ending(buffer, length);
    return *length == 0u ? -1 : 0;
}

int bitnet_prompt_read_line(FILE *stream, char *buffer, size_t capacity,
                            size_t *length) {
    int ch;

    if (!stream || !buffer || capacity < 2u || !length) {
        return -1;
    }
    if (!fgets(buffer, (int)capacity + 1, stream)) {
        return feof(stream) ? 0 : -1;
    }
    *length = strlen(buffer);
    if (*length == capacity && buffer[*length - 1u] != '\n') {
        ch = fgetc(stream);
        if (ch == '\r') {
            ch = fgetc(stream);
        }
        if (ch == '\n' || ch == EOF) {
            return 1;
        }
        while ((ch = fgetc(stream)) != '\n' && ch != EOF) {
        }
        fprintf(stderr, "prompt 行超过 %lu 字节，已丢弃\n",
                (unsigned long)capacity);
        return -1;
    }
    trim_line_ending(buffer, length);
    return 1;
}
