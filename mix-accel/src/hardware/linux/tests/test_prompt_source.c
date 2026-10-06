#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "prompt_source.h"

#define TEST_CAPACITY 2048u

static FILE *make_stream(size_t character_count, const char *suffix) {
    FILE *stream = tmpfile();

    assert(stream != NULL);
    for (size_t i = 0u; i < character_count; i++) {
        fputc('A', stream);
    }
    fputs(suffix, stream);
    rewind(stream);
    return stream;
}

int main(void) {
    char prompt[TEST_CAPACITY + 1u];
    size_t length;
    FILE *stream;

    stream = make_stream(TEST_CAPACITY, "\n");
    assert(bitnet_prompt_read_line(stream, prompt, TEST_CAPACITY, &length) == 1);
    assert(length == TEST_CAPACITY);
    fclose(stream);

    stream = make_stream(TEST_CAPACITY + 1u, "\nnext\r\n");
    assert(bitnet_prompt_read_line(stream, prompt, TEST_CAPACITY, &length) == -1);
    assert(bitnet_prompt_read_line(stream, prompt, TEST_CAPACITY, &length) == 1);
    assert(length == 4u && strcmp(prompt, "next") == 0);
    assert(bitnet_prompt_read_line(stream, prompt, TEST_CAPACITY, &length) == 0);
    fclose(stream);

    stream = make_stream(3u, "\r\n");
    assert(bitnet_prompt_read_line(stream, prompt, TEST_CAPACITY, &length) == 1);
    assert(length == 3u && memcmp(prompt, "AAA", 3u) == 0);
    fclose(stream);

    puts("PROMPT_SOURCE_TEST_PASS");
    return 0;
}
