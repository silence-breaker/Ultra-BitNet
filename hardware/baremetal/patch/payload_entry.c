#include <stdint.h>

extern uint8_t __payload_bss_start[];
extern uint8_t __payload_bss_end[];
extern int bitnet_payload_main(void);

__attribute__((section(".text.payload_entry"), noreturn))
void bitnet_payload_entry(void) {
    for (uint8_t *cursor = __payload_bss_start;
         cursor < __payload_bss_end; cursor++) {
        *cursor = 0u;
    }

    (void)bitnet_payload_main();
    for (;;) {
        __asm__ volatile("wfe");
    }
}
