ENTRY(bitnet_payload_entry)

SECTIONS
{
    . = 0x00400000;

    .text : ALIGN(64)
    {
        KEEP(*(.text.payload_entry))
        *(.text .text.*)
    }

    .rodata : ALIGN(64)
    {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(64)
    {
        *(.data .data.*)
    }

    . = ALIGN(64);
    __payload_bss_start = .;
    .bss (NOLOAD) : ALIGN(64)
    {
        *(.bss .bss.*)
        *(COMMON)
    }
    __payload_bss_end = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
