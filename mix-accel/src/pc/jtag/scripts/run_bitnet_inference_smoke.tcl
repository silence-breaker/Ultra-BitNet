set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir .. .. .. ..]]
set elf [expr {[info exists ::env(BITNET_ELF)] ? $::env(BITNET_ELF) : [file join $root_dir build baremetal a.elf]}]
set bitstream [expr {[info exists ::env(BITNET_BITSTREAM)] ? $::env(BITNET_BITSTREAM) : [file join $root_dir build firmware axu3egb_bitnet_accel.bit]}]
set prompt "Analyze traffic event."
set prompt_file ""
set prompt_length [string length $prompt]
set max_new_tokens 1
set input_mode 2
set schema_file ""
if {$argc >= 2} {
    set prompt_file [file normalize [lindex $argv 0]]
    set prompt_length [lindex $argv 1]
}
if {$argc >= 3} { set max_new_tokens [lindex $argv 2] }
if {$argc >= 4} { set input_mode [lindex $argv 3] }
if {$argc >= 5} { set schema_file [file normalize [lindex $argv 4]] }
if {$input_mode == 3 && $prompt_file eq ""} { set prompt_length 24 }

proc write_ascii {addr value} {
    set offset 0
    while {$offset < [string length $value]} {
        set word 0
        for {set i 0} {$i < 4} {incr i} {
            set p [expr {$offset + $i}]
            if {$p < [string length $value]} {
                scan [string index $value $p] %c byte
                set word [expr {$word | (($byte & 0xFF) << (8 * $i))}]
            }
        }
        mwr [expr {$addr + $offset}] $word
        incr offset 4
    }
}

proc read_ascii {addr count} {
    set result ""
    for {set offset 0} {$offset < $count} {incr offset 4} {
        set word [mrd -value [expr {$addr + $offset}]]
        for {set i 0} {$i < 4 && ($offset + $i) < $count} {incr i} {
            append result [format %c [expr {($word >> (8 * $i)) & 0xFF}]]
        }
    }
    return $result
}

proc read_hex {addr count} {
    set result ""
    for {set offset 0} {$offset < $count} {incr offset 4} {
        set word [mrd -value [expr {$addr + $offset}]]
        for {set i 0} {$i < 4 && ($offset + $i) < $count} {incr i} {
            append result [format %02X [expr {($word >> (8 * $i)) & 0xFF}]]
        }
    }
    return $result
}

set hw_server_url [expr {[info exists ::env(BITNET_HW_SERVER)] ? $::env(BITNET_HW_SERVER) : "tcp:127.0.0.1:3121"}]
connect -url $hw_server_url
jtag targets 1
jtag frequency 10000000
set init_marker [file join $root_dir runtime tmp bitnet_ps_initialized.marker]
set initialized [file exists $init_marker]
if {!$initialized} {
    targets -set -filter {name =~ "PSU"}
    rst -system
    after 1000
    set psu_init [expr {[info exists ::env(BITNET_PSU_INIT)] ? $::env(BITNET_PSU_INIT) : [file join $root_dir build firmware psu_init.tcl]}]
    if {![file exists $psu_init]} { error "Missing PS initialization script. Set BITNET_PSU_INIT." }
    source $psu_init
    psu_init
    psu_post_config
    targets -set -filter {name =~ "PL"}
    fpga -file $bitstream
    after 1000
    targets -set -filter {name =~ "PSU"}
    configparams force-mem-accesses 1
    psu_ps_pl_isolation_removal
    init_ps [subst {$psu_afi_config}]
    psu_ps_pl_reset_config
    after 5000
    targets -set -filter {name =~ "Cortex-A53 #0"}
    rst -processor -clear-registers
    file mkdir [file dirname $init_marker]
    set marker [open $init_marker w]
    puts $marker "initialized=[clock seconds]"
    close $marker
} else {
    # Keep the PS/A53 session alive between reports.  Do not reset the
    # system, reinitialize PS/PL, or redownload the bitstream per inference.
    targets -set -filter {name =~ "Cortex-A53 #0"}
}
# Ensure the application processor is out of reset before any mailbox/AP
# transaction.  A visible PSU target alone is insufficient: when A53 is in
# APU reset, mwr to the shared mailbox can fail with AP transaction timeout.
targets -set -filter {name =~ "Cortex-A53 #0"}
if {[catch {con} con_error]} {
    # XSDB reports "Already running" when A53 is already executing.  That is
    # the desired steady-state for mailbox inference, not a fatal error.
    if {[string first "Already running" $con_error] < 0} {
        error $con_error
    }
}
after 500
targets -set -filter {name =~ "PSU"}
configparams force-mem-accesses 1
foreach addr {0x4A000000 0x4B000000 0x4C000000 0x4D000000 0x4E000000 0x64630000 0x70000000} { mwr $addr 0 }
mwr 0x4B020000 0
if {$input_mode == 3 && $prompt_file eq ""} {
    set official_tokens {128000 2127 56956 9629 1567 13}
    set token_offset 0
    foreach token_id $official_tokens {
        mwr [expr {0x4B010000 + $token_offset}] $token_id
        incr token_offset 4
    }
} elseif {$prompt_file eq ""} { write_ascii 0x4B010000 $prompt }
mwr 0x4B000000 0x42494631
mwr 0x4B000004 1
mwr 0x4B000008 $input_mode
mwr 0x4B00000C 0x4B010000
mwr 0x4B000010 $prompt_length
mwr 0x4B000014 0
mwr 0x4B000018 30
mwr 0x4B00001C 1
# Official microsoft/bitnet-b1.58-2B-4T codes are 0=-1, 1=0, 2=+1.
mwr 0x4B000020 1
mwr 0x4B000024 $max_new_tokens
mwr 0x4B000028 0
mwr 0x4B00002C 1
targets -set -filter {name =~ "Cortex-A53 #0"}
if {$prompt_file ne ""} { dow -data $prompt_file 0x4B010000 }
if {$schema_file ne ""} { dow -data $schema_file 0x4B020000 }
dow -force -bypass-cache-sync $elf
con
targets -set -filter {name =~ "PSU"}
configparams force-mem-accesses 1
set done 0
after 3000
for {set poll 0} {$poll < 1800} {incr poll} {
    after 1000
    set magic [mrd -value 0x70000000]
    set stage [mrd -value 0x70000010]
    set result [mrd -value 0x70000014]
    set code [mrd -value 0x70000018]
    if {($poll % 10) == 0} { puts "INFER_PROGRESS stage=$stage result=$result code=$code seconds=$poll" }
    if {$magic == 0x42545354 && $stage == 26 && $result == 1} { set done 1; break }
    if {$magic == 0x42545354 && $result == 0xFFFFFFFF} { error "Inference failed stage=$stage code=$code" }
}
if {!$done} { error "Inference timeout" }
set out_magic [mrd -value 0x64630000]
set out_version [mrd -value 0x64630004]
set byte_count [mrd -value 0x64630008]
set token_count [mrd -value 0x6463000C]
if {$byte_count <= 0 || $byte_count > 8192} { error "Invalid output byte_count=$byte_count" }
set output [read_ascii 0x64630010 $byte_count]
set output_hex [read_hex 0x64630010 $byte_count]
if {[string length $output_hex] != ($byte_count * 2)} { error "Output hex length mismatch bytes=$byte_count hex_chars=[string length $output_hex]" }
puts "INFER_OUTPUT_MAGIC=[format 0x%08X $out_magic]"
puts "INFER_OUTPUT_VERSION=$out_version"
puts "INFER_OUTPUT_BYTES=$byte_count"
puts "INFER_OUTPUT_TOKENS=$token_count"
puts "INFER_OUTPUT_TEXT=$output"
puts "INFER_OUTPUT_HEX=$output_hex"
disconnect
