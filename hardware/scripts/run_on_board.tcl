set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".."]]
set bit_file [file normalize [file join $root_dir "build" "vivado" "axu3egb_bitnet_accel.runs" "impl_1" "design_1_wrapper.bit"]]
set elf_file [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "Debug" "bitnet_accel_smoke.elf"]]
set psu_init_tcl [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "_ide" "psinit" "psu_init.tcl"]]
set infer_control_base 0x4B000000
set emmc_control_base 0x4D000000
set layer_control_base 0x4E000000
set control_base 0x4F000000
set control_words 8
set status_base 0x70000000
set status_words 32
set status_magic 0x42545354

foreach required_file [list $bit_file $elf_file $psu_init_tcl] {
    if { ![file exists $required_file] } {
        error "Missing required board-run file: $required_file"
    }
}

proc read_status_word {base index} {
    set addr [expr {$base + ($index * 4)}]
    set value [mrd -value $addr]
    if {[llength $value] > 1} {
        set value [lindex $value 0]
    }
    return [expr {$value & 0xFFFFFFFF}]
}

proc dump_status_block {base words} {
    puts "DDR status block:"
    for {set row 0} {$row < $words} {incr row 4} {
        set addr [expr {$base + ($row * 4)}]
        set v0 [read_status_word $base [expr {$row + 0}]]
        set v1 [read_status_word $base [expr {$row + 1}]]
        set v2 [read_status_word $base [expr {$row + 2}]]
        set v3 [read_status_word $base [expr {$row + 3}]]
        puts [format "  0x%08X: %08X %08X %08X %08X" $addr $v0 $v1 $v2 $v3]
    }
}

proc clear_status_block {base words} {
    for {set index 0} {$index < $words} {incr index} {
        set addr [expr {$base + ($index * 4)}]
        mwr $addr 0
    }
}

connect -url tcp:127.0.0.1:3121
puts "Connected targets:"
targets

puts "Resetting PS and initializing ZynqMP..."
targets -set -filter {name =~ "PSU"}
rst -system
after 3000
targets -set -filter {name =~ "PSU"}
source $psu_init_tcl
psu_init

puts "Programming PL bitstream: $bit_file"
targets -set -filter {name =~ "PL"}
fpga -file $bit_file

puts "Removing PS-PL isolation and configuring resets..."
targets -set -filter {name =~ "PSU"}
psu_ps_pl_isolation_removal
psu_ps_pl_reset_config

puts "Downloading ELF to Cortex-A53 #0: $elf_file"
targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor
puts "Clearing inference control block at 0x[format %08X $infer_control_base]"
clear_status_block $infer_control_base $control_words
puts "Clearing eMMC layer-chain control block at 0x[format %08X $emmc_control_base]"
clear_status_block $emmc_control_base $control_words
puts "Clearing layer-chain control block at 0x[format %08X $layer_control_base]"
clear_status_block $layer_control_base $control_words
puts "Clearing real-tensor control block at 0x[format %08X $control_base]"
clear_status_block $control_base $control_words
puts "Clearing DDR status block at 0x[format %08X $status_base]"
clear_status_block $status_base $status_words
dow $elf_file
con

puts "Board smoke test started on Cortex-A53 #0."
puts "Waiting for smoke test status block..."
set result 0
for {set poll 0} {$poll < 30} {incr poll} {
    after 1000
    set magic [read_status_word $status_base 0]
    set result [read_status_word $status_base 5]
    if {$magic == $status_magic && $result != 0} {
        break
    }
}
catch { stop }

set magic [read_status_word $status_base 0]
set stage [read_status_word $status_base 4]
set result [read_status_word $status_base 5]
set status_code [read_status_word $status_base 6]
set mismatches [read_status_word $status_base 12]
set first_mismatch [read_status_word $status_base 13]
dump_status_block $status_base $status_words

if {$magic != $status_magic} {
    error [format "BOARD_SMOKE_TIMEOUT: status magic is 0x%08X, expected 0x%08X" $magic $status_magic]
}

if {$result == 1} {
    puts "BOARD_SMOKE_PASS"
} elseif {$result == 0} {
    error [format "BOARD_SMOKE_TIMEOUT: stage=%u status=0x%08X" $stage $status_code]
} else {
    error [format "BOARD_SMOKE_FAIL: stage=%u status=0x%08X mismatches=%u first_mismatch=%u" \
        $stage $status_code $mismatches $first_mismatch]
}
