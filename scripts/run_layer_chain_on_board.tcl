set layer_count 1
set first_layer 0
set layer_count_from_arg 0
set first_layer_from_arg 0
if { $argc >= 1 } {
    set layer_count [lindex $argv 0]
    set layer_count_from_arg 1
}
if { $argc >= 2 } {
    set first_layer [lindex $argv 1]
    set first_layer_from_arg 1
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set bit_file [file normalize [file join $root_dir "build" "vivado" "axu3egb_bitnet_accel.runs" "impl_1" "design_1_wrapper.bit"]]
set elf_file [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "Debug" "bitnet_accel_smoke.elf"]]
set psu_init_tcl [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "_ide" "psinit" "psu_init.tcl"]]
set model_dir [file normalize [file join $root_dir "build" "model"]]
set control_file [file normalize [file join $model_dir "layer_chain.ctrl.bin"]]
set run_control_file [file normalize [file join $model_dir "layer_chain.run.ctrl.bin"]]

set infer_control_base 0x4B000000
set emmc_control_base 0x4D000000
set layer_control_base 0x4E000000
set real_control_base 0x4F000000
set control_words 8
set weight_base 0x56000000
set layer_stride 0x02000000
set status_base 0x70000000
set status_words 32
set status_magic 0x42545354
set poll_timeout_ms 180000
set layer_control_magic 0x424C4331

set tensors {
    {q_proj    0x000000}
    {k_proj    0x200000}
    {v_proj    0x300000}
    {o_proj    0x400000}
    {gate_proj 0x600000}
    {up_proj   0xB00000}
    {down_proj 0x1000000}
}

proc read_layer_control_words {control_file} {
    set fh [open $control_file "rb"]
    fconfigure $fh -translation binary -encoding binary
    set bytes [read $fh 32]
    close $fh
    if {[string length $bytes] != 32} {
        error "Layer-chain control block is too small: $control_file"
    }
    binary scan $bytes "iiiiiiii" magic first layers seed flags r0 r1 r2
    return [list $magic $first $layers $seed $flags $r0 $r1 $r2]
}

proc write_layer_control_words {control_file words} {
    set fh [open $control_file "wb"]
    fconfigure $fh -translation binary -encoding binary
    puts -nonewline $fh [binary format "iiiiiiii" \
        [lindex $words 0] \
        [lindex $words 1] \
        [lindex $words 2] \
        [lindex $words 3] \
        [lindex $words 4] \
        [lindex $words 5] \
        [lindex $words 6] \
        [lindex $words 7]]
    close $fh
}

foreach required_file [list $bit_file $elf_file $psu_init_tcl $control_file] {
    if { ![file exists $required_file] } {
        error "Missing required layer-chain run file: $required_file"
    }
}

set control_header [read_layer_control_words $control_file]
if { !$first_layer_from_arg } {
    set first_layer [lindex $control_header 1]
}
if { !$layer_count_from_arg } {
    set layer_count [lindex $control_header 2]
}
set control_header [lreplace $control_header 0 2 $layer_control_magic $first_layer $layer_count]
write_layer_control_words $run_control_file $control_header

for {set slot 0} {$slot < $layer_count} {incr slot} {
    set layer [expr {$first_layer + $slot}]
    foreach tensor $tensors {
        set name [lindex $tensor 0]
        set tensor_file [file normalize [file join $model_dir [format "layer%d_%s.fpga.bin" $layer $name]]]
        if { ![file exists $tensor_file] } {
            error "Missing layer tensor file: $tensor_file"
        }
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

proc clear_words {base words} {
    for {set index 0} {$index < $words} {incr index} {
        set addr [expr {$base + ($index * 4)}]
        mwr $addr 0
    }
}

proc wait_for_result {base expected_magic timeout_ms} {
    set deadline [expr {[clock milliseconds] + $timeout_ms}]
    set last_stage 0
    while {[clock milliseconds] < $deadline} {
        set magic [read_status_word $base 0]
        set stage [read_status_word $base 4]
        set result [read_status_word $base 5]
        if {$magic == $expected_magic && $stage != $last_stage} {
            puts [format "  status stage=%u result=0x%08X" $stage $result]
            set last_stage $stage
        }
        if {$magic == $expected_magic && $result != 0} {
            return
        }
        after 500
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

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor

puts "Clearing stale control/status blocks"
clear_words $infer_control_base $control_words
clear_words $emmc_control_base $control_words
clear_words $layer_control_base $control_words
clear_words $real_control_base $control_words
clear_words $status_base $status_words

puts "Downloading layer-chain control block: $run_control_file -> 0x[format %08X $layer_control_base]"
dow -data $run_control_file $layer_control_base

for {set slot 0} {$slot < $layer_count} {incr slot} {
    set layer [expr {$first_layer + $slot}]
    foreach tensor $tensors {
        set name [lindex $tensor 0]
        set offset [lindex $tensor 1]
        set tensor_file [file normalize [file join $model_dir [format "layer%d_%s.fpga.bin" $layer $name]]]
        set addr [expr {$weight_base + ($slot * $layer_stride) + $offset}]
        puts [format "Downloading layer%d %s -> 0x%08X" $layer $name $addr]
        dow -data $tensor_file $addr
    }
}

puts "Downloading ELF to Cortex-A53 #0: $elf_file"
dow $elf_file
con

puts "Layer-chain test started on Cortex-A53 #0."
wait_for_result $status_base $status_magic $poll_timeout_ms
catch { stop }

set magic [read_status_word $status_base 0]
set out_features [read_status_word $status_base 2]
set layer_status [read_status_word $status_base 3]
set stage [read_status_word $status_base 4]
set result [read_status_word $status_base 5]
set status_code [read_status_word $status_base 6]
set tx_words [read_status_word $status_base 7]
set mismatches [read_status_word $status_base 12]
set first_mismatch [read_status_word $status_base 13]
set active_tensor [read_status_word $status_base 14]
set active_layers [read_status_word $status_base 15]
dump_status_block $status_base $status_words

if {$magic != $status_magic} {
    error [format "BOARD_LAYER_CHAIN_TIMEOUT: status magic is 0x%08X, expected 0x%08X" $magic $status_magic]
}

if {$result == 1} {
    puts [format "BOARD_LAYER_CHAIN_PASS: out=%u layers=%u tx_words=%u" $out_features $active_layers $tx_words]
} elseif {$result == 0} {
    error [format "BOARD_LAYER_CHAIN_TIMEOUT: stage=%u status=0x%08X layer_status=%u tensor=%u" \
        $stage $status_code $layer_status $active_tensor]
} else {
    error [format "BOARD_LAYER_CHAIN_FAIL: stage=%u status=0x%08X mismatches=%u first=0x%08X tensor=%u layers=%u" \
        $stage $status_code $mismatches $first_mismatch $active_tensor $active_layers]
}
