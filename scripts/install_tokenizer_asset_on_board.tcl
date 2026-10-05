set tokenizer_file ""
if { $argc >= 1 } {
    set tokenizer_file [file normalize [lindex $argv 0]]
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set elf_file [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "Debug" "bitnet_accel_smoke.elf"]]
set psu_init_tcl [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "_ide" "psinit" "psu_init.tcl"]]
set work_dir [file normalize [file join $root_dir "build" "emmc_install"]]

if { $tokenizer_file eq "" } {
    set tokenizer_file [file normalize [file join $root_dir "build" "emmc" "BITNET" "TOK" "TRIE.BIN"]]
}

set emmc_path "0:/BITNET/TOK/TRIE.BIN"
set infer_control_base 0x4B000000
set install_control_base 0x4C000000
set emmc_control_base 0x4D000000
set layer_control_base 0x4E000000
set real_control_base 0x4F000000
set staging_data_base 0x52000000
set install_control_words 64
set control_words 16
set status_base 0x70000000
set status_words 48
set status_magic 0x42545354
set install_magic 0x42494E31
set install_version 1
set install_op_write_file 1
set poll_timeout_ms 180000
set path_field_bytes 128

foreach required_file [list $elf_file $psu_init_tcl $tokenizer_file] {
    if { ![file exists $required_file] } {
        error "Missing required tokenizer install file: $required_file"
    }
}
file mkdir $work_dir

proc write_install_control {control_file emmc_path file_size} {
    global install_magic install_version install_op_write_file staging_data_base path_field_bytes

    if {[string length $emmc_path] >= $path_field_bytes} {
        error "eMMC path is too long: $emmc_path"
    }

    set path_bytes [encoding convertto ascii $emmc_path]
    append path_bytes [string repeat "\x00" [expr {$path_field_bytes - [string length $path_bytes]}]]

    set fh [open $control_file "wb"]
    fconfigure $fh -translation binary -encoding binary
    puts -nonewline $fh [binary format "iiiiiiii" \
        $install_magic \
        $install_version \
        $install_op_write_file \
        $staging_data_base \
        $file_size \
        0 \
        1 \
        1]
    puts -nonewline $fh $path_bytes
    close $fh
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

set file_size [file size $tokenizer_file]
set control_file [file normalize [file join $work_dir "install_tokenizer.ctrl.bin"]]
write_install_control $control_file $emmc_path $file_size

puts "Installing tokenizer trie to eMMC:"
puts "  host file: $tokenizer_file"
puts "  eMMC path: $emmc_path"
puts "  bytes:     $file_size"

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

targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor
after 100

clear_words $infer_control_base $control_words
clear_words $install_control_base $install_control_words
clear_words $emmc_control_base $control_words
clear_words $layer_control_base $control_words
clear_words $real_control_base $control_words
clear_words $status_base $status_words

puts [format "Staging tokenizer -> 0x%08X" $staging_data_base]
dow -data $tokenizer_file $staging_data_base

puts [format "Control -> 0x%08X" $install_control_base]
dow -data $control_file $install_control_base

puts "Starting installer ELF"
dow $elf_file
con

wait_for_result $status_base $status_magic $poll_timeout_ms
catch { stop }

set magic [read_status_word $status_base 0]
set stage [read_status_word $status_base 4]
set result [read_status_word $status_base 5]
set status_code [read_status_word $status_base 6]
set bytes_written [read_status_word $status_base 7]

if {$magic != $status_magic} {
    dump_status_block $status_base $status_words
    error [format "BOARD_TOKENIZER_INSTALL_TIMEOUT: status magic is 0x%08X, expected 0x%08X" $magic $status_magic]
}

if {$result == 1} {
    puts [format "BOARD_TOKENIZER_INSTALL_PASS: bytes=%u" $bytes_written]
} elseif {$result == 0} {
    dump_status_block $status_base $status_words
    error [format "BOARD_TOKENIZER_INSTALL_TIMEOUT: stage=%u status=0x%08X bytes=%u" \
        $stage $status_code $bytes_written]
} else {
    dump_status_block $status_base $status_words
    error [format "BOARD_TOKENIZER_INSTALL_FAIL: stage=%u status=0x%08X bytes=%u" \
        $stage $status_code $bytes_written]
}
