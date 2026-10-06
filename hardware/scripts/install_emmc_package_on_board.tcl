set package_dir ""
if { $argc >= 1 } {
    set package_dir [file normalize [lindex $argv 0]]
}
set install_mode "full"
if { $argc >= 2 } {
    set install_mode [string tolower [lindex $argv 1]]
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".."]]
set elf_file [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "Debug" "bitnet_accel_smoke.elf"]]
set psu_init_tcl [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "_ide" "psinit" "psu_init.tcl"]]
set work_dir [file normalize [file join $root_dir "build" "emmc_install"]]

if { $package_dir eq "" } {
    set package_dir [file normalize [file join $root_dir "build" "emmc" "BITNET"]]
}

set install_control_base 0x4C000000
set infer_control_base 0x4B000000
set emmc_control_base 0x4D000000
set layer_control_base 0x4E000000
set real_control_base 0x4F000000
set staging_data_base 0x52000000
set staging_data_bytes 0x04000000
set install_control_words 64
set control_words 8
set status_base 0x70000000
set status_words 32
set status_magic 0x42545354
set install_magic 0x42494E31
set install_version 1
set install_op_write_file 1
set poll_timeout_ms 180000
set path_field_bytes 128

set tensor_names {Q.BIN K.BIN V.BIN O.BIN GATE.BIN UP.BIN DOWN.BIN}

foreach required_file [list $elf_file $psu_init_tcl] {
    if { ![file exists $required_file] } {
        error "Missing required eMMC installer file: $required_file"
    }
}
if { ![file isdirectory $package_dir] } {
    error "Missing BITNET package directory: $package_dir"
}
file mkdir $work_dir

proc collect_bitnet_files {package_dir tensor_names install_mode} {
    set files {}

    set manifest_file [file normalize [file join $package_dir "MANIFEST.TXT"]]
    if {($install_mode eq "full") && [file exists $manifest_file]} {
        lappend files [list $manifest_file "0:/BITNET/MANIFEST.TXT"]
    }

    if {($install_mode eq "full") || ($install_mode eq "weights")} {
        foreach layer_dir [lsort [glob -nocomplain -type d [file join $package_dir {L[0-9][0-9]}]]] {
            set layer_name [file tail $layer_dir]
            foreach tensor_name $tensor_names {
                set host_file [file normalize [file join $layer_dir $tensor_name]]
                if { ![file exists $host_file] } {
                    error "Missing tensor file in BITNET package: $host_file"
                }
                set emmc_path [format "0:/BITNET/%s/%s" $layer_name $tensor_name]
                lappend files [list $host_file $emmc_path]
            }
        }
    }

    if {($install_mode eq "full") || ($install_mode eq "assets")} {
    foreach aux_candidate [list \
        [file join $package_dir "MODEL_AUX.BIN"] \
        [file join $package_dir "AUX.BIN"] \
        [file join $package_dir "AUX" "AUX.BIN"]] {
        set aux_file [file normalize $aux_candidate]
        if {[file exists $aux_file]} {
            lappend files [list $aux_file "0:/BITNET/AUX/AUX.BIN"]
            break
        }
    }

    foreach rope_candidate [list \
        [file join $package_dir "MODEL_ROPE.BIN"] \
        [file join $package_dir "ROPE.BIN"] \
        [file join $package_dir "AUX" "ROPE.BIN"]] {
        set rope_file [file normalize $rope_candidate]
        if {[file exists $rope_file]} {
            lappend files [list $rope_file "0:/BITNET/AUX/ROPE.BIN"]
            break
        }
    }

    foreach embed_file [lsort [glob -nocomplain -type f [file join $package_dir "EMB" "E*.BIN"]]] {
        set host_file [file normalize $embed_file]
        set emmc_path [format "0:/BITNET/EMB/%s" [file tail $host_file]]
        lappend files [list $host_file $emmc_path]
    }

    foreach lm_head_file [lsort [glob -nocomplain -type f [file join $package_dir "LMH" "H*.BIN"]]] {
        set host_file [file normalize $lm_head_file]
        set emmc_path [format "0:/BITNET/LMH/%s" [file tail $host_file]]
        lappend files [list $host_file $emmc_path]
    }

    foreach tokenizer_file [lsort [glob -nocomplain -type f [file join $package_dir "TOK" "*.BIN"]]] {
        set host_file [file normalize $tokenizer_file]
        set emmc_path [format "0:/BITNET/TOK/%s" [file tail $host_file]]
        lappend files [list $host_file $emmc_path]
    }
    }

    if {$install_mode eq "lmh"} {
        foreach lm_head_file [lsort [glob -nocomplain -type f [file join $package_dir "LMH" "H*.BIN"]]] {
            set host_file [file normalize $lm_head_file]
            set emmc_path [format "0:/BITNET/LMH/%s" [file tail $host_file]]
            lappend files [list $host_file $emmc_path]
        }
    }

    return $files
}

proc write_install_control {control_file emmc_path file_size file_index file_count} {
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
        $file_index \
        $file_count]
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

set install_files [collect_bitnet_files $package_dir $tensor_names $install_mode]
set file_count [llength $install_files]
if { $file_count == 0 } {
    error "No BITNET files found under: $package_dir mode=$install_mode"
}

puts "Installing BITNET package to eMMC:"
puts "  host package: $package_dir"
puts "  mode:         $install_mode"
puts "  file count:   $file_count"

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

for {set i 0} {$i < $file_count} {incr i} {
    set item [lindex $install_files $i]
    set host_file [lindex $item 0]
    set emmc_path [lindex $item 1]
    set file_size [file size $host_file]
    set file_number [expr {$i + 1}]
    set control_file [file normalize [file join $work_dir "install_one.ctrl.bin"]]

    if {$file_size > $staging_data_bytes} {
        error [format "File is too large for JTAG staging window: %s bytes=%u limit=%u" \
            $host_file $file_size $staging_data_bytes]
    }

    puts [format "\n\[%03u/%03u\] %s -> %s (%u bytes)" \
        $file_number $file_count $host_file $emmc_path $file_size]

    write_install_control $control_file $emmc_path $file_size $file_number $file_count

    targets -set -filter {name =~ "Cortex-A53 #0"}
    rst -processor
    after 100

    clear_words $install_control_base $install_control_words
    clear_words $infer_control_base $control_words
    clear_words $emmc_control_base $control_words
    clear_words $layer_control_base $control_words
    clear_words $real_control_base $control_words
    clear_words $status_base $status_words

    puts [format "  staging data -> 0x%08X" $staging_data_base]
    dow -data $host_file $staging_data_base

    puts [format "  control -> 0x%08X" $install_control_base]
    dow -data $control_file $install_control_base

    puts "  starting installer ELF"
    dow $elf_file
    con

    wait_for_result $status_base $status_magic $poll_timeout_ms
    catch { stop }

    set magic [read_status_word $status_base 0]
    set stage [read_status_word $status_base 4]
    set result [read_status_word $status_base 5]
    set status_code [read_status_word $status_base 6]
    set bytes_written [read_status_word $status_base 7]
    set active_file [read_status_word $status_base 14]
    set active_total [read_status_word $status_base 15]

    if {$magic != $status_magic} {
        dump_status_block $status_base $status_words
        error [format "BOARD_EMMC_INSTALL_TIMEOUT: status magic is 0x%08X, expected 0x%08X" $magic $status_magic]
    }

    if {$result == 1} {
        puts [format "  BOARD_EMMC_INSTALL_FILE_PASS: file=%u/%u bytes=%u" $active_file $active_total $bytes_written]
    } elseif {$result == 0} {
        dump_status_block $status_base $status_words
        error [format "BOARD_EMMC_INSTALL_TIMEOUT: stage=%u status=0x%08X file=%u/%u bytes=%u" \
            $stage $status_code $active_file $active_total $bytes_written]
    } else {
        dump_status_block $status_base $status_words
        error [format "BOARD_EMMC_INSTALL_FAIL: stage=%u status=0x%08X file=%u/%u bytes=%u" \
            $stage $status_code $active_file $active_total $bytes_written]
    }
}

puts [format "\nBOARD_EMMC_INSTALL_PASS: files=%u package=%s" $file_count $package_dir]
