set prompt_text ""
set max_new_tokens 1
set layer_count 30
set first_layer 0
set repeat_token_count 0
set prefill_batch_rows 16
if { $argc >= 1 } {
    set prompt_text [lindex $argv 0]
}
if { $argc >= 2 } {
    set max_new_tokens [lindex $argv 1]
}
if { $argc >= 3 } {
    set layer_count [lindex $argv 2]
}
if { $argc >= 4 } {
    set first_layer [lindex $argv 3]
}
if { $argc >= 7 } {
    set repeat_token_count [lindex $argv 6]
}
if { $argc >= 8 } {
    set prefill_batch_rows [lindex $argv 7]
}
if { ($prefill_batch_rows < 1) || ($prefill_batch_rows > 16) } {
    error [format "Prefill batch rows must be in 1..16: %u" $prefill_batch_rows]
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set bit_file [file normalize [file join $root_dir "build" "vivado" "axu3egb_bitnet_accel.runs" "impl_1" "design_1_wrapper.bit"]]
if { $argc >= 9 } {
    set bit_file [file normalize [lindex $argv 8]]
}
set elf_file [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "Debug" "bitnet_accel_smoke.elf"]]
set payload_file [file normalize [file join $root_dir "build" "baremetal-payload" "payload.bin"]]
set psu_init_tcl [file normalize [file join $root_dir "build" "vitis_ws" "bitnet_accel_smoke" "_ide" "psinit" "psu_init.tcl"]]
set work_dir [file normalize [file join $root_dir "build" "infer"]]

set infer_control_base 0x4B000000
set infer_prompt_base 0x4B010000
set install_control_base 0x4C000000
set emmc_control_base 0x4D000000
set layer_control_base 0x4E000000
set real_control_base 0x4F000000
set infer_output_text_base 0x64630000
set infer_output_text_magic 0x314F4942
set infer_output_text_max_bytes 8192
set infer_hidden_base 0x60000000
set infer_final_norm_base 0x600A0000
set infer_vector_words 640
set infer_q_out_base 0x60030000
set infer_down_out_base 0x60090000
set infer_output_words 2560
set control_words 64
set infer_control_words 16
set status_base 0x70000000
set status_words 48
set status_magic 0x42545354
set infer_magic 0x42494631
set infer_version 1
set payload_base 0x00400000
set baseline_main_addr 0x0001C050
set payload_branch_instruction 0x140F8FEC
set infer_mode_uart_text 1
set infer_mode_ddr_text 2
set infer_mode_ddr_token_ids 3
set activation_seed 23
set flags 1
set temperature_milli 600
set top_k 40
set poll_timeout_ms 900000
if { $argc >= 5 } {
    set temperature_milli [lindex $argv 4]
}
if { $argc >= 6 } {
    set top_k [lindex $argv 5]
}
if { $repeat_token_count > 0 } {
    set poll_timeout_ms 3600000
}

foreach required_file [list $bit_file $elf_file $payload_file $psu_init_tcl] {
    if { ![file exists $required_file] } {
        error "Missing required inference run file: $required_file"
    }
}
file mkdir $work_dir

proc write_infer_control {control_file mode prompt_addr prompt_len first_layer layer_count seed flags max_new temp top_k prefill_batch_rows} {
    global infer_magic infer_version
    set fh [open $control_file "wb"]
    fconfigure $fh -translation binary -encoding binary
    puts -nonewline $fh [binary format "iiiiiiiiiiiiiiii" \
        $infer_magic \
        $infer_version \
        $mode \
        $prompt_addr \
        $prompt_len \
        $first_layer \
        $layer_count \
        $seed \
        $flags \
        $max_new \
        $temp \
        $top_k \
        $prefill_batch_rows 0 0 0]
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

proc read_output_text {base expected_magic max_bytes} {
    set magic [read_status_word $base 0]
    set version [read_status_word $base 1]
    set byte_count [read_status_word $base 2]
    if {$magic != $expected_magic || $version != 1 || $byte_count == 0} {
        return ""
    }
    if {$byte_count > $max_bytes} {
        set byte_count $max_bytes
    }

    set raw ""
    for {set offset 0} {$offset < $byte_count} {incr offset 4} {
        set word [mrd -value [expr {$base + 16 + $offset}]]
        if {[llength $word] > 1} {
            set word [lindex $word 0]
        }
        for {set b 0} {$b < 4 && ($offset + $b) < $byte_count} {incr b} {
            set byte [expr {($word >> (8 * $b)) & 0xFF}]
            if {$byte >= 128} {
                set byte [expr {$byte - 256}]
            }
            append raw [binary format c $byte]
        }
    }
    return [encoding convertfrom utf-8 $raw]
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

set control_file [file normalize [file join $work_dir "infer.ctrl.bin"]]
set prompt_file [file normalize [file join $work_dir "prompt.bin"]]
set mode $infer_mode_uart_text
set prompt_len 0

if { $repeat_token_count > 0 } {
    if { $repeat_token_count > 4096 } {
        error "Token-ID prompt exceeds the 4096-token context limit: $repeat_token_count"
    }
    set mode $infer_mode_ddr_token_ids
    set token_ids {}
    for {set index 0} {$index < $repeat_token_count} {incr index} {
        if {$index == 0} {
            lappend token_ids 128000
        } elseif {$index == 1} {
            lappend token_ids 9906
        } else {
            lappend token_ids 1917
        }
    }
    set prompt_bytes [binary format "i*" $token_ids]
    set prompt_len [string length $prompt_bytes]
    set fh [open $prompt_file "wb"]
    fconfigure $fh -translation binary -encoding binary
    puts -nonewline $fh $prompt_bytes
    close $fh
} elseif { $prompt_text ne "" } {
    set mode $infer_mode_ddr_text
    set prompt_bytes [encoding convertto utf-8 $prompt_text]
    set prompt_len [string length $prompt_bytes]
    set fh [open $prompt_file "wb"]
    fconfigure $fh -translation binary -encoding binary
    puts -nonewline $fh $prompt_bytes
    close $fh
}

write_infer_control $control_file $mode $infer_prompt_base $prompt_len \
    $first_layer $layer_count $activation_seed $flags $max_new_tokens \
    $temperature_milli $top_k $prefill_batch_rows

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

puts "Clearing control/status blocks"
clear_words $infer_control_base $infer_control_words
clear_words $install_control_base $control_words
clear_words $emmc_control_base $control_words
clear_words $layer_control_base $control_words
clear_words $real_control_base $control_words
clear_words $status_base $status_words

if { $mode != $infer_mode_uart_text } {
    puts [format "Downloading prompt payload (%u bytes, mode=%u) -> 0x%08X" \
        $prompt_len $mode $infer_prompt_base]
    dow -data $prompt_file $infer_prompt_base
} else {
    puts "UART prompt mode selected. Open the PL/PS UART terminal and type the prompt after the app prints prompt>."
}

puts [format "Downloading inference control -> 0x%08X" $infer_control_base]
dow -data $control_file $infer_control_base

puts "Downloading ELF to Cortex-A53 #0: $elf_file"
dow $elf_file
puts [format "Downloading optimized inference payload -> 0x%08X" $payload_base]
dow -data $payload_file $payload_base
puts [format "Redirecting baseline main at 0x%08X to payload entry" $baseline_main_addr]
mwr $baseline_main_addr $payload_branch_instruction
set patched_instruction [mrd -value $baseline_main_addr]
if {[llength $patched_instruction] > 1} {
    set patched_instruction [lindex $patched_instruction 0]
}
if {$patched_instruction != $payload_branch_instruction} {
    error [format "Payload entry patch verification failed: got 0x%08X" $patched_instruction]
}
con

if { $mode == $infer_mode_uart_text } {
    puts "BOARD_INFER_UART_STARTED"
    puts "Use the UART terminal for prompt input. XSCT will not poll because the app is waiting for serial input."
    return
}

puts "DDR prompt inference started."
wait_for_result $status_base $status_magic $poll_timeout_ms
catch { stop }

set magic [read_status_word $status_base 0]
set stage [read_status_word $status_base 4]
set result [read_status_word $status_base 5]
set status_code [read_status_word $status_base 6]
set token_count [read_status_word $status_base 16]
set generated_count [read_status_word $status_base 17]
set cache_hits [read_status_word $status_base 18]
set cache_misses [read_status_word $status_base 19]
set last_token [read_status_word $status_base 20]
set pseudo_score [read_status_word $status_base 21]
set emmc_us [read_status_word $status_base 32]
set accel_us [read_status_word $status_base 33]
set total_us [read_status_word $status_base 34]
set prefill_us [read_status_word $status_base 35]
set decode_us [read_status_word $status_base 36]
set lm_head_us [read_status_word $status_base 37]
set context_limit [read_status_word $status_base 38]
set kv_cache_mib [read_status_word $status_base 39]
set attention_block_us [read_status_word $status_base 40]
set mlp_block_us [read_status_word $status_base 41]
set attention_projection_us [read_status_word $status_base 42]
set attention_core_us [read_status_word $status_base 43]
set attention_nonlinear_us [read_status_word $status_base 44]
set mlp_projection_us [read_status_word $status_base 45]
set mlp_nonlinear_us [read_status_word $status_base 46]
set layer_forward_us [read_status_word $status_base 47]
dump_status_block $status_base $status_words

if {$magic != $status_magic} {
    error [format "BOARD_INFER_TIMEOUT: status magic is 0x%08X, expected 0x%08X" $magic $status_magic]
}

if {$result == 1} {
    set hidden_dump [file normalize [file join $work_dir "board_hidden.i8.bin"]]
    set final_norm_dump [file normalize [file join $work_dir "board_final_norm.i8.bin"]]
    set q_out_dump [file normalize [file join $work_dir "board_q_out.i32.bin"]]
    set down_out_dump [file normalize [file join $work_dir "board_down_out.i32.bin"]]
    mrd -bin -file $hidden_dump $infer_hidden_base $infer_vector_words
    mrd -bin -file $final_norm_dump $infer_final_norm_base $infer_vector_words
    mrd -bin -file $q_out_dump $infer_q_out_base $infer_output_words
    mrd -bin -file $down_out_dump $infer_down_out_base $infer_output_words
    puts "BOARD_INFER_DUMPS: hidden=$hidden_dump final_norm=$final_norm_dump"
    set output_text [read_output_text $infer_output_text_base $infer_output_text_magic $infer_output_text_max_bytes]
    puts [format "BOARD_INFER_PASS: tokens=%u generated=%u last_token=%u pseudo_score=0x%08X cache_hits=%u cache_misses=%u emmc_us=%u accel_us=%u prefill_us=%u decode_us=%u lm_head_us=%u total_us=%u context_limit=%u kv_cache_mib=%u" \
        $token_count $generated_count $last_token $pseudo_score $cache_hits $cache_misses $emmc_us $accel_us $prefill_us $decode_us $lm_head_us $total_us $context_limit $kv_cache_mib]
    set nonlinear_us [expr {$attention_nonlinear_us + $mlp_nonlinear_us}]
    set accounted_us [expr {$attention_block_us + $mlp_block_us}]
    set overhead_us [expr {$layer_forward_us - $accounted_us}]
    puts [format "BLOCK_TIMING_US: attention=%u mlp=%u attn_proj=%u attn_core=%u attn_nonlin=%u mlp_proj=%u mlp_nonlin=%u nonlinear_total=%u layer=%u overhead=%d" \
        $attention_block_us $mlp_block_us $attention_projection_us $attention_core_us \
        $attention_nonlinear_us $mlp_projection_us $mlp_nonlinear_us $nonlinear_us \
        $layer_forward_us $overhead_us]
    if {$layer_forward_us > 0} {
        puts [format "BLOCK_TIMING_PCT: attention=%.3f mlp=%.3f nonlinear_total=%.3f attn_proj=%.3f attn_core=%.3f attn_nonlin=%.3f mlp_proj=%.3f mlp_nonlin=%.3f overhead=%.3f" \
            [expr {100.0 * $attention_block_us / $layer_forward_us}] \
            [expr {100.0 * $mlp_block_us / $layer_forward_us}] \
            [expr {100.0 * $nonlinear_us / $layer_forward_us}] \
            [expr {100.0 * $attention_projection_us / $layer_forward_us}] \
            [expr {100.0 * $attention_core_us / $layer_forward_us}] \
            [expr {100.0 * $attention_nonlinear_us / $layer_forward_us}] \
            [expr {100.0 * $mlp_projection_us / $layer_forward_us}] \
            [expr {100.0 * $mlp_nonlinear_us / $layer_forward_us}] \
            [expr {100.0 * $overhead_us / $layer_forward_us}]]
    }
    puts "BOARD_INFER_TEXT: $output_text"
} elseif {$result == 0} {
    error [format "BOARD_INFER_TIMEOUT: stage=%u status=0x%08X" $stage $status_code]
} else {
    error [format "BOARD_INFER_FAIL: stage=%u status=0x%08X tokens=%u generated=%u" \
        $stage $status_code $token_count $generated_count]
}
