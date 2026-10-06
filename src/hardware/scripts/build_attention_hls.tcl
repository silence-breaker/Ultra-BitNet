set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".." ".."]]
set project_dir [file normalize [file join $root_dir "build" "hls_attention"]]
set source_file [file normalize [file join $root_dir "hardware" "accelerator" "hls" "bitnet_attention_axis.cpp"]]
set test_file [file normalize [file join $root_dir "hardware" "accelerator" "hls" "tb_bitnet_attention_axis.cpp"]]
set export_dir [file normalize [file join $root_dir "build" "hls_rtl" "bitnet_attention_axis"]]

file mkdir [file dirname $project_dir]
file delete -force $export_dir
cd [file dirname $project_dir]
open_project -reset [file tail $project_dir]
set_top bitnet_attention_axis
add_files $source_file
add_files -tb $test_file
open_solution -reset solution1
set_part {xazu3eg-sfvc784-1-i}
create_clock -period 6.4 -name default
config_schedule -enable_dsp_full_reg

csim_design
csynth_design
set generated_rtl_dir [file join $project_dir "solution1" "syn" "verilog"]
if { ![file exists [file join $generated_rtl_dir "bitnet_attention_axis.v"]] } {
    error "HLS synthesis completed without bitnet_attention_axis.v"
}
file mkdir [file dirname $export_dir]
file copy -force $generated_rtl_dir $export_dir
exit
