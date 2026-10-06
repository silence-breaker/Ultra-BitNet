set project_root [file normalize [file join [file dirname [info script]] ..]]
set xsa [file join $project_root build axu3egb_bitnet_accel.xsa]
set repo [file join $project_root build device-tree-xlnx-v2020.1]
set output_dir [file join $project_root build linux-device-tree]

foreach required_path [list $xsa $repo] {
    if {![file exists $required_path]} {
        error "Required path does not exist: $required_path"
    }
}

file delete -force $output_dir
file mkdir $output_dir

hsi::open_hw_design $xsa
hsi::set_repo_path $repo

set processor ""
foreach cell [hsi::get_cells -hier -filter {IP_TYPE==PROCESSOR}] {
    if {[string match *cortexa53* $cell]} {
        set processor $cell
        break
    }
}

if {$processor eq ""} {
    hsi::close_hw_design [hsi::current_hw_design]
    error "No Cortex-A53 processor was found in $xsa"
}

puts "Generating Linux device tree for $processor"
hsi::create_sw_design device-tree -os device_tree -proc $processor
hsi::generate_target -dir $output_dir
hsi::close_hw_design [hsi::current_hw_design]

puts "DEVICE_TREE_READY=$output_dir"
exit
