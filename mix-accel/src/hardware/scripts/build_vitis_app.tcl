set build_stage "build"
if { $argc >= 1 } {
    set build_stage [lindex $argv 0]
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".." ".."]]
set workspace_dir [file normalize [file join $root_dir "build" "vitis_ws"]]
set xsa_file [file normalize [file join $root_dir "build" "axu3egb_bitnet_accel.xsa"]]
set src_dir [file normalize [file join $root_dir "hardware" "baremetal"]]
set app_name "bitnet_accel_smoke"
set proc_name "psu_cortexa53_0"

if { $build_stage eq "clean" } {
    file delete -force $workspace_dir
    puts "Cleaned Vitis workspace: $workspace_dir"
    return
}

if { ![file exists $xsa_file] } {
    error "Missing XSA file: $xsa_file. Run src/hardware/scripts/build_vivado_project.bat xsa first."
}
if { ![file isdirectory $src_dir] } {
    error "Missing bare-metal source directory: $src_dir"
}

file delete -force $workspace_dir
file mkdir $workspace_dir
setws $workspace_dir

puts "Creating Vitis app '$app_name' from $xsa_file"
app create -name $app_name -hw $xsa_file -proc $proc_name -os standalone -lang C -template {Empty Application}

puts "Importing bare-metal sources from $src_dir"
importsources -name $app_name -path $src_dir

puts "Enabling xilffs for eMMC FATFS access"
bsp setlib -name xilffs -ver 4.3
bsp regenerate

puts "Linking the math library for scale-aware inference"
app config -name $app_name -add libraries m

puts "Enabling -O3 for Cortex-A53 inference performance"
app config -name $app_name -set compiler-optimization {Optimize most (-O3)}

puts "Building Vitis app '$app_name'"
app build -name $app_name

set elf_file [file join $workspace_dir $app_name "Debug" "${app_name}.elf"]
if { ![file exists $elf_file] } {
    error "Vitis build completed without expected ELF: $elf_file"
}

puts "Built ELF: $elf_file"
