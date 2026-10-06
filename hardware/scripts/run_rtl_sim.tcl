set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".."]]
set sim_dir [file normalize [file join $root_dir "build" "xsim" "direct"]]

file mkdir $sim_dir
cd $sim_dir

foreach artifact {xsim.dir webtalk.jou webtalk.log xvlog.log xelab.log xsim.log} {
    file delete -force [file join $sim_dir $artifact]
}

proc find_vivado_tool {tool_name} {
    set exe_dir [file dirname [info nameofexecutable]]
    set candidates [list \
        [file join $exe_dir "${tool_name}.bat"] \
        [file join $exe_dir $tool_name] \
        [file normalize [file join $exe_dir ".." ".." ".." "${tool_name}.bat"]] \
        [file normalize [file join $exe_dir ".." ".." ".." $tool_name]] \
    ]

    foreach candidate $candidates {
        if {[file exists $candidate]} {
            return $candidate
        }
    }

    set from_path [auto_execok $tool_name]
    if {$from_path ne ""} {
        return $from_path
    }

    error "Unable to locate Vivado tool: $tool_name"
}

set xvlog [find_vivado_tool "xvlog"]
set xelab [find_vivado_tool "xelab"]
set xsim  [find_vivado_tool "xsim"]

set rtl [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_axis.sv"]
set bd_wrapper [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_axis_bd.v"]
set tb  [file join $root_dir "hardware" "accelerator" "sim" "tb_bitnet_gemv_axis.sv"]
set vivado_root [expr {[info exists ::env(XILINX_VIVADO)] ? $::env(XILINX_VIVADO) : [file normalize [file join [file dirname [info nameofexecutable]] ".." ".." "data"]]}]
set xpm_memory [file normalize [file join $vivado_root "data" "ip" "xpm" "xpm_memory" "hdl" "xpm_memory.sv"]]
if { ![file exists $xpm_memory] } {
    error "Missing Vivado XPM memory simulation source: $xpm_memory"
}

exec $xvlog -sv $xpm_memory $rtl $bd_wrapper $tb >@ stdout 2>@ stderr
exec $xelab tb_bitnet_gemv_axis -L xpm -s tb_bitnet_gemv_axis_sim >@ stdout 2>@ stderr
exec $xsim tb_bitnet_gemv_axis_sim -runall >@ stdout 2>@ stderr
puts "RTL simulation completed."
