set build_stage "project"
if { $argc >= 1 } {
    set build_stage [lindex $argv 0]
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".." ".."]]
set project_dir [file normalize [file join $root_dir "build" "vivado"]]
if { $argc >= 2 } {
    set project_dir [file normalize [lindex $argv 1]]
}
set project_name "axu3egb_bitnet_accel"
set bd_name "design_1"
set contest_dir [file dirname $root_dir]
set local_ps_bd [file join $root_dir "hardware" "board" "axu3egb" "design_1_bd.tcl"]
set ps_bd_candidates [list]

if { [file exists $local_ps_bd] } {
    lappend ps_bd_candidates $local_ps_bd
}

set ps_bd_candidates [concat $ps_bd_candidates [glob -nocomplain -types f \
    [file join $contest_dir "AXU3EGB*" "course_s2_addition" "ps_emmc" "vivado" "ps_emmc.srcs" "sources_1" "bd" "design_1" "hw_handoff" "design_1_bd.tcl"]]]

if { [llength $ps_bd_candidates] == 0 } {
    error "Cannot find AXU3EGB PS reference BD Tcl. Expected local copy: $local_ps_bd"
}
set ps_bd_tcl [file normalize [lindex $ps_bd_candidates 0]]

file delete -force $project_dir
file mkdir $project_dir

create_project $project_name $project_dir -part xazu3eg-sfvc784-1-i
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# Restore the PS block design before generating the HLS floating-point IPs.
# Vivado 2020.1 on Windows can fail to reload its BD Tcl framework after the
# generated IP scripts have opened all of their synthesis/simulation targets.
source $ps_bd_tcl
current_bd_design $bd_name

set rtl_file [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_axis.sv"]
set bd_wrapper_file [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_axis_bd.v"]
set router_file [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_axis_router.sv"]
set attention_wrapper_file [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_attention_axis_bd.v"]
set attention_rtl_dir [file join $root_dir "build" "hls_rtl" "bitnet_attention_axis"]
set attention_top [file join $attention_rtl_dir "bitnet_attention_axis.v"]
if { ![file exists $attention_top] } {
    error "Missing HLS attention RTL. Run src/hardware/scripts/build_attention_hls.bat first."
}
set attention_rtl_files [glob -nocomplain -types f [file join $attention_rtl_dir "*.v"]]
add_files -norecurse [concat [list $rtl_file $bd_wrapper_file $router_file $attention_wrapper_file] $attention_rtl_files]
set_property file_type SystemVerilog [get_files $rtl_file]
set_property file_type Verilog [get_files $router_file]

foreach ip_script [glob -nocomplain -types f [file join $attention_rtl_dir "*_ip.tcl"]] {
    source $ip_script
}
update_compile_order -fileset sources_1

set ps [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
    CONFIG.PSU__USE__IRQ {1} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {125} \
] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_pl_0]
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] $rst

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_include_mm2s_dre {0} \
    CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_addr_width {32} \
] $dma

set irq_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 dma_irq_concat]
set_property -dict [list CONFIG.NUM_PORTS {2}] $irq_concat

set kern [create_bd_cell -type module -reference bitnet_gemv_axis_bd bitnet_gemv_axis_0]
set_property -dict [list CONFIG.S_AXIS_DATA_WIDTH {512} CONFIG.MAX_K {8192}] $kern
set router [create_bd_cell -type module -reference bitnet_axis_router bitnet_axis_router_0]
set_property -dict [list CONFIG.S_AXIS_DATA_WIDTH {128}] $router
set attention [create_bd_cell -type module -reference bitnet_attention_axis_bd bitnet_attention_axis_0]

set gemv_width [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 axis_gemv_128_to_512]
set_property -dict [list \
    CONFIG.S_TDATA_NUM_BYTES {16} \
    CONFIG.M_TDATA_NUM_BYTES {64} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.HAS_TSTRB {1} \
] $gemv_width

set attn_width [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 axis_attn_128_to_64]
set_property -dict [list \
    CONFIG.S_TDATA_NUM_BYTES {16} \
    CONFIG.M_TDATA_NUM_BYTES {8} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
    CONFIG.HAS_TSTRB {1} \
] $attn_width

set smc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc_ctrl

set smc_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_mem]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $smc_mem

set pl_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
set pl_resetn [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0]

connect_bd_net $pl_clk [get_bd_pins rst_pl_0/slowest_sync_clk]
connect_bd_net $pl_resetn [get_bd_pins rst_pl_0/ext_reset_in]

connect_bd_net $pl_clk [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net $pl_clk [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net $pl_clk [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
connect_bd_net $pl_clk [get_bd_pins axi_smc_ctrl/aclk]
connect_bd_net $pl_clk [get_bd_pins axi_smc_mem/aclk]
connect_bd_net $pl_clk [get_bd_pins bitnet_gemv_axis_0/ap_clk]
connect_bd_net $pl_clk [get_bd_pins bitnet_axis_router_0/ap_clk]
connect_bd_net $pl_clk [get_bd_pins bitnet_attention_axis_0/ap_clk]
connect_bd_net $pl_clk [get_bd_pins axis_gemv_128_to_512/aclk]
connect_bd_net $pl_clk [get_bd_pins axis_attn_128_to_64/aclk]
connect_bd_net $pl_clk [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]

connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins axi_dma_0/axi_resetn]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins axi_smc_ctrl/aresetn]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins axi_smc_mem/aresetn]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins bitnet_gemv_axis_0/ap_rst_n]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins bitnet_axis_router_0/ap_rst_n]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins bitnet_attention_axis_0/ap_rst_n]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins axis_gemv_128_to_512/aresetn]
connect_bd_net [get_bd_pins rst_pl_0/peripheral_aresetn] [get_bd_pins axis_attn_128_to_64/aresetn]

connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins dma_irq_concat/In0]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins dma_irq_concat/In1]
connect_bd_net [get_bd_pins dma_irq_concat/dout] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins axi_smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_ctrl/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins bitnet_axis_router_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins bitnet_axis_router_0/M_GEMV] [get_bd_intf_pins axis_gemv_128_to_512/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_gemv_128_to_512/M_AXIS] [get_bd_intf_pins bitnet_gemv_axis_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins bitnet_axis_router_0/M_ATTN] [get_bd_intf_pins axis_attn_128_to_64/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_attn_128_to_64/M_AXIS] [get_bd_intf_pins bitnet_attention_axis_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins bitnet_gemv_axis_0/M_AXIS] [get_bd_intf_pins bitnet_axis_router_0/S_GEMV]
connect_bd_intf_net [get_bd_intf_pins bitnet_attention_axis_0/M_AXIS] [get_bd_intf_pins bitnet_axis_router_0/S_ATTN]
connect_bd_intf_net [get_bd_intf_pins bitnet_axis_router_0/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_smc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_smc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_mem/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

assign_bd_address
validate_bd_design
save_bd_design

set bd_file [get_files */${bd_name}.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrapper_candidates [concat \
    [glob -nocomplain [file join $project_dir "${project_name}.srcs" "sources_1" "bd" $bd_name "hdl" "${bd_name}_wrapper.v"]] \
    [glob -nocomplain [file join $project_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hdl" "${bd_name}_wrapper.v"]] \
]
if { [llength $wrapper_candidates] == 0 } {
    error "Cannot find generated BD wrapper for $bd_name"
}
add_files -norecurse [lindex $wrapper_candidates 0]
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

if { $build_stage eq "synth" || $build_stage eq "impl" || $build_stage eq "bitstream" || $build_stage eq "xsa" } {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        error "synth_1 did not complete"
    }
}

if { $build_stage eq "impl" || $build_stage eq "bitstream" || $build_stage eq "xsa" } {
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
        error "impl_1 did not complete"
    }
}

if { $build_stage eq "xsa" } {
    set xsa_file [file join $root_dir "build" "axu3egb_bitnet_accel.xsa"]
    write_hw_platform -fixed -include_bit -force -file $xsa_file
    puts "Exported hardware platform: $xsa_file"
}

puts "Vivado build stage '$build_stage' completed."
