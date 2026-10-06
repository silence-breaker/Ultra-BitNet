set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".." ".."]]
set project_dir [file normalize [file join $root_dir "build" "vivado-plddr-probe"]]
set ps_bd_tcl [file join $root_dir "hardware" "board" "axu3egb" "design_1_bd.tcl"]
set pl_xdc [file join $root_dir "hardware" "board" "axu3egb" "pl_ddr4.xdc"]

file delete -force $project_dir
create_project axu3eg_plddr_probe $project_dir -part xazu3eg-sfvc784-1-i
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
source $ps_bd_tcl
current_bd_design design_1
add_files -fileset constrs_1 -norecurse $pl_xdc

set ps [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {125} \
] $ps

set ddr [create_bd_cell -type ip -vlnv xilinx.com:ip:ddr4:2.2 pl_ddr4_0]
set_property -dict [list \
    CONFIG.C0.DDR4_MemoryPart {MT40A512M16LY-075} \
    CONFIG.C0.DDR4_DataWidth {16} \
    CONFIG.C0.DDR4_Capacity {1024} \
    CONFIG.C0.BANK_GROUP_WIDTH {1} \
    CONFIG.C0.DDR4_InputClockPeriod {5003} \
    CONFIG.C0.DDR4_TimePeriod {938} \
    CONFIG.C0.DDR4_PhyClockRatio {4:1} \
    CONFIG.C0.DDR4_AxiSelection {true} \
    CONFIG.C0.DDR4_AxiDataWidth {128} \
    CONFIG.C0.DDR4_AxiIDWidth {4} \
    CONFIG.C0.DDR4_AxiNarrowBurst {true} \
    CONFIG.C0_CLOCK_BOARD_INTERFACE {Custom} \
    CONFIG.C0_DDR4_BOARD_INTERFACE {Custom} \
    CONFIG.System_Clock {Differential} \
    CONFIG.Reference_Clock {Differential} \
] $ddr
puts "PLDDR_SYSCLK_INTF=[get_bd_intf_pins $ddr/C0_SYS_CLK]"
make_bd_intf_pins_external -name c0_sys_clk [get_bd_intf_pins $ddr/C0_SYS_CLK]
set_property CONFIG.FREQ_HZ {200000000} [get_bd_intf_ports c0_sys_clk]
make_bd_intf_pins_external -name c0_ddr4 [get_bd_intf_pins $ddr/C0_DDR4]

set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_plddr]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc
set rst_sys [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_plddr_sys]
set_property CONFIG.C_EXT_RESET_HIGH {0} $rst_sys
set rst_ui [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_plddr_ui]
set_property CONFIG.C_EXT_RESET_HIGH {1} $rst_ui

set pl_clk [get_bd_pins $ps/pl_clk0]
set pl_resetn [get_bd_pins $ps/pl_resetn0]
connect_bd_net $pl_clk [get_bd_pins $rst_sys/slowest_sync_clk]
connect_bd_net $pl_resetn [get_bd_pins $rst_sys/ext_reset_in]
connect_bd_net [get_bd_pins $ddr/c0_ddr4_ui_clk] [get_bd_pins $rst_ui/slowest_sync_clk]
connect_bd_net [get_bd_pins $ddr/c0_ddr4_ui_clk_sync_rst] [get_bd_pins $rst_ui/ext_reset_in]
connect_bd_net [get_bd_pins $ddr/c0_ddr4_ui_clk] [get_bd_pins $smc/aclk]
connect_bd_net [get_bd_pins $ddr/c0_ddr4_ui_clk] [get_bd_pins $ps/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins $rst_ui/peripheral_aresetn] [get_bd_pins $smc/aresetn]

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $smc/M00_AXI] [get_bd_intf_pins $ddr/C0_DDR4_S_AXI]

connect_bd_net [get_bd_pins $rst_sys/peripheral_reset] [get_bd_pins $ddr/sys_rst]
connect_bd_net [get_bd_pins $rst_ui/peripheral_aresetn] [get_bd_pins $ddr/c0_ddr4_aresetn]

assign_bd_address
puts "PLDDR_ADDR_SEGS=[get_bd_addr_segs -of_objects [get_bd_intf_pins $ddr/C0_DDR4_S_AXI]]"
puts "PLDDR_SMC_ADDR_SEGS=[get_bd_addr_segs -of_objects [get_bd_intf_pins $smc/M00_AXI]]"
validate_bd_design
save_bd_design

set bd_file [get_files */design_1.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrappers [glob -nocomplain [file join $project_dir "axu3eg_plddr_probe.srcs" "sources_1" "bd" "design_1" "hdl" "design_1_wrapper.v"]]
if {[llength $wrappers] > 0} {
    add_files -norecurse [lindex $wrappers 0]
    set_property top design_1_wrapper [current_fileset]
}
update_compile_order -fileset sources_1
puts "PLDDR_PROBE_PROJECT=$project_dir"
