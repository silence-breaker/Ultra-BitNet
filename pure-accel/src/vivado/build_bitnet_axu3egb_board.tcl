# Native R10 TP2 AXU3EGB board build.
#
# Usage:
#   vivado -mode batch -source src/vivado/build_bitnet_axu3egb_board.tcl \
#     -tclargs project|synth|impl|xsa [project_dir]

set build_stage "project"
if {[info exists argc] && $argc >= 1} {
  set build_stage [string tolower [lindex $argv 0]]
}
if {$build_stage ni {project synth impl bitstream xsa}} {
  error "unsupported build stage '$build_stage'"
}

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".."]]
set project_dir [file join $root_dir build vivado-bitnet-r10-tp2-board]
if {[info exists argc] && $argc >= 2} {
  set project_dir [file normalize [lindex $argv 1]]
}
set project_name axu3egb_bitnet_r10_tp2
set bd_name design_1
set ps_bd_tcl [file join $script_dir board design_1_bd.tcl]
set native_rtl [file join $root_dir build rtl \
  BitNetResidentBoardAccelerator.v]

foreach required [list $ps_bd_tcl $native_rtl] {
  if {![file exists $required]} {
    error "missing required board input: $required"
  }
}

# MIG output-product generation is more reliable with a stable local TEMP and
# one worker on this Windows installation.
set vivado_temp_dir [file join $root_dir build vivado-temp-bitnet]
file mkdir $vivado_temp_dir
set ::env(TEMP) $vivado_temp_dir
set ::env(TMP) $vivado_temp_dir
set_param general.maxThreads 1
# This Windows installation inherits cluster/LSF run-manager defaults even
# though no cluster backend exists; launch_runs then leaves every synthesis in
# *.queue.rst forever.  Force the local process launcher for reproducible CLI
# builds.
set_param runs.enableClusterConf false
set_param runs.monitorLSFJobs false
set_param general.usePosixSpawnForFork false

if {[llength [get_projects -quiet]] > 0} {
  close_project
}
if {[file exists $project_dir]} {
  error "project directory already exists; choose a new output directory: $project_dir"
}
file mkdir $project_dir
create_project $project_name $project_dir -part xczu3eg-sfvc784-1-i
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

source $ps_bd_tcl
current_bd_design $bd_name
add_files -norecurse $native_rtl
set_property file_type Verilog [get_files $native_rtl]
update_compile_order -fileset sources_1

set ps [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP2 {1} \
  CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
  CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} \
  CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} \
  CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__USE__IRQ {1} \
  CONFIG.PSU__USE__IRQ0 {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {50} \
] $ps

set native [create_bd_cell -type module \
  -reference BitNetResidentBoardAccelerator bitnet_r10_tp2_0]

set clk_fast [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_fast_100]
set_property CONFIG.CLKOUT2_USED {true} $clk_fast
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {49.999500} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {99.999000} \
  CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {49.999500} \
  CONFIG.NUM_OUT_CLKS {2} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_RESET {false} \
] $clk_fast

set rst_slow [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_slow_50]
set rst_fast [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_fast_100]

set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_ctrl]
# This path only carries 32-bit single-register software accesses at 50 MHz.
# Keep SmartConnect's default low-area implementation; forcing the high-area
# engine cost about 1.6k LUT and worsened routing without improving the native
# accelerator datapath.
set_property -dict [list \
  CONFIG.NUM_SI {1} \
  CONFIG.NUM_MI {1} \
  CONFIG.NUM_CLKS {2} \
] $ctrl

set irq [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 bitnet_irq]
set_property CONFIG.NUM_PORTS {1} $irq
set zero1 [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 axis_zero1]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {0}] $zero1
set zero256 [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 axis_zero256]
set_property -dict [list CONFIG.CONST_WIDTH {256} CONFIG.CONST_VAL {0}] $zero256

set reference_clk [get_bd_pins $ps/pl_clk0]
set ps_resetn [get_bd_pins $ps/pl_resetn0]
set fast_clk [get_bd_pins $clk_fast/clk_out1]
set slow_clk [get_bd_pins $clk_fast/clk_out2]

connect_bd_net $reference_clk [get_bd_pins $clk_fast/clk_in1]
connect_bd_net $slow_clk [get_bd_pins $rst_slow/slowest_sync_clk]
connect_bd_net $ps_resetn [get_bd_pins $rst_slow/ext_reset_in]
connect_bd_net $fast_clk [get_bd_pins $rst_fast/slowest_sync_clk]
connect_bd_net $ps_resetn [get_bd_pins $rst_fast/ext_reset_in]
connect_bd_net [get_bd_pins $clk_fast/locked] [get_bd_pins $rst_slow/dcm_locked]
connect_bd_net [get_bd_pins $clk_fast/locked] [get_bd_pins $rst_fast/dcm_locked]

set slow_resetn [get_bd_pins $rst_slow/peripheral_aresetn]
set fast_resetn [get_bd_pins $rst_fast/peripheral_aresetn]
connect_bd_net $slow_clk [get_bd_pins $native/io_slowClk]
connect_bd_net $slow_resetn [get_bd_pins $native/io_slowResetn]
connect_bd_net $fast_clk [get_bd_pins $native/io_fastClk]
connect_bd_net $fast_resetn [get_bd_pins $native/io_fastResetn]

connect_bd_net $reference_clk [get_bd_pins $ctrl/aclk]
connect_bd_net $slow_clk [get_bd_pins $ctrl/aclk1]
foreach pin [list \
  zynq_ultra_ps_e_0/saxihpc0_fpd_aclk \
  zynq_ultra_ps_e_0/saxihp0_fpd_aclk \
  zynq_ultra_ps_e_0/saxihp1_fpd_aclk \
  zynq_ultra_ps_e_0/saxihp2_fpd_aclk \
  zynq_ultra_ps_e_0/saxihp3_fpd_aclk] {
  connect_bd_net $slow_clk [get_bd_pins $pin]
}
connect_bd_net $slow_resetn [get_bd_pins $ctrl/aresetn]

# The first image uses AXI-Lite PIO for HiddenResident beats.  Keep the AXIS
# pins tied low while preserving the interface for a later DMA-enabled image.
connect_bd_net [get_bd_pins $zero1/dout] [get_bd_pins $native/io_sAxisHidden_valid]
connect_bd_net [get_bd_pins $zero1/dout] [get_bd_pins $native/io_sAxisHidden_payload_last]
connect_bd_net [get_bd_pins $zero256/dout] [get_bd_pins $native/io_sAxisHidden_payload_data]

connect_bd_net [get_bd_pins $native/io_interrupt] [get_bd_pins $irq/In0]
connect_bd_net [get_bd_pins $irq/dout] [get_bd_pins $ps/pl_ps_irq0]

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_LPD] \
  [get_bd_intf_pins $ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M00_AXI] \
  [get_bd_intf_pins $native/S_AXI_CONTROL]
connect_bd_intf_net [get_bd_intf_pins $native/M_AXI_PS0] \
  [get_bd_intf_pins $ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins $native/M_AXI_PS1] \
  [get_bd_intf_pins $ps/S_AXI_HP1_FPD]
connect_bd_intf_net [get_bd_intf_pins $native/M_AXI_PS2] \
  [get_bd_intf_pins $ps/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins $native/M_AXI_PS3] \
  [get_bd_intf_pins $ps/S_AXI_HP3_FPD]
connect_bd_intf_net [get_bd_intf_pins $native/M_AXI_PL] \
  [get_bd_intf_pins $ps/S_AXI_HPC0_FPD]

assign_bd_address
set control_segments [get_bd_addr_segs -quiet -of_objects \
  [get_bd_intf_pins $native/S_AXI_CONTROL]]
puts "BITNET_CONTROL_SEGMENTS=$control_segments"
puts "BITNET_ADDRESS_SPACES=[get_bd_addr_spaces -hierarchical]"
puts "BITNET_ADDRESS_SEGMENTS=[get_bd_addr_segs -hierarchical]"
if {[llength $control_segments] != 1} {
  error "expected one native control address segment, got: $control_segments"
}
assign_bd_address -offset 0x80000000 -range 0x00010000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] \
  $control_segments -force

set bank_segments [list \
  [get_bd_addr_segs $ps/SAXIGP0/HPC0_DDR_HIGH] \
  [get_bd_addr_segs $ps/SAXIGP2/HP0_DDR_HIGH] \
  [get_bd_addr_segs $ps/SAXIGP3/HP1_DDR_HIGH] \
  [get_bd_addr_segs $ps/SAXIGP4/HP2_DDR_HIGH] \
  [get_bd_addr_segs $ps/SAXIGP5/HP3_DDR_HIGH]]
set bank_spaces [list \
  [get_bd_addr_spaces $native/M_AXI_PL] \
  [get_bd_addr_spaces $native/M_AXI_PS0] \
  [get_bd_addr_spaces $native/M_AXI_PS1] \
  [get_bd_addr_spaces $native/M_AXI_PS2] \
  [get_bd_addr_spaces $native/M_AXI_PS3]]
for {set bank 0} {$bank < 5} {incr bank} {
  # All five masters see the complete 2 GiB PS high-DDR aperture.  Banks 1--4
  # already emit 0x8_0000_0000 through 0x8_3000_0000.  The board boundary
  # translates bank 0 local addresses to 0x8_5000_0000, above the 0x8_4000_0000
  # source-staging region, so boot/runtime memory in low DDR is untouched.
  assign_bd_address -offset 0x800000000 -range 0x80000000 \
    -target_address_space [lindex $bank_spaces $bank] \
    [lindex $bank_segments $bank] -force
}

validate_bd_design
save_bd_design

set bd_file [get_files */${bd_name}.bd]
generate_target all $bd_file
set wrapper_candidates [make_wrapper -files $bd_file -top]
if {[llength $wrapper_candidates] == 0} {
  set wrapper_candidates [glob -nocomplain [file join $project_dir \
    "${project_name}.gen" sources_1 bd $bd_name hdl "${bd_name}_wrapper.v"]]
}
if {[llength $wrapper_candidates] != 1} {
  error "cannot find generated BD wrapper: $wrapper_candidates"
}
add_files -norecurse [lindex $wrapper_candidates 0]
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$build_stage in {synth impl bitstream xsa}} {
  launch_runs synth_1 -jobs 2
  wait_on_run synth_1
  if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "synth_1 did not complete"
  }
  open_run synth_1
  report_utilization -hierarchical \
    -file [file join $project_dir bitnet_board_utilization_synth.rpt]
}

if {$build_stage in {impl bitstream xsa}} {
  set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
  launch_runs impl_1 -to_step write_bitstream -jobs 1
  wait_on_run impl_1
  if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "impl_1 did not complete"
  }
  open_run impl_1
  report_utilization -hierarchical \
    -file [file join $project_dir bitnet_board_utilization_routed.rpt]
  report_timing_summary -delay_type min_max -report_unconstrained -max_paths 100 \
    -file [file join $project_dir bitnet_board_timing_routed.rpt]
  report_route_status \
    -file [file join $project_dir bitnet_board_route_status.rpt]
  report_drc -file [file join $project_dir bitnet_board_drc_routed.rpt]
  set run_bit [file join $project_dir "${project_name}.runs" impl_1 \
    "${bd_name}_wrapper.bit"]
  if {![file exists $run_bit]} {
    error "implementation completed without bitstream: $run_bit"
  }
  file copy -force $run_bit [file join $project_dir bitnet_r10_tp2.bit]
}

if {$build_stage eq "xsa"} {
  write_hw_platform -fixed -include_bit -force \
    -file [file join $project_dir axu3egb_bitnet_r10_tp2.xsa]
}

puts "BITNET_BOARD_STAGE=$build_stage"
puts "BITNET_BOARD_PROJECT=$project_dir"
exit 0
