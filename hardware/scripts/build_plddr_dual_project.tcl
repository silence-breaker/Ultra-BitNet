# Candidate PL-DDR + dual-lane GEMV project.
#
# This project is intentionally separate from hardware/scripts/build_vivado_project.tcl:
# the existing full LLM bitstream remains the known-good fallback while this
# candidate validates the board DDR4 pinout, 64-bit PL-DDR address map and the
# two-lane AXI data path.
#
# Data path (default "plddr" topology):
#   DMA0 MM2S -> AXI clock converter -> PL-DDR SmartConnect -> DDR4
#   DMA1 MM2S -> AXI clock converter -> PL-DDR SmartConnect -> DDR4
#
# Data path (optional "split" topology):
#   DMA0 MM2S -> PS-DDR SmartConnect -> PS DDR via HP1
#   DMA1 MM2S -> AXI clock converter -> PL-DDR SmartConnect -> DDR4
#
# Both topologies keep HPM0_FPD connected to PL DDR for initial packet-cache
# population and DMA0 S2MM connected to PS DDR via HP0 for result writes.
#
# DMA0 and DMA1 are controlled through HPM0_LPD.  The dual GEMV emits lane0
# followed by lane1 and only asserts TLAST on the final lane.

set build_stage "project"
if { [info exists ::PLDDR_DUAL_BUILD_STAGE] } {
    set build_stage $::PLDDR_DUAL_BUILD_STAGE
} elseif { [info exists argc] && $argc >= 1 } {
    set build_stage [lindex $argv 0]
}

set memory_topology "plddr"
if { [info exists ::PLDDR_DUAL_MEMORY_TOPOLOGY] } {
    set memory_topology $::PLDDR_DUAL_MEMORY_TOPOLOGY
} elseif { [info exists argc] && $argc >= 3 } {
    set memory_topology [string tolower [lindex $argv 2]]
}
if { $memory_topology ni {plddr split} } {
    error "Unsupported memory topology '$memory_topology'; expected plddr or split"
}
set split_memory [expr {$memory_topology eq "split"}]

set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ".." ".."]]
set project_dir [file normalize [file join $root_dir "build" "vivado-plddr-dual"]]
if { ![info exists ::PLDDR_DUAL_BUILD_STAGE] &&
     [info exists argc] && $argc >= 2 } {
    set project_dir [file normalize [lindex $argv 1]]
}
set project_name "axu3egb_plddr_dual"
set bd_name "design_1"
set artifact_stem [expr {$split_memory ? "pspl_dual" : "plddr_dual"}]
set ps_bd_tcl [file join $root_dir "hardware" "board" "axu3egb" "design_1_bd.tcl"]
set pl_xdc [file join $root_dir "hardware" "board" "axu3egb" "pl_ddr4.xdc"]

# MIG creates a generated PHY XDC and immediately starts a child synthesis
# process that consumes it.  Vivado 2020.1 can race with Windows TEMP cleanup
# or file scanners, producing a false "Cannot open file" failure.  Keep all
# generated temporary files in a stable, project-owned directory.
set vivado_temp_dir [file join $root_dir "build" "vivado-temp"]
file mkdir $vivado_temp_dir
set ::env(TEMP) $vivado_temp_dir
set ::env(TMP) $vivado_temp_dir
# Vivado 2020.1's MIG PHY regeneration launches a helper process even when
# the parent implementation run uses one job.  On this Windows host the helper
# intermittently fails to open generated XDC or installation Tcl files.  A
# single Vivado worker is slower but deterministic and only affects builds.
set_param general.maxThreads 1

# Continue from a completed synthesis checkpoint.  This is the normal path
# for timing/placement iterations and avoids rebuilding all generated DDR4
# and SmartConnect output products on every attempt.
if { $build_stage eq "resume-impl" } {
    set project_file [file join $project_dir "${project_name}.xpr"]
    if { ![file exists $project_file] } {
        error "Synthesized project is missing: $project_file"
    }
    if { [llength [get_projects -quiet]] > 0 } {
        close_project
    }
    open_project $project_file
    if { [get_property PROGRESS [get_runs synth_1]] ne "100%" } {
        error "synth_1 is not complete; run the synth stage first"
    }
    if { [get_property STATUS [get_runs impl_1]] ne "Not started" } {
        reset_run impl_1
    }
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
    # Keep MIG PHY generation single-job on Windows; its generated XDC is
    # otherwise occasionally opened before the producer has released it.
    launch_runs impl_1 -to_step write_bitstream -jobs 1
    wait_on_run impl_1
    if { [get_property PROGRESS [get_runs impl_1]] ne "100%" } {
        error "impl_1 did not complete"
    }
    open_run impl_1
    report_utilization -file [file join $root_dir "build" "${artifact_stem}_utilization_placed.rpt"]
    report_timing_summary -delay_type min_max -report_unconstrained -max_paths 10 \
        -file [file join $root_dir "build" "${artifact_stem}_timing_summary_routed.rpt"]
    report_bus_skew -file [file join $root_dir "build" "${artifact_stem}_bus_skew_routed.rpt"]
    report_drc -file [file join $root_dir "build" "${artifact_stem}_drc_routed.rpt"]

    set run_bit [file join $project_dir "${project_name}.runs" "impl_1" "${bd_name}_wrapper.bit"]
    if { ![file exists $run_bit] } {
        error "Implementation completed without expected bitstream: $run_bit"
    }
    set candidate_bit [file join $root_dir "build" "bitnet_accel_${artifact_stem}.bit"]
    file copy -force $run_bit $candidate_bit
    puts "Candidate bitstream: $candidate_bit"
    puts "Vivado PL-DDR dual-lane build stage '$build_stage' completed."
    exit 0
}

# Export the hardware platform from an already implemented project without
# deleting and rebuilding it.  This keeps packaging iterations fast while the
# normal "xsa" stage remains the clean, from-scratch reproducibility check.
if { $build_stage eq "export-xsa" } {
    set project_file [file join $project_dir "${project_name}.xpr"]
    set run_bit [file join $project_dir "${project_name}.runs" "impl_1" "${bd_name}_wrapper.bit"]
    if { ![file exists $project_file] } {
        error "Implemented project is missing: $project_file"
    }
    if { ![file exists $run_bit] } {
        error "Implemented bitstream is missing: $run_bit"
    }
    if { [llength [get_projects -quiet]] > 0 } {
        close_project
    }
    open_project $project_file
    if { [get_property PROGRESS [get_runs impl_1]] ne "100%" } {
        error "impl_1 is not complete; run the impl stage before export-xsa"
    }
    open_run impl_1
    report_bus_skew -file [file join $root_dir "build" "${artifact_stem}_bus_skew_routed.rpt"]
    set xsa_file [file join $root_dir "build" "axu3egb_${artifact_stem}.xsa"]
    write_hw_platform -fixed -include_bit -force -file $xsa_file
    puts "Exported hardware platform: $xsa_file"
    puts "Vivado PL-DDR dual-lane build stage '$build_stage' completed."
    exit 0
}

if { ![file exists $ps_bd_tcl] } {
    error "Missing PS reference BD Tcl: $ps_bd_tcl"
}
if { ![file exists $pl_xdc] } {
    error "Missing PL DDR4 constraints: $pl_xdc"
}

if { [llength [get_projects -quiet]] > 0 } {
    close_project
}
file delete -force $project_dir
file mkdir $project_dir
create_project $project_name $project_dir -part xazu3eg-sfvc784-1-i
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

source $ps_bd_tcl
current_bd_design $bd_name
add_files -fileset constrs_1 -norecurse $pl_xdc
add_files -norecurse [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_axis.sv"]
add_files -norecurse [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_dual_axis.sv"]
add_files -norecurse [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_dual_axis_bd.v"]
set_property file_type SystemVerilog [get_files [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_axis.sv"]]
set_property file_type SystemVerilog [get_files [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_dual_axis.sv"]]
set_property file_type Verilog [get_files [file join $root_dir "hardware" "accelerator" "rtl" "bitnet_gemv_dual_axis_bd.v"]]
update_compile_order -fileset sources_1

set ps [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__IRQ {1} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
] $ps
if {$split_memory} {
    set_property -dict [list \
        CONFIG.PSU__USE__S_AXI_GP3 {1} \
        CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    ] $ps
}

set rst_pl [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_pl_0]
set rst_ddr_sys [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_plddr_sys]
set rst_ddr_ui [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_plddr_ui]

set ddr_xgui_file [file join $::env(XILINX_VIVADO) data ip xilinx ddr4_v2_2 xgui ddr4_v2_2.tcl]
puts "DDR4_XGUI_FILE=$ddr_xgui_file readable=[file readable $ddr_xgui_file]"
if {![file readable $ddr_xgui_file]} {
    error "DDR4 IP XGUI file is not readable: $ddr_xgui_file"
}
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
make_bd_intf_pins_external -name c0_sys_clk [get_bd_intf_pins $ddr/C0_SYS_CLK]
set_property CONFIG.FREQ_HZ {200000000} [get_bd_intf_ports c0_sys_clk]
make_bd_intf_pins_external -name c0_ddr4 [get_bd_intf_pins $ddr/C0_DDR4]

set dma0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
  # Keep the 256-bit DMA stream used by the dual GEMV.  The PL DDR AXI width
  # is matched to this master below, so the PL lane avoids a downsizer.
set_property -dict [list \
    CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {1} \
    CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_m_axi_mm2s_data_width {256} CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {256} CONFIG.c_s_axis_s2mm_tdata_width {32} \
      CONFIG.c_mm2s_burst_size {32} CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_sg_length_width {23} CONFIG.c_addr_width {40} \
] $dma0

set dma1 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_1]
set_property -dict [list \
    CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_mm2s_dre {0} CONFIG.c_m_axi_mm2s_data_width {256} \
      CONFIG.c_m_axis_mm2s_tdata_width {256} CONFIG.c_mm2s_burst_size {32} \
    CONFIG.c_sg_length_width {23} CONFIG.c_addr_width {40} \
] $dma1

set dual [create_bd_cell -type module -reference bitnet_gemv_dual_axis_bd bitnet_gemv_dual_axis_0]
set_property -dict [list \
    CONFIG.S_AXIS_DATA_WIDTH {256} CONFIG.M_AXIS_DATA_WIDTH {32} \
    CONFIG.MAX_K {8192} CONFIG.MAX_BATCH_ROWS {16} \
] $dual

set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_ctrl]
# Control endpoints are AXI4-Lite; keep the DMA burst capability from
# triggering SmartConnect's misleading low-area-mode warning.
# Set this before NUM_SI/NUM_MI so the SmartConnect configuration engine sees
# it while the endpoint topology is being instantiated (Vivado 2020.1 emits
# the low-area diagnostic while NUM_MI is applied).
set_property CONFIG.ADVANCED_PROPERTIES {**experimental_features** {disable_low_area_mode 1} __experimental_features__ {disable_low_area_mode 1}} $ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] $ctrl
set mem_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_mem_ps]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $mem_ps
set mem_pl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_mem_pl]
set_property -dict [list CONFIG.NUM_SI [expr {$split_memory ? 2 : 3}] CONFIG.NUM_MI {1}] $mem_pl

if {$split_memory} {
    set mem_ps_read [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_mem_ps_read]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $mem_ps_read
}

set cc1 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_cc_lane1]
set_property CONFIG.DATA_WIDTH {256} $cc1
if {!$split_memory} {
    set cc0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_cc_lane0]
    set_property CONFIG.DATA_WIDTH {256} $cc0
}

set irq [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 dma_irq_concat]
set_property CONFIG.NUM_PORTS {3} $irq
set status_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 plddr_status_concat]
set_property CONFIG.NUM_PORTS {4} $status_concat
set topology_flag [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 plddr_topology_flag]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL [expr {$split_memory ? 1 : 0}]] $topology_flag
set status_gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 plddr_status_gpio]
set_property -dict [list CONFIG.C_GPIO_WIDTH {4} CONFIG.C_ALL_INPUTS {1}] $status_gpio

set pl_clk [get_bd_pins $ps/pl_clk0]
set pl_resetn [get_bd_pins $ps/pl_resetn0]
set ddr_ui_clk [get_bd_pins $ddr/c0_ddr4_ui_clk]
set ddr_ui_rst [get_bd_pins $ddr/c0_ddr4_ui_clk_sync_rst]

connect_bd_net $pl_clk [get_bd_pins $rst_pl/slowest_sync_clk]
connect_bd_net $pl_resetn [get_bd_pins $rst_pl/ext_reset_in]
connect_bd_net $pl_clk [get_bd_pins $rst_ddr_sys/slowest_sync_clk]
connect_bd_net $pl_resetn [get_bd_pins $rst_ddr_sys/ext_reset_in]
connect_bd_net $ddr_ui_clk [get_bd_pins $rst_ddr_ui/slowest_sync_clk]
connect_bd_net $ddr_ui_rst [get_bd_pins $rst_ddr_ui/ext_reset_in]

foreach pin [list axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk axi_dma_0/m_axi_s2mm_aclk \
                 axi_dma_1/s_axi_lite_aclk axi_dma_1/m_axi_mm2s_aclk \
                 axi_smc_ctrl/aclk axi_smc_mem_ps/aclk \
                 zynq_ultra_ps_e_0/saxihp0_fpd_aclk \
                 plddr_status_gpio/s_axi_aclk \
                 bitnet_gemv_dual_axis_0/ap_clk] {
    connect_bd_net $pl_clk [get_bd_pins $pin]
}
connect_bd_net $pl_clk [get_bd_pins $cc1/s_axi_aclk]
connect_bd_net $ddr_ui_clk [get_bd_pins $cc1/m_axi_aclk]
connect_bd_net $ddr_ui_clk [get_bd_pins $mem_pl/aclk]
connect_bd_net $ddr_ui_clk [get_bd_pins $ps/maxihpm0_fpd_aclk]
if {$split_memory} {
    connect_bd_net $pl_clk [get_bd_pins $mem_ps_read/aclk]
    connect_bd_net $pl_clk [get_bd_pins $ps/saxihp1_fpd_aclk]
} else {
    connect_bd_net $pl_clk [get_bd_pins $cc0/s_axi_aclk]
    connect_bd_net $ddr_ui_clk [get_bd_pins $cc0/m_axi_aclk]
}

set rstn_pl [get_bd_pins $rst_pl/peripheral_aresetn]
set rstn_ui [get_bd_pins $rst_ddr_ui/peripheral_aresetn]
foreach pin [list axi_dma_0/axi_resetn axi_dma_1/axi_resetn axi_smc_ctrl/aresetn \
                 axi_smc_mem_ps/aresetn bitnet_gemv_dual_axis_0/ap_resetn \
                 axi_cc_lane1/s_axi_aresetn] {
    connect_bd_net $rstn_pl [get_bd_pins $pin]
}
connect_bd_net $rstn_pl [get_bd_pins $status_gpio/s_axi_aresetn]
connect_bd_net $rstn_ui [get_bd_pins $cc1/m_axi_aresetn]
connect_bd_net $rstn_ui [get_bd_pins $mem_pl/aresetn]
if {$split_memory} {
    connect_bd_net $rstn_pl [get_bd_pins $mem_ps_read/aresetn]
} else {
    connect_bd_net $rstn_pl [get_bd_pins $cc0/s_axi_aresetn]
    connect_bd_net $rstn_ui [get_bd_pins $cc0/m_axi_aresetn]
}
connect_bd_net [get_bd_pins $rst_ddr_sys/peripheral_reset] [get_bd_pins $ddr/sys_rst]
connect_bd_net $rstn_ui [get_bd_pins $ddr/c0_ddr4_aresetn]

connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins $irq/In0]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins $irq/In1]
connect_bd_net [get_bd_pins axi_dma_1/mm2s_introut] [get_bd_pins $irq/In2]
connect_bd_net [get_bd_pins $irq/dout] [get_bd_pins $ps/pl_ps_irq0]
connect_bd_net [get_bd_pins $ddr/c0_init_calib_complete] [get_bd_pins $status_concat/In0]
connect_bd_net [get_bd_pins $dual/busy] [get_bd_pins $status_concat/In1]
connect_bd_net [get_bd_pins $dual/error] [get_bd_pins $status_concat/In2]
connect_bd_net [get_bd_pins $topology_flag/dout] [get_bd_pins $status_concat/In3]
connect_bd_net [get_bd_pins $status_concat/dout] [get_bd_pins $status_gpio/gpio_io_i]

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_LPD] [get_bd_intf_pins $ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M01_AXI] [get_bd_intf_pins axi_dma_1/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M02_AXI] [get_bd_intf_pins $status_gpio/S_AXI]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins $dual/S0_AXIS]
connect_bd_intf_net [get_bd_intf_pins axi_dma_1/M_AXIS_MM2S] [get_bd_intf_pins $dual/S1_AXIS]
connect_bd_intf_net [get_bd_intf_pins $dual/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins $mem_ps/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $mem_ps/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0_FPD]

connect_bd_intf_net [get_bd_intf_pins axi_dma_1/M_AXI_MM2S] [get_bd_intf_pins $cc1/S_AXI]
if {$split_memory} {
    connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins $mem_ps_read/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $mem_ps_read/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP1_FPD]
    connect_bd_intf_net [get_bd_intf_pins $cc1/M_AXI] [get_bd_intf_pins $mem_pl/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $mem_pl/S01_AXI]
} else {
    connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins $cc0/S_AXI]
    connect_bd_intf_net [get_bd_intf_pins $cc0/M_AXI] [get_bd_intf_pins $mem_pl/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $cc1/M_AXI] [get_bd_intf_pins $mem_pl/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $mem_pl/S02_AXI]
}
connect_bd_intf_net [get_bd_intf_pins $mem_pl/M00_AXI] [get_bd_intf_pins $ddr/C0_DDR4_S_AXI]

set ddr_segment [get_bd_addr_segs pl_ddr4_0/C0_DDR4_MEMORY_MAP/C0_DDR4_ADDRESS_BLOCK]
# Map DMA masters before the PS master.  Vivado 2020.1 can mark downstream
# DMA paths as excluded if the shared slave is first assigned from the PS.
set plddr_address_spaces [list \
    [get_bd_addr_spaces axi_dma_1/Data_MM2S] \
    [get_bd_addr_spaces zynq_ultra_ps_e_0/Data]]
if {!$split_memory} {
    lappend plddr_address_spaces [get_bd_addr_spaces axi_dma_0/Data_MM2S]
}
foreach address_space $plddr_address_spaces {
    assign_bd_address -offset 0x000400000000 -range 0x40000000 \
        -target_address_space $address_space $ddr_segment -force
}
if {$split_memory} {
    # DMA0 reads lane-0 packets from the low 2GiB PS DDR aperture through HP1.
    assign_bd_address -offset 0x0000000000 -range 0x80000000 \
        -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] \
        [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP3/HP1_DDR_LOW] -force
}
# DMA0 S2MM writes results into the PS DDR low window.
assign_bd_address -offset 0x0000000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
assign_bd_address -offset 0x80000000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0x80010000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs axi_dma_1/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0x80020000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs plddr_status_gpio/S_AXI/Reg] -force
set plddr_mappings [get_bd_addr_segs -quiet -of $ddr_segment]
set expected_plddr_count [expr {$split_memory ? 2 : 3}]
if {[llength $plddr_mappings] != $expected_plddr_count} {
    error "Expected $expected_plddr_count PL DDR mappings for '$memory_topology', got: $plddr_mappings"
}
set expected_plddr_paths [list \
    */axi_dma_1/Data_MM2S/* \
    */zynq_ultra_ps_e_0/Data/*]
if {!$split_memory} {
    lappend expected_plddr_paths */axi_dma_0/Data_MM2S/*
}
foreach expected_path $expected_plddr_paths {
    if {[lsearch -glob $plddr_mappings $expected_path] < 0} {
        error "Missing PL DDR mapping matching $expected_path in: $plddr_mappings"
    }
}
foreach plddr_mapping $plddr_mappings {
    set plddr_offset [string map {_ ""} [get_property OFFSET $plddr_mapping]]
    if {$plddr_offset != 0x400000000} {
        error "Unexpected PL DDR offset for $plddr_mapping: $plddr_offset"
    }
}
if {$split_memory} {
    set dma0_ps_mappings [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces axi_dma_0/Data_MM2S]]
    if {[lsearch -glob $dma0_ps_mappings */SEG_zynq_ultra_ps_e_0_HP1_DDR_LOW] < 0} {
        error "DMA0 split lane is not mapped to HP1 PS DDR: $dma0_ps_mappings"
    }
}
puts "PLDDR_ADDR_SEGS=[get_bd_addr_segs -of_objects [get_bd_intf_pins $ddr/C0_DDR4_S_AXI]]"
puts "MEMORY_TOPOLOGY=$memory_topology"
puts "DMA0_CTRL_SEGS=[get_bd_addr_segs -of_objects [get_bd_intf_pins axi_dma_0/S_AXI_LITE]]"
puts "DMA1_CTRL_SEGS=[get_bd_addr_segs -of_objects [get_bd_intf_pins axi_dma_1/S_AXI_LITE]]"
validate_bd_design
save_bd_design

set bd_file [get_files */${bd_name}.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrapper_candidates [glob -nocomplain [file join $project_dir "${project_name}.srcs" "sources_1" "bd" $bd_name "hdl" "${bd_name}_wrapper.v"]]
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
    open_run synth_1
    report_utilization -file [file join $root_dir "build" "${artifact_stem}_utilization_synth.rpt"]
}
if { $build_stage eq "impl" || $build_stage eq "bitstream" || $build_stage eq "xsa" } {
    # The dual GEMV lanes are intentionally timed at 100 MHz for the first
    # 64-beat streaming candidate; this preserves functional headroom while
    # the persistent-weight command path is brought up.
    # Explore post-route physical optimization rather than accepting a
    # marginal negative slack from the default implementation strategy.
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
    launch_runs impl_1 -to_step write_bitstream -jobs 1
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
        error "impl_1 did not complete"
    }
    open_run impl_1
    report_utilization -file [file join $root_dir "build" "${artifact_stem}_utilization_placed.rpt"]
    report_timing_summary -delay_type min_max -report_unconstrained -max_paths 10 \
        -file [file join $root_dir "build" "${artifact_stem}_timing_summary_routed.rpt"]
    report_bus_skew -file [file join $root_dir "build" "${artifact_stem}_bus_skew_routed.rpt"]
    report_drc -file [file join $root_dir "build" "${artifact_stem}_drc_routed.rpt"]

    set run_bit [file join $project_dir "${project_name}.runs" "impl_1" "${bd_name}_wrapper.bit"]
    if {![file exists $run_bit]} {
        error "Implementation completed without expected bitstream: $run_bit"
    }
    set candidate_bit [file join $root_dir "build" "bitnet_accel_${artifact_stem}.bit"]
    file copy -force $run_bit $candidate_bit
    puts "Candidate bitstream: $candidate_bit"
}
if { $build_stage eq "xsa" } {
    set xsa_file [file join $root_dir "build" "axu3egb_${artifact_stem}.xsa"]
    write_hw_platform -fixed -include_bit -force -file $xsa_file
    puts "Exported hardware platform: $xsa_file"
}
puts "Vivado dual-lane build stage '$build_stage' completed with '$memory_topology' memory topology."
