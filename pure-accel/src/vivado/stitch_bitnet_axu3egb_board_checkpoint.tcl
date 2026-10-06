foreach required {
  BITNET_BOARD_PROJECT
  BITNET_BOARD_TOP_DCP
  BITNET_BOARD_RUN_ROOT
  BITNET_BOARD_STITCH_OUT
} {
  if {![info exists ::env($required)] || $::env($required) eq ""} {
    error "missing required environment variable $required"
  }
}

set project_file [file normalize $::env(BITNET_BOARD_PROJECT)]
set top_dcp [file normalize $::env(BITNET_BOARD_TOP_DCP)]
set run_root [file normalize $::env(BITNET_BOARD_RUN_ROOT)]
set out_dir [file normalize $::env(BITNET_BOARD_STITCH_OUT)]
foreach required_path [list $project_file $top_dcp $run_root] {
  if {![file exists $required_path]} {
    error "required path not found: $required_path"
  }
}
file mkdir $out_dir

# Preserve Vivado's generated XDC scoping before leaving project mode.  The
# Windows run manager is unusable on this host, but its project metadata still
# gives us the exact implementation constraint set and scope.
open_project $project_file
set xdc_records {}
foreach xdc [get_files -all -quiet -filter {
    FILE_TYPE == "XDC" && USED_IN_IMPLEMENTATION}] {
  set xdc_path [file normalize $xdc]
  set xdc_name [file tail $xdc_path]
  # OOC XDCs create placeholder clocks at each partition boundary.  Reading
  # them into the assembled board would split the single PS clock tree into
  # several unrelated logical clocks and make cross-partition timing unsafe.
  # The clock-wizard input clock is likewise inherited from the PS in the
  # board design; only its MMCM-derived output clock should be auto-generated.
  if {[string match "*_ooc.xdc" $xdc_name] || $xdc_name eq "ooc.xdc" ||
      $xdc_name eq "design_1_clk_fast_100_0.xdc"} {
    puts "BITNET_STITCH_SKIP_OOC_XDC=$xdc_path"
    continue
  }
  lappend xdc_records [list \
    [get_property PROCESSING_ORDER $xdc] \
    [get_property SCOPED_TO_REF $xdc] \
    [get_property SCOPED_TO_CELLS $xdc] \
    $xdc_path]
}
close_project

open_checkpoint $top_dcp
set partitions [list \
  [list design_1_i/axi_ctrl \
    design_1_axi_ctrl_0_synth_1/design_1_axi_ctrl_0.dcp] \
  [list design_1_i/clk_fast_100 \
    design_1_clk_fast_100_0_synth_1/design_1_clk_fast_100_0.dcp] \
  [list design_1_i/bitnet_r10_tp2_0 \
    design_1_bitnet_r10_tp2_0_0_synth_1/design_1_bitnet_r10_tp2_0_0.dcp] \
  [list design_1_i/rst_fast_100 \
    design_1_rst_fast_100_0_synth_1/design_1_rst_fast_100_0.dcp] \
  [list design_1_i/rst_slow_50 \
    design_1_rst_slow_50_0_synth_1/design_1_rst_slow_50_0.dcp] \
  [list design_1_i/zynq_ultra_ps_e_0 \
    design_1_zynq_ultra_ps_e_0_0_synth_1/design_1_zynq_ultra_ps_e_0_0.dcp]]

foreach partition $partitions {
  lassign $partition cell relative_dcp
  set partition_dcp [file join $run_root $relative_dcp]
  if {![file exists $partition_dcp]} {
    error "partition checkpoint not found: $partition_dcp"
  }
  puts "BITNET_STITCH_PARTITION=$cell DCP=$partition_dcp"
  read_checkpoint -cell $cell $partition_dcp
}

set black_boxes [get_cells -hierarchical -quiet -filter {IS_BLACKBOX}]
if {[llength $black_boxes] != 0} {
  error "stitched design still has black boxes: $black_boxes"
}

# EARLY constraints must be processed before the single NORMAL BD constraint.
foreach order {EARLY NORMAL LATE} {
  foreach record $xdc_records {
    lassign $record processing_order scope_ref scope_cells xdc_path
    if {$processing_order ne $order} {
      continue
    }
    set command [list read_xdc]
    if {$scope_ref ne ""} {
      lappend command -ref $scope_ref
    }
    if {$scope_cells ne ""} {
      lappend command -cells $scope_cells
    }
    lappend command $xdc_path
    puts "BITNET_STITCH_XDC=$xdc_path ORDER=$order REF=$scope_ref CELLS=$scope_cells"
    {*}$command
  }
}

set clocks [get_clocks -quiet]
puts "BITNET_STITCH_CLOCKS=$clocks"
if {[llength $clocks] != 3} {
  error "stitched design must contain the 50 MHz reference and MMCM 50/100 MHz clocks: $clocks"
}
set reference_clock [get_clocks -quiet clk_pl_0]
set slow_clock [get_clocks -quiet clk_out2_design_1_clk_fast_100_0]
set fast_clock [get_clocks -quiet clk_out1_design_1_clk_fast_100_0]
if {[llength $reference_clock] != 1 || [llength $slow_clock] != 1 ||
    [llength $fast_clock] != 1} {
  error "cannot identify reference and canonical MMCM 50/100 MHz board clocks"
}
# The 50 and 100 MHz accelerator clocks are phase-zero outputs of the same
# MMCM.  Keep the relationship timed: several R10 bundled-data mailboxes use
# held payloads with related-clock request/ack controls, and cutting the two
# clocks globally would hide those paths from implementation.  The default
# one-fast-cycle requirement is deliberately stricter than the two-cycle data
# multicycle described by the SuperTile note; control paths are never relaxed.
puts "BITNET_STITCH_RELATED_CLOCKS=$slow_clock,$fast_clock REFERENCE=$reference_clock"

write_checkpoint -force [file join $out_dir stitched.dcp]
report_clocks -file [file join $out_dir clocks.rpt]
report_clock_interaction -delay_type min_max \
  -file [file join $out_dir clock_interaction.rpt]
report_utilization -hierarchical \
  -file [file join $out_dir utilization_stitched.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 40 \
  -file [file join $out_dir timing_stitched.rpt]
check_timing -verbose -file [file join $out_dir check_timing.rpt]

puts "BITNET_STITCH_DCP=[file join $out_dir stitched.dcp]"
exit 0
