foreach required {BITNET_BOARD_IMPL_IN BITNET_BOARD_IMPL_OUT BITNET_BOARD_IMPL_STAGE} {
  if {![info exists ::env($required)] || $::env($required) eq ""} {
    error "missing required environment variable $required"
  }
}

set input_dcp [file normalize $::env(BITNET_BOARD_IMPL_IN)]
set out_dir [file normalize $::env(BITNET_BOARD_IMPL_OUT)]
set stage $::env(BITNET_BOARD_IMPL_STAGE)
if {![file exists $input_dcp]} {
  error "implementation input checkpoint not found: $input_dcp"
}
if {$stage ni {place route bitstream}} {
  error "unsupported BITNET_BOARD_IMPL_STAGE '$stage'"
}
file mkdir $out_dir

set place_directive ExtraNetDelay_high
set route_directive AlternateCLBRouting
if {[info exists ::env(BITNET_BOARD_PLACE_DIRECTIVE)] &&
    $::env(BITNET_BOARD_PLACE_DIRECTIVE) ne ""} {
  set place_directive $::env(BITNET_BOARD_PLACE_DIRECTIVE)
}
if {[info exists ::env(BITNET_BOARD_ROUTE_DIRECTIVE)] &&
    $::env(BITNET_BOARD_ROUTE_DIRECTIVE) ne ""} {
  set route_directive $::env(BITNET_BOARD_ROUTE_DIRECTIVE)
}

open_checkpoint $input_dcp
set black_boxes [get_cells -hierarchical -quiet -filter {IS_BLACKBOX}]
if {[llength $black_boxes] != 0} {
  error "implementation input still has black boxes: $black_boxes"
}
set clocks [get_clocks -quiet]
if {[llength $clocks] != 3} {
  error "implementation input must contain the 50 MHz reference and MMCM 50/100 MHz clocks: $clocks"
}
set reference_clock [get_clocks -quiet clk_pl_0]
set slow_clock [get_clocks -quiet clk_out2_design_1_clk_fast_100_0]
set fast_clock [get_clocks -quiet clk_out1_design_1_clk_fast_100_0]
if {[llength $reference_clock] != 1 || [llength $slow_clock] != 1 ||
    [llength $fast_clock] != 1} {
  error "cannot identify reference and canonical MMCM 50/100 MHz implementation clocks: $clocks"
}
set slow_period [get_property PERIOD $slow_clock]
set fast_period [get_property PERIOD $fast_clock]
if {abs($slow_period - 20.0) > 0.01 || abs($fast_period - 10.0) > 0.01} {
  error "unexpected clock periods: slow=$slow_period ns fast=$fast_period ns"
}
puts "BITNET_BOARD_IMPL_STAGE=$stage"
puts "BITNET_BOARD_IMPL_INPUT=$input_dcp"
puts "BITNET_BOARD_IMPL_CLOCKS=$clocks"

if {$stage eq "place"} {
  puts "BITNET_BOARD_PLACE_DIRECTIVE=$place_directive"
  opt_design -directive ExploreWithRemap
  write_checkpoint -force [file join $out_dir optimized.dcp]
  report_utilization -hierarchical \
    -file [file join $out_dir utilization_optimized.rpt]
  report_timing_summary -delay_type min_max -report_unconstrained -max_paths 40 \
    -file [file join $out_dir timing_optimized.rpt]

  place_design -directive $place_directive
  phys_opt_design -directive AggressiveFanoutOpt
  write_checkpoint -force [file join $out_dir placed.dcp]
  report_utilization -hierarchical \
    -file [file join $out_dir utilization_placed.rpt]
  report_timing_summary -delay_type min_max -report_unconstrained -max_paths 80 \
    -file [file join $out_dir timing_placed.rpt]
  report_design_analysis -congestion -complexity \
    -file [file join $out_dir design_analysis_placed.rpt]
  report_high_fanout_nets -timing -load_types -max_nets 200 \
    -file [file join $out_dir high_fanout_nets_placed.rpt]
  report_control_sets -verbose \
    -file [file join $out_dir control_sets_placed.rpt]
  report_methodology -file [file join $out_dir methodology_placed.rpt]
}

if {$stage eq "route"} {
  puts "BITNET_BOARD_ROUTE_DIRECTIVE=$route_directive"
  route_design -directive $route_directive
  phys_opt_design -directive AggressiveExplore
  write_checkpoint -force [file join $out_dir routed.dcp]
  report_utilization -hierarchical \
    -file [file join $out_dir utilization_routed.rpt]
  report_timing_summary -delay_type min_max -report_unconstrained -max_paths 100 \
    -file [file join $out_dir timing_routed.rpt]
  report_route_status -file [file join $out_dir route_status.rpt]
  report_drc -file [file join $out_dir drc_routed.rpt]
  report_methodology -file [file join $out_dir methodology_routed.rpt]
  report_power -file [file join $out_dir power_routed.rpt]
}

if {$stage eq "bitstream"} {
  set unrouted [llength [get_nets -hierarchical -filter {ROUTE_STATUS == UNROUTED}]]
  if {$unrouted != 0} {
    error "cannot write bitstream with $unrouted unrouted nets"
  }
  set bit_setup_path [get_timing_paths -delay_type max -max_paths 1]
  set bit_hold_path [get_timing_paths -delay_type min -max_paths 1]
  set bit_wns [expr {[llength $bit_setup_path] ? \
    [get_property SLACK $bit_setup_path] : -1.0}]
  set bit_whs [expr {[llength $bit_hold_path] ? \
    [get_property SLACK $bit_hold_path] : -1.0}]
  if {$bit_wns < 0.0 || $bit_whs < 0.0} {
    error "cannot write bitstream with timing violations: WNS=$bit_wns WHS=$bit_whs"
  }
  write_bitstream -force [file join $out_dir bitnet_r10_tp2.bit]
}

set worst_setup_path [get_timing_paths -delay_type max -max_paths 1]
set worst_hold_path [get_timing_paths -delay_type min -max_paths 1]
set wns [expr {[llength $worst_setup_path] ? \
  [get_property SLACK $worst_setup_path] : "NA"}]
set whs [expr {[llength $worst_hold_path] ? \
  [get_property SLACK $worst_hold_path] : "NA"}]
puts "BITNET_BOARD_IMPL_WNS_NS=$wns"
puts "BITNET_BOARD_IMPL_WHS_NS=$whs"
puts "BITNET_BOARD_IMPL_OUT=$out_dir"
exit 0
