foreach required {BITNET_BOARD_HOLD_IN BITNET_BOARD_HOLD_OUT} {
  if {![info exists ::env($required)] || $::env($required) eq ""} {
    error "missing required environment variable $required"
  }
}

set input_dcp [file normalize $::env(BITNET_BOARD_HOLD_IN)]
set out_dir [file normalize $::env(BITNET_BOARD_HOLD_OUT)]
if {![file exists $input_dcp]} {
  error "hold-closure input checkpoint not found: $input_dcp"
}
file mkdir $out_dir

open_checkpoint $input_dcp
set reference_clock [get_clocks -quiet clk_pl_0]
set slow_clock [get_clocks -quiet clk_out2_design_1_clk_fast_100_0]
set fast_clock [get_clocks -quiet clk_out1_design_1_clk_fast_100_0]
if {[llength $reference_clock] != 1 || [llength $slow_clock] != 1 ||
    [llength $fast_clock] != 1} {
  error "cannot identify the reference and MMCM 50/100 MHz clocks"
}

# Every slow/fast crossing in this checkpoint belongs to BitNet's retained
# bundled-data mailboxes.  Their payload is immutable until a synchronized
# request is observed and the acknowledgement returns.  Only the synchronous
# hold check is inapplicable to these protocol-held paths; setup remains timed.
set native_prefix "design_1_i/bitnet_r10_tp2_0/inst/"
foreach direction [list \
    [list $slow_clock $fast_clock slow_to_fast] \
    [list $fast_clock $slow_clock fast_to_slow]] {
  lassign $direction from_clock to_clock direction_name
  set crossing_paths [get_timing_paths -quiet -from $from_clock -to $to_clock \
    -max_paths 20000 -nworst 1]
  if {[llength $crossing_paths] == 0} {
    error "no $direction_name paths found"
  }
  foreach path $crossing_paths {
    set startpoint [get_property STARTPOINT_PIN $path]
    set endpoint [get_property ENDPOINT_PIN $path]
    if {![string match "${native_prefix}*" $startpoint] ||
        ![string match "${native_prefix}*" $endpoint]} {
      error "unproven $direction_name crossing: $startpoint -> $endpoint"
    }
  }
  set_false_path -hold -from $from_clock -to $to_clock
  puts "BITNET_BOARD_HOLD_${direction_name}_ENDPOINTS=[llength $crossing_paths]"
}

# SmartConnect uses AMD's XPM handshake protocol.  The XPM scoped constraint
# already retains setup as a datapath-only max-delay plus bus-skew check.  Its
# held payload and first synchronizer stage do not have a synchronous hold
# requirement, even though these particular clocks share a physical source.
# Validate every reference-to-slow crossing before adding a pin-pair-specific
# hold exception so an unrelated future board path makes this script fail.
set xpm_paths [get_timing_paths -quiet -from $reference_clock -to $slow_clock \
  -max_paths 1000 -nworst 1]
if {[llength $xpm_paths] == 0} {
  error "no reference-to-slow SmartConnect paths found"
}
set xpm_payload_paths 0
set xpm_sync_stage0_paths 0
foreach path $xpm_paths {
  set startpoint [get_property STARTPOINT_PIN $path]
  set endpoint [get_property ENDPOINT_PIN $path]
  if {[regexp {inst_cdc_handshake/src_hsdata_ff_reg\[[0-9]+\]/C$} \
          $startpoint] &&
      [regexp {inst_cdc_handshake/dest_hsdata_ff_reg\[[0-9]+\]/D$} \
          $endpoint]} {
    incr xpm_payload_paths
  } elseif {[regexp \
      {inst_cdc_handshake/.*/syncstages_ff_reg\[0\]/D$} $endpoint]} {
    incr xpm_sync_stage0_paths
  } else {
    error "unproven reference-to-slow crossing: $startpoint -> $endpoint"
  }
  set_false_path -hold -from [get_pins $startpoint] -to [get_pins $endpoint]
}

puts "BITNET_BOARD_HOLD_XPM_PAYLOAD_PATHS=$xpm_payload_paths"
puts "BITNET_BOARD_HOLD_XPM_SYNC_STAGE0_PATHS=$xpm_sync_stage0_paths"
write_checkpoint -force [file join $out_dir constrained.dcp]
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 100 \
  -file [file join $out_dir timing_constrained.rpt]
report_exceptions -coverage \
  -file [file join $out_dir exception_coverage_constrained.rpt]

phys_opt_design -directive ExploreWithHoldFix
write_checkpoint -force [file join $out_dir hold_fixed.dcp]
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 100 \
  -file [file join $out_dir timing_hold_fixed.rpt]
report_route_status -file [file join $out_dir route_status_hold_fixed.rpt]
report_drc -file [file join $out_dir drc_hold_fixed.rpt]
report_cdc -details -file [file join $out_dir cdc_hold_fixed.rpt]
check_timing -verbose -file [file join $out_dir check_timing_hold_fixed.rpt]

set worst_setup_path [get_timing_paths -delay_type max -max_paths 1]
set worst_hold_path [get_timing_paths -delay_type min -max_paths 1]
set wns [expr {[llength $worst_setup_path] ? \
  [get_property SLACK $worst_setup_path] : "NA"}]
set whs [expr {[llength $worst_hold_path] ? \
  [get_property SLACK $worst_hold_path] : "NA"}]
puts "BITNET_BOARD_HOLD_WNS_NS=$wns"
puts "BITNET_BOARD_HOLD_WHS_NS=$whs"
puts "BITNET_BOARD_HOLD_OUT=$out_dir"
exit 0
