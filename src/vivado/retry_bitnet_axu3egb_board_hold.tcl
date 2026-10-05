foreach required {BITNET_BOARD_HOLD_IN BITNET_BOARD_HOLD_OUT} {
  if {![info exists ::env($required)] || $::env($required) eq ""} {
    error "missing required environment variable $required"
  }
}

set input_dcp [file normalize $::env(BITNET_BOARD_HOLD_IN)]
set out_dir [file normalize $::env(BITNET_BOARD_HOLD_OUT)]
if {![file exists $input_dcp]} {
  error "hold-retry input checkpoint not found: $input_dcp"
}
file mkdir $out_dir

open_checkpoint $input_dcp
set before_setup [get_timing_paths -delay_type max -max_paths 1]
set before_hold [get_timing_paths -delay_type min -max_paths 1]
set before_wns [get_property SLACK $before_setup]
set before_whs [get_property SLACK $before_hold]
if {$before_wns < 0.0} {
  error "hold retry input has setup violations: WNS=$before_wns"
}
puts "BITNET_BOARD_HOLD_RETRY_BEFORE_WNS_NS=$before_wns"
puts "BITNET_BOARD_HOLD_RETRY_BEFORE_WHS_NS=$before_whs"

set retry_mode aggressive_directive
if {[info exists ::env(BITNET_BOARD_HOLD_RETRY_MODE)] &&
    $::env(BITNET_BOARD_HOLD_RETRY_MODE) ne ""} {
  set retry_mode $::env(BITNET_BOARD_HOLD_RETRY_MODE)
}
if {$retry_mode eq "aggressive_directive"} {
  phys_opt_design -directive ExploreWithAggressiveHoldFix
} elseif {$retry_mode eq "aggressive_only"} {
  phys_opt_design -aggressive_hold_fix
} elseif {$retry_mode eq "route_aggressive"} {
  route_design -directive AggressiveExplore
  phys_opt_design -directive ExploreWithHoldFix
} else {
  error "unsupported BITNET_BOARD_HOLD_RETRY_MODE '$retry_mode'"
}
puts "BITNET_BOARD_HOLD_RETRY_MODE=$retry_mode"
write_checkpoint -force [file join $out_dir aggressive_hold_fixed.dcp]
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 100 \
  -file [file join $out_dir timing_aggressive_hold_fixed.rpt]
report_route_status \
  -file [file join $out_dir route_status_aggressive_hold_fixed.rpt]
report_drc -file [file join $out_dir drc_aggressive_hold_fixed.rpt]
report_cdc -details -file [file join $out_dir cdc_aggressive_hold_fixed.rpt]
check_timing -verbose \
  -file [file join $out_dir check_timing_aggressive_hold_fixed.rpt]

set worst_setup_path [get_timing_paths -delay_type max -max_paths 1]
set worst_hold_path [get_timing_paths -delay_type min -max_paths 1]
set wns [expr {[llength $worst_setup_path] ? \
  [get_property SLACK $worst_setup_path] : "NA"}]
set whs [expr {[llength $worst_hold_path] ? \
  [get_property SLACK $worst_hold_path] : "NA"}]
puts "BITNET_BOARD_HOLD_RETRY_WNS_NS=$wns"
puts "BITNET_BOARD_HOLD_RETRY_WHS_NS=$whs"
puts "BITNET_BOARD_HOLD_RETRY_OUT=$out_dir"
exit 0
