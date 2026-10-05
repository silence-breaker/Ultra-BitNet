`timescale 1ns / 1ps

// Vivado 2020.1 IP Integrator module-reference wrapper.
// The implementation remains SystemVerilog, while this Verilog wrapper is
// used as the BD top file because that Vivado release does not accept an SV
// source directly as a module-reference top.
module bitnet_gemv_dual_axis_bd #(
    parameter integer S_AXIS_DATA_WIDTH = 256,
    parameter integer M_AXIS_DATA_WIDTH = 32,
    parameter integer MAX_K = 8192,
    parameter integer MAX_BATCH_ROWS = 16,
    parameter integer EMIT_INTERMEDIATE_TLAST = 0
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S0_AXIS:S1_AXIS:M_AXIS, ASSOCIATED_RESET ap_resetn, FREQ_HZ 99999001" *)
    input wire ap_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_resetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire ap_resetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TDATA" *)
    input wire [S_AXIS_DATA_WIDTH-1:0] s0_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TKEEP" *)
    input wire [(S_AXIS_DATA_WIDTH/8)-1:0] s0_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TVALID" *)
    input wire s0_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TREADY" *)
    output wire s0_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TLAST" *)
    input wire s0_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TDATA" *)
    input wire [S_AXIS_DATA_WIDTH-1:0] s1_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TKEEP" *)
    input wire [(S_AXIS_DATA_WIDTH/8)-1:0] s1_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TVALID" *)
    input wire s1_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TREADY" *)
    output wire s1_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TLAST" *)
    input wire s1_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [M_AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [(M_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire m_axis_tlast,

    output wire busy,
    output wire error
);
    bitnet_gemv_dual_axis #(
        .S_AXIS_DATA_WIDTH(S_AXIS_DATA_WIDTH),
        .M_AXIS_DATA_WIDTH(M_AXIS_DATA_WIDTH),
        .MAX_K(MAX_K),
        .MAX_BATCH_ROWS(MAX_BATCH_ROWS),
        .EMIT_INTERMEDIATE_TLAST(EMIT_INTERMEDIATE_TLAST != 0)
    ) u_dual (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_resetn),
        .s0_axis_tdata(s0_axis_tdata),
        .s0_axis_tkeep(s0_axis_tkeep),
        .s0_axis_tvalid(s0_axis_tvalid),
        .s0_axis_tready(s0_axis_tready),
        .s0_axis_tlast(s0_axis_tlast),
        .s1_axis_tdata(s1_axis_tdata),
        .s1_axis_tkeep(s1_axis_tkeep),
        .s1_axis_tvalid(s1_axis_tvalid),
        .s1_axis_tready(s1_axis_tready),
        .s1_axis_tlast(s1_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .busy(busy),
        .error(error)
    );
endmodule
