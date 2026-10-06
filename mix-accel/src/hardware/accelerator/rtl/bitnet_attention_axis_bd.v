`timescale 1ns / 1ps

module bitnet_attention_axis_bd (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET ap_rst_n, FREQ_HZ 124998749" *)
    input wire ap_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire ap_rst_n,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input wire [63:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input wire [7:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TSTRB" *)
    input wire [7:0] s_axis_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input wire s_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TSTRB" *)
    output wire [3:0] m_axis_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire m_axis_tlast
);
    bitnet_attention_axis u_attention (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .S_AXIS_TDATA(s_axis_tdata),
        .S_AXIS_TVALID(s_axis_tvalid),
        .S_AXIS_TREADY(s_axis_tready),
        .S_AXIS_TKEEP(s_axis_tkeep),
        .S_AXIS_TSTRB(s_axis_tstrb),
        .S_AXIS_TLAST(s_axis_tlast),
        .M_AXIS_TDATA(m_axis_tdata),
        .M_AXIS_TVALID(m_axis_tvalid),
        .M_AXIS_TREADY(m_axis_tready),
        .M_AXIS_TKEEP(m_axis_tkeep),
        .M_AXIS_TSTRB(m_axis_tstrb),
        .M_AXIS_TLAST(m_axis_tlast)
    );
endmodule
