`timescale 1ns / 1ps

// Two-way output-channel partitioned GEMV wrapper.
//
// Each lane receives an independent BTN1 packet whose N field is N/2 and
// whose weight stream contains only that lane's output groups.  Activations
// are duplicated by software into the two PL-DDR input buffers.  Both cores
// therefore execute concurrently; the result stream is lane 0 followed by
// lane 1.  This ordering avoids an on-chip reorder RAM and is consumed by the
// software merge helper.
module bitnet_gemv_dual_axis #(
    parameter int S_AXIS_DATA_WIDTH = 256,
    parameter int M_AXIS_DATA_WIDTH = 32,
    parameter int MAX_K = 8192,
    parameter int MAX_BATCH_ROWS = 16,
    // AXI DMA S2MM consumes one frame.  Keep TLAST only on lane 1 for the
    // concatenated lane0||lane1 result; standalone users can set this to 1
    // to retain a TLAST at the end of each lane.
    parameter bit EMIT_INTERMEDIATE_TLAST = 1'b0
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S0_AXIS:S1_AXIS:M_AXIS, ASSOCIATED_RESET ap_rst_n, FREQ_HZ 99999001" *)
    input  logic ap_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  logic ap_rst_n,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TDATA" *)
    input  logic [S_AXIS_DATA_WIDTH-1:0] s0_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TKEEP" *)
    input  logic [(S_AXIS_DATA_WIDTH/8)-1:0] s0_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TVALID" *)
    input  logic s0_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TREADY" *)
    output logic s0_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S0_AXIS TLAST" *)
    input logic s0_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TDATA" *)
    input logic [S_AXIS_DATA_WIDTH-1:0] s1_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TKEEP" *)
    input logic [(S_AXIS_DATA_WIDTH/8)-1:0] s1_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TVALID" *)
    input logic s1_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TREADY" *)
    output logic s1_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S1_AXIS TLAST" *)
    input logic s1_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output logic [M_AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output logic [(M_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output logic m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input logic m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output logic m_axis_tlast,

    output logic busy,
    output logic error
);
    logic [M_AXIS_DATA_WIDTH-1:0] lane0_tdata;
    logic [(M_AXIS_DATA_WIDTH/8)-1:0] lane0_tkeep;
    logic lane0_tvalid, lane0_tready, lane0_tlast;
    logic lane0_busy, lane0_error;
    logic [M_AXIS_DATA_WIDTH-1:0] lane1_tdata;
    logic [(M_AXIS_DATA_WIDTH/8)-1:0] lane1_tkeep;
    logic lane1_tvalid, lane1_tready, lane1_tlast;
    logic lane1_busy, lane1_error;
    logic lane1_input_valid, lane1_input_ready;

    bitnet_gemv_axis #(
        .S_AXIS_DATA_WIDTH(S_AXIS_DATA_WIDTH),
        .M_AXIS_DATA_WIDTH(M_AXIS_DATA_WIDTH),
        .MAX_K(MAX_K),
        .MAX_BATCH_ROWS(MAX_BATCH_ROWS)
    ) lane0 (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
        .s_axis_tdata(s0_axis_tdata), .s_axis_tkeep(s0_axis_tkeep),
        .s_axis_tvalid(s0_axis_tvalid), .s_axis_tready(s0_axis_tready),
        .s_axis_tlast(s0_axis_tlast),
        .m_axis_tdata(lane0_tdata), .m_axis_tkeep(lane0_tkeep),
        .m_axis_tvalid(lane0_tvalid), .m_axis_tready(lane0_tready),
        .m_axis_tlast(lane0_tlast), .busy(lane0_busy), .error(lane0_error)
    );

    bitnet_gemv_axis #(
        .S_AXIS_DATA_WIDTH(S_AXIS_DATA_WIDTH),
        .M_AXIS_DATA_WIDTH(M_AXIS_DATA_WIDTH),
        .MAX_K(MAX_K),
        .MAX_BATCH_ROWS(MAX_BATCH_ROWS)
    ) lane1 (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
        .s_axis_tdata(s1_axis_tdata), .s_axis_tkeep(s1_axis_tkeep),
        /* Lane 1 is allowed to consume and compute while the output mux
         * drains lane 0.  Its output is held by m_axis_tready until lane 0
         * emits TLAST, so the two MM2S/compute paths overlap safely. */
        .s_axis_tvalid(lane1_input_valid), .s_axis_tready(lane1_input_ready),
        .s_axis_tlast(s1_axis_tlast),
        .m_axis_tdata(lane1_tdata), .m_axis_tkeep(lane1_tkeep),
        .m_axis_tvalid(lane1_tvalid), .m_axis_tready(lane1_tready),
        .m_axis_tlast(lane1_tlast), .busy(lane1_busy), .error(lane1_error)
    );

    // 0: lane 0 selected, 1: lane 1 selected.  The mux does not start lane 1
    // until lane 0's TLAST has actually transferred, so the output stream is
    // lossless even when the S2MM side applies backpressure.
    logic select_lane1;
    always_comb begin
        if (select_lane1) begin
            m_axis_tdata = lane1_tdata;
            m_axis_tkeep = lane1_tkeep;
            m_axis_tvalid = lane1_tvalid;
            m_axis_tlast = lane1_tlast;
        end else begin
            m_axis_tdata = lane0_tdata;
            m_axis_tkeep = lane0_tkeep;
            m_axis_tvalid = lane0_tvalid;
            m_axis_tlast = EMIT_INTERMEDIATE_TLAST ? lane0_tlast : 1'b0;
        end
    end

    assign lane1_input_valid = s1_axis_tvalid;
    assign s1_axis_tready = lane1_input_ready;
    assign lane0_tready = !select_lane1 && m_axis_tready;
    assign lane1_tready = select_lane1 && m_axis_tready;
    assign busy = lane0_busy || lane1_busy || m_axis_tvalid;
    assign error = lane0_error || lane1_error;

    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            select_lane1 <= 1'b0;
        end else if (!select_lane1 && lane0_tvalid && lane0_tready && lane0_tlast) begin
            select_lane1 <= 1'b1;
        end else if (select_lane1 && lane1_tvalid && lane1_tready && lane1_tlast) begin
            select_lane1 <= 1'b0;
        end
    end
endmodule
