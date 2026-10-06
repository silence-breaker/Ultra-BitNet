`timescale 1ns / 1ps

module bitnet_axis_router #(
    parameter integer S_AXIS_DATA_WIDTH = 64,
    parameter integer M_AXIS_DATA_WIDTH = 32
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_GEMV:M_ATTN:S_GEMV:S_ATTN:M_AXIS, ASSOCIATED_RESET ap_rst_n" *)
    input  wire ap_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire ap_rst_n,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input  wire [S_AXIS_DATA_WIDTH-1:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input  wire [(S_AXIS_DATA_WIDTH/8)-1:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TSTRB" *)
    input  wire [(S_AXIS_DATA_WIDTH/8)-1:0] s_axis_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input  wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input  wire s_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_GEMV TDATA" *)
    output wire [S_AXIS_DATA_WIDTH-1:0] m_gemv_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_GEMV TKEEP" *)
    output wire [(S_AXIS_DATA_WIDTH/8)-1:0] m_gemv_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_GEMV TSTRB" *)
    output wire [(S_AXIS_DATA_WIDTH/8)-1:0] m_gemv_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_GEMV TVALID" *)
    output wire m_gemv_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_GEMV TREADY" *)
    input  wire m_gemv_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_GEMV TLAST" *)
    output wire m_gemv_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_ATTN TDATA" *)
    output wire [S_AXIS_DATA_WIDTH-1:0] m_attn_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_ATTN TKEEP" *)
    output wire [(S_AXIS_DATA_WIDTH/8)-1:0] m_attn_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_ATTN TSTRB" *)
    output wire [(S_AXIS_DATA_WIDTH/8)-1:0] m_attn_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_ATTN TVALID" *)
    output wire m_attn_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_ATTN TREADY" *)
    input  wire m_attn_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_ATTN TLAST" *)
    output wire m_attn_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_GEMV TDATA" *)
    input  wire [M_AXIS_DATA_WIDTH-1:0] s_gemv_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_GEMV TKEEP" *)
    input  wire [(M_AXIS_DATA_WIDTH/8)-1:0] s_gemv_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_GEMV TSTRB" *)
    input  wire [(M_AXIS_DATA_WIDTH/8)-1:0] s_gemv_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_GEMV TVALID" *)
    input  wire s_gemv_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_GEMV TREADY" *)
    output wire s_gemv_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_GEMV TLAST" *)
    input  wire s_gemv_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_ATTN TDATA" *)
    input  wire [M_AXIS_DATA_WIDTH-1:0] s_attn_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_ATTN TKEEP" *)
    input  wire [(M_AXIS_DATA_WIDTH/8)-1:0] s_attn_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_ATTN TSTRB" *)
    input  wire [(M_AXIS_DATA_WIDTH/8)-1:0] s_attn_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_ATTN TVALID" *)
    input  wire s_attn_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_ATTN TREADY" *)
    output wire s_attn_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_ATTN TLAST" *)
    input  wire s_attn_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [M_AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [(M_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TSTRB" *)
    output wire [(M_AXIS_DATA_WIDTH/8)-1:0] m_axis_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire m_axis_tlast,

    output wire busy,
    output reg error
);
    localparam [31:0] GEMV_MAGIC = 32'h42544e31;
    localparam [31:0] ATTN_MAGIC = 32'h41544e31;
    localparam [1:0] ROUTE_IDLE = 2'd0;
    localparam [1:0] ROUTE_GEMV = 2'd1;
    localparam [1:0] ROUTE_ATTN = 2'd2;

    reg [1:0] route;
    reg header_valid;
    reg [S_AXIS_DATA_WIDTH-1:0] header_data;
    reg [(S_AXIS_DATA_WIDTH/8)-1:0] header_keep;
    reg [(S_AXIS_DATA_WIDTH/8)-1:0] header_strb;
    reg header_last;

    wire selected_input_ready = (route == ROUTE_GEMV) ? m_gemv_tready :
                                (route == ROUTE_ATTN) ? m_attn_tready : 1'b0;
    wire selected_output_valid = (route == ROUTE_GEMV) ? s_gemv_tvalid :
                                 (route == ROUTE_ATTN) ? s_attn_tvalid : 1'b0;
    wire selected_output_last = (route == ROUTE_GEMV) ? s_gemv_tlast :
                                (route == ROUTE_ATTN) ? s_attn_tlast : 1'b0;

    assign busy = (route != ROUTE_IDLE) || header_valid;
    assign s_axis_tready = (route == ROUTE_IDLE) ? !header_valid :
                           (!header_valid && selected_input_ready);

    assign m_gemv_tdata = header_valid ? header_data : s_axis_tdata;
    assign m_gemv_tkeep = header_valid ? header_keep : s_axis_tkeep;
    assign m_gemv_tstrb = header_valid ? header_strb : s_axis_tstrb;
    assign m_gemv_tlast = header_valid ? header_last : s_axis_tlast;
    assign m_gemv_tvalid = (route == ROUTE_GEMV) &&
                           (header_valid || s_axis_tvalid);

    assign m_attn_tdata = header_valid ? header_data : s_axis_tdata;
    assign m_attn_tkeep = header_valid ? header_keep : s_axis_tkeep;
    assign m_attn_tstrb = header_valid ? header_strb : s_axis_tstrb;
    assign m_attn_tlast = header_valid ? header_last : s_axis_tlast;
    assign m_attn_tvalid = (route == ROUTE_ATTN) &&
                           (header_valid || s_axis_tvalid);

    assign m_axis_tdata = (route == ROUTE_GEMV) ? s_gemv_tdata : s_attn_tdata;
    assign m_axis_tkeep = (route == ROUTE_GEMV) ? s_gemv_tkeep : s_attn_tkeep;
    assign m_axis_tstrb = (route == ROUTE_GEMV) ? s_gemv_tstrb : s_attn_tstrb;
    assign m_axis_tvalid = selected_output_valid;
    assign m_axis_tlast = selected_output_last;
    assign s_gemv_tready = (route == ROUTE_GEMV) && m_axis_tready;
    assign s_attn_tready = (route == ROUTE_ATTN) && m_axis_tready;

    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            route <= ROUTE_IDLE;
            header_valid <= 1'b0;
            header_data <= {S_AXIS_DATA_WIDTH{1'b0}};
            header_keep <= {(S_AXIS_DATA_WIDTH/8){1'b0}};
            header_strb <= {(S_AXIS_DATA_WIDTH/8){1'b0}};
            header_last <= 1'b0;
            error <= 1'b0;
        end else begin
            if ((route == ROUTE_IDLE) && s_axis_tvalid && s_axis_tready) begin
                header_data <= s_axis_tdata;
                header_keep <= s_axis_tkeep;
                header_strb <= s_axis_tstrb;
                header_last <= s_axis_tlast;
                header_valid <= 1'b1;
                error <= 1'b0;
                if (s_axis_tdata[31:0] == GEMV_MAGIC) begin
                    route <= ROUTE_GEMV;
                end else if (s_axis_tdata[31:0] == ATTN_MAGIC) begin
                    route <= ROUTE_ATTN;
                end else begin
                    header_valid <= 1'b0;
                    error <= 1'b1;
                end
            end else if (header_valid && selected_input_ready) begin
                header_valid <= 1'b0;
            end

            if (selected_output_valid && m_axis_tready && selected_output_last) begin
                route <= ROUTE_IDLE;
                header_valid <= 1'b0;
            end
        end
    end
endmodule
