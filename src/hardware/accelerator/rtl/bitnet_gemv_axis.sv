`timescale 1ns / 1ps

// AXI-Stream BitNet GEMV/GEMM accelerator for the HuggingFace deployed
// safetensors layout used by bitnet-b1.58-2B-4T-deploy.
//
// Input stream, 32-bit little-endian words on the first 128-bit beat:
//   word 0: magic = 0x42544e31 ("BTN1")
//   word 1: out_features, N, must be a multiple of 4
//   word 2: in_features, K, must be <= MAX_K
//   word 3: flags[1:0]  = ternary code map
//           flags[15:8] = GEMM batch rows, 0/1 means GEMV-compatible one row
//
// Then batch_rows padded int8 activation rows.  Each row is padded to the
// S_AXIS beat width.  Then (N / 4) * K packed weight bytes, group-major and
// k-minor.  Each packed weight byte contains four 2-bit weights for four
// adjacent output channels at the same input index.
//
// Output:
//   batch_rows == 1: original GEMV channel order, N int32 accumulators.
//   batch_rows  > 1: stream-major GEMM order:
//       for output group g, for row r, emit lanes 0..3.
module bitnet_gemv_axis #(
    parameter int S_AXIS_DATA_WIDTH = 128,
    parameter int M_AXIS_DATA_WIDTH = 32,
    parameter int MAX_K = 8192,
    parameter int MAX_BATCH_ROWS = 16
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET ap_rst_n, FREQ_HZ 99999001" *)
    input  logic                         ap_clk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  logic                         ap_rst_n,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input  logic [S_AXIS_DATA_WIDTH-1:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input  logic [(S_AXIS_DATA_WIDTH/8)-1:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input  logic                         s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output logic                         s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input  logic                         s_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output logic [M_AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output logic [(M_AXIS_DATA_WIDTH/8)-1:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output logic                         m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  logic                         m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output logic                         m_axis_tlast,

    output logic                         busy,
    output logic                         error
);
    localparam logic [31:0] MAGIC = 32'h42544e31;

    typedef enum logic [3:0] {
        ST_IDLE          = 4'd0,
        ST_HDR           = 4'd1,
        ST_ACT           = 4'd2,
        ST_WEIGHT_FAST   = 4'd3,
        ST_OUT_FAST      = 4'd4,
        ST_WEIGHT_BATCH  = 4'd5,
        ST_BATCH_READ    = 4'd6,
        ST_BATCH_ACC     = 4'd7,
        ST_BATCH_WRITE   = 4'd8,
        ST_OUT_BATCH     = 4'd9
    } state_t;

    localparam int S_AXIS_BYTES = S_AXIS_DATA_WIDTH / 8;
    localparam int S_AXIS_BYTE_SHIFT = $clog2(S_AXIS_BYTES);
    localparam int ACT_WORDS = (MAX_K + S_AXIS_BYTES - 1) / S_AXIS_BYTES;
    localparam int ACT_TOTAL_WORDS = ACT_WORDS * MAX_BATCH_ROWS;
    localparam int ACT_ADDR_WIDTH = $clog2(ACT_TOTAL_WORDS);
    localparam int BATCH_PAIRS = (MAX_BATCH_ROWS + 1) / 2;

    state_t state;

    logic [31:0] n_reg;
    logic [31:0] k_reg;
    logic [31:0] groups_reg;
    logic [7:0]  batch_rows_reg;
    logic [1:0]  map_mode;

    logic [31:0] act_count;
    logic [7:0]  act_row;
    logic [31:0] weight_k;
    logic [31:0] group_idx;
    logic [1:0]  out_idx;

    logic signed [31:0] acc0;
    logic signed [31:0] acc1;
    logic signed [31:0] acc2;
    logic signed [31:0] acc3;

    logic [ACT_ADDR_WIDTH-1:0] act_rd_addr;
    logic [ACT_ADDR_WIDTH-1:0] act_rd_addr_b;
    logic [S_AXIS_DATA_WIDTH-1:0] act_word_reg;
    logic [S_AXIS_DATA_WIDTH-1:0] act_word_reg_b;
    logic [ACT_ADDR_WIDTH-1:0] act_mem_addr_a;
    logic                      act_mem_en_a;
    logic                      act_mem_en_b;
    logic [0:0]                act_mem_we_a;

    logic [S_AXIS_DATA_WIDTH-1:0] weight_word_pipe0;
    logic [S_AXIS_DATA_WIDTH-1:0] weight_word_pipe1;
    logic [(S_AXIS_DATA_WIDTH/8)-1:0] weight_keep_pipe0;
    logic [(S_AXIS_DATA_WIDTH/8)-1:0] weight_keep_pipe1;
    logic [31:0] weight_k_pipe0;
    logic [31:0] weight_k_pipe1;
    logic        weight_last_pipe0;
    logic        weight_last_pipe1;
    logic        weight_pipe0_valid;
    logic        weight_pipe1_valid;
    logic        weight_input_done;

    logic [S_AXIS_DATA_WIDTH-1:0] batch_weight_word;
    logic [(S_AXIS_DATA_WIDTH/8)-1:0] batch_weight_keep;
    logic [31:0] batch_weight_k;
    logic        batch_weight_last;
    logic [7:0]  batch_calc_row;
    logic [7:0]  batch_out_row;
    logic signed [31:0] batch_acc0_a [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc1_a [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc2_a [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc3_a [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc0_b [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc1_b [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc2_b [0:BATCH_PAIRS-1];
    logic signed [31:0] batch_acc3_b [0:BATCH_PAIRS-1];

    logic signed [31:0] add0;
    logic signed [31:0] add1;
    logic signed [31:0] add2;
    logic signed [31:0] add3;
    logic signed [31:0] add0_b;
    logic signed [31:0] add1_b;
    logic signed [31:0] add2_b;
    logic signed [31:0] add3_b;
    logic signed [31:0] add0_hi;
    logic signed [31:0] add1_hi;
    logic signed [31:0] add2_hi;
    logic signed [31:0] add3_hi;
    logic signed [31:0] add0_b_hi;
    logic signed [31:0] add1_b_hi;
    logic signed [31:0] add2_b_hi;
    logic signed [31:0] add3_b_hi;
    logic signed [31:0] term0_a [0:S_AXIS_BYTES-1];
    logic signed [31:0] term1_a [0:S_AXIS_BYTES-1];
    logic signed [31:0] term2_a [0:S_AXIS_BYTES-1];
    logic signed [31:0] term3_a [0:S_AXIS_BYTES-1];
    logic signed [31:0] term0_b [0:S_AXIS_BYTES-1];
    logic signed [31:0] term1_b [0:S_AXIS_BYTES-1];
    logic signed [31:0] term2_b [0:S_AXIS_BYTES-1];
    logic signed [31:0] term3_b [0:S_AXIS_BYTES-1];
    logic signed [31:0] batch_sum0_a;
    logic signed [31:0] batch_sum1_a;
    logic signed [31:0] batch_sum2_a;
    logic signed [31:0] batch_sum3_a;
    logic signed [31:0] batch_sum0_b;
    logic signed [31:0] batch_sum1_b;
    logic signed [31:0] batch_sum2_b;
    logic signed [31:0] batch_sum3_b;
    logic signed [31:0] batch_sum0_a_hi;
    logic signed [31:0] batch_sum1_a_hi;
    logic signed [31:0] batch_sum2_a_hi;
    logic signed [31:0] batch_sum3_a_hi;
    logic signed [31:0] batch_sum0_b_hi;
    logic signed [31:0] batch_sum1_b_hi;
    logic signed [31:0] batch_sum2_b_hi;
    logic signed [31:0] batch_sum3_b_hi;
    // Batch reduction pipeline.  The 256-bit input word contains four
    // independent 8-byte reduction groups.  Registering those groups before
    // the final reduction keeps the BRAM-to-accumulator path short while
    // preserving one weight word per cycle throughput.
    logic signed [31:0] batch_part0_a [0:3];
    logic signed [31:0] batch_part1_a [0:3];
    logic signed [31:0] batch_part2_a [0:3];
    logic signed [31:0] batch_part3_a [0:3];
    logic signed [31:0] batch_part0_b [0:3];
    logic signed [31:0] batch_part1_b [0:3];
    logic signed [31:0] batch_part2_b [0:3];
    logic signed [31:0] batch_part3_b [0:3];
    logic signed [31:0] batch_part0_a_next [0:3];
    logic signed [31:0] batch_part1_a_next [0:3];
    logic signed [31:0] batch_part2_a_next [0:3];
    logic signed [31:0] batch_part3_a_next [0:3];
    logic signed [31:0] batch_part0_b_next [0:3];
    logic signed [31:0] batch_part1_b_next [0:3];
    logic signed [31:0] batch_part2_b_next [0:3];
    logic signed [31:0] batch_part3_b_next [0:3];
    logic signed [31:0] batch_add0_pipe;
    logic signed [31:0] batch_add1_pipe;
    logic signed [31:0] batch_add2_pipe;
    logic signed [31:0] batch_add3_pipe;
    logic signed [31:0] batch_add0_b_pipe;
    logic signed [31:0] batch_add1_b_pipe;
    logic signed [31:0] batch_add2_b_pipe;
    logic signed [31:0] batch_add3_b_pipe;
    logic signed [31:0] fast_sum0;
    logic signed [31:0] fast_sum1;
    logic signed [31:0] fast_sum2;
    logic signed [31:0] fast_sum3;
    logic signed [31:0] fast_sum0_hi;
    logic signed [31:0] fast_sum1_hi;
    logic signed [31:0] fast_sum2_hi;
    logic signed [31:0] fast_sum3_hi;
    logic fast_sum_valid;
    logic fast_sum_last;
    logic [7:0] batch_write_row;
    logic       batch_write_is_last;
    logic [31:0] input_weight_bytes;
    logic [31:0] batch_input_weight_bytes;
    logic input_fire;
    logic output_fire;

    assign busy = (state != ST_IDLE);
    assign input_fire = s_axis_tvalid && s_axis_tready;
    assign output_fire = m_axis_tvalid && m_axis_tready;
    assign m_axis_tkeep = {(M_AXIS_DATA_WIDTH/8){1'b1}};

    assign act_mem_en_a = ((state == ST_ACT) && input_fire) ||
                          ((state == ST_WEIGHT_FAST) && weight_pipe0_valid) ||
                          (state == ST_BATCH_READ) ||
                          ((state == ST_BATCH_ACC) &&
                           ((batch_calc_row + 8'd2) < batch_rows_reg)) ||
                          ((state == ST_BATCH_WRITE) &&
                           ((batch_calc_row + 8'd2) < batch_rows_reg));
    assign act_mem_en_b = (state == ST_BATCH_READ) ||
                          ((state == ST_BATCH_ACC) &&
                           ((batch_calc_row + 8'd2) < batch_rows_reg)) ||
                          ((state == ST_BATCH_WRITE) &&
                           ((batch_calc_row + 8'd2) < batch_rows_reg));
    assign act_mem_we_a = {((state == ST_ACT) && input_fire)};
    assign act_mem_addr_a = ((state == ST_ACT) && input_fire) ?
                            act_addr(act_row, act_count) : act_rd_addr;

    xpm_memory_tdpram #(
        .ADDR_WIDTH_A(ACT_ADDR_WIDTH),
        .ADDR_WIDTH_B(ACT_ADDR_WIDTH),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(S_AXIS_DATA_WIDTH),
        .BYTE_WRITE_WIDTH_B(S_AXIS_DATA_WIDTH),
        .CASCADE_HEIGHT(0),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(ACT_TOTAL_WORDS * S_AXIS_DATA_WIDTH),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(S_AXIS_DATA_WIDTH),
        .READ_DATA_WIDTH_B(S_AXIS_DATA_WIDTH),
        .READ_LATENCY_A(1),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_A("0"),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(S_AXIS_DATA_WIDTH),
        .WRITE_DATA_WIDTH_B(S_AXIS_DATA_WIDTH),
        .WRITE_MODE_A("read_first"),
        .WRITE_MODE_B("read_first")
    ) u_activation_ram (
        .clka(ap_clk),
        .clkb(ap_clk),
        .ena(act_mem_en_a),
        .enb(act_mem_en_b),
        .wea(act_mem_we_a),
        .web(1'b0),
        .addra(act_mem_addr_a),
        .addrb(act_rd_addr_b),
        .dina(s_axis_tdata),
        .dinb({S_AXIS_DATA_WIDTH{1'b0}}),
        .douta(act_word_reg),
        .doutb(act_word_reg_b),
        .regcea(1'b1),
        .regceb(1'b1),
        .rsta(1'b0),
        .rstb(1'b0),
        .sleep(1'b0),
        .injectdbiterra(1'b0),
        .injectdbiterrb(1'b0),
        .injectsbiterra(1'b0),
        .injectsbiterrb(1'b0),
        .dbiterra(),
        .dbiterrb(),
        .sbiterra(),
        .sbiterrb()
    );

    function automatic logic signed [31:0] ternary_contrib(
        input logic signed [7:0] act,
        input logic [1:0] code,
        input logic [1:0] mode
    );
        begin
            unique case (mode)
                2'd1: begin
                    unique case (code)
                        2'b00: ternary_contrib = -{{24{act[7]}}, act};
                        2'b10: ternary_contrib =  {{24{act[7]}}, act};
                        default: ternary_contrib = 32'sd0;
                    endcase
                end
                default: begin
                    unique case (code)
                        2'b01: ternary_contrib = -{{24{act[7]}}, act};
                        2'b11: ternary_contrib =  {{24{act[7]}}, act};
                        default: ternary_contrib = 32'sd0;
                    endcase
                end
            endcase
        end
    endfunction

    function automatic logic [7:0] get_byte(
        input logic [S_AXIS_DATA_WIDTH-1:0] word,
        input int unsigned idx
    );
        begin
            get_byte = word[idx*8 +: 8];
        end
    endfunction

    function automatic logic [ACT_ADDR_WIDTH-1:0] act_addr(
        input logic [7:0] row,
        input logic [31:0] byte_index
    );
        int unsigned word_index;
        begin
            word_index = (int'(row) * ACT_WORDS) +
                         int'(byte_index >> S_AXIS_BYTE_SHIFT);
            act_addr = word_index[ACT_ADDR_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [31:0] balanced_sum8(
        input logic signed [31:0] terms [0:S_AXIS_BYTES-1],
        input int unsigned base
    );
        begin
            balanced_sum8 =
                ((terms[base + 0] + terms[base + 1]) +
                 (terms[base + 2] + terms[base + 3])) +
                ((terms[base + 4] + terms[base + 5]) +
                 (terms[base + 6] + terms[base + 7]));
        end
    endfunction

    function automatic logic signed [31:0] balanced_sum32(
        input logic signed [31:0] terms [0:S_AXIS_BYTES-1],
        input int unsigned base
    );
        begin
            balanced_sum32 =
                (balanced_sum8(terms, base) +
                 balanced_sum8(terms, base + 8)) +
                (balanced_sum8(terms, base + 16) +
                 balanced_sum8(terms, base + 24));
        end
    endfunction

    // Keep two independent reduction trees.  The 128-bit configuration uses
    // one 8-term tree per half; the 512-bit configuration uses one 32-term
    // tree per half.  This preserves the existing accumulator interface while
    // allowing the wider stream to consume four times as many input bytes per
    // clock.
    function automatic logic signed [31:0] balanced_sum_lo(
        input logic signed [31:0] terms [0:S_AXIS_BYTES-1]
    );
        begin
            balanced_sum_lo = (S_AXIS_BYTES <= 16) ?
                balanced_sum8(terms, 0) : balanced_sum32(terms, 0);
        end
    endfunction

    function automatic logic signed [31:0] balanced_sum_hi(
        input logic signed [31:0] terms [0:S_AXIS_BYTES-1]
    );
        begin
            // A 256-bit stream has 32 byte lanes and balanced_sum32 already
            // reduces all of them in balanced_sum_lo.  The old expression
            // indexed terms[32..63] for this legal width, which was outside
            // the array and made the generated RTL tool-dependent.  Only a
            // wider (512-bit) stream has a second 32-term reduction half.
            if (S_AXIS_BYTES <= 16) begin
                balanced_sum_hi = balanced_sum8(terms, 8);
            end else if (S_AXIS_BYTES <= 32) begin
                balanced_sum_hi = 32'sd0;
            end else begin
                balanced_sum_hi = balanced_sum32(terms, 32);
            end
        end
    endfunction

    always_comb begin
        for (int b = 0; b < S_AXIS_BYTES; b++) begin
            term0_a[b] = 32'sd0;
            term1_a[b] = 32'sd0;
            term2_a[b] = 32'sd0;
            term3_a[b] = 32'sd0;
            term0_b[b] = 32'sd0;
            term1_b[b] = 32'sd0;
            term2_b[b] = 32'sd0;
            term3_b[b] = 32'sd0;
        end

        if ((state == ST_WEIGHT_FAST) && weight_pipe1_valid) begin
            for (int b = 0; b < S_AXIS_BYTES; b++) begin
                if (weight_keep_pipe1[b] && ((weight_k_pipe1 + b) < k_reg)) begin
                    logic [7:0] wbyte;
                    logic signed [7:0] act;

                    wbyte = get_byte(weight_word_pipe1, b);
                    act = get_byte(act_word_reg, b);

                    term0_a[b] = ternary_contrib(act, wbyte[1:0], map_mode);
                    term1_a[b] = ternary_contrib(act, wbyte[3:2], map_mode);
                    term2_a[b] = ternary_contrib(act, wbyte[5:4], map_mode);
                    term3_a[b] = ternary_contrib(act, wbyte[7:6], map_mode);
                end
            end
        end else if ((state == ST_BATCH_ACC) || (state == ST_BATCH_WRITE)) begin
            for (int b = 0; b < S_AXIS_BYTES; b++) begin
                if (batch_weight_keep[b] && ((batch_weight_k + b) < k_reg)) begin
                    logic [7:0] wbyte;
                    logic signed [7:0] act;

                    wbyte = get_byte(batch_weight_word, b);
                    act = get_byte(act_word_reg, b);
                    term0_a[b] = ternary_contrib(act, wbyte[1:0], map_mode);
                    term1_a[b] = ternary_contrib(act, wbyte[3:2], map_mode);
                    term2_a[b] = ternary_contrib(act, wbyte[5:4], map_mode);
                    term3_a[b] = ternary_contrib(act, wbyte[7:6], map_mode);
                    if ((batch_calc_row + 8'd1) < batch_rows_reg) begin
                        act = get_byte(act_word_reg_b, b);
                        term0_b[b] = ternary_contrib(act, wbyte[1:0], map_mode);
                        term1_b[b] = ternary_contrib(act, wbyte[3:2], map_mode);
                        term2_b[b] = ternary_contrib(act, wbyte[5:4], map_mode);
                        term3_b[b] = ternary_contrib(act, wbyte[7:6], map_mode);
                    end
                end
            end
        end

        add0 = balanced_sum_lo(term0_a);
        add1 = balanced_sum_lo(term1_a);
        add2 = balanced_sum_lo(term2_a);
        add3 = balanced_sum_lo(term3_a);
        add0_b = balanced_sum_lo(term0_b);
        add1_b = balanced_sum_lo(term1_b);
        add2_b = balanced_sum_lo(term2_b);
        add3_b = balanced_sum_lo(term3_b);
        add0_hi = balanced_sum_hi(term0_a);
        add1_hi = balanced_sum_hi(term1_a);
        add2_hi = balanced_sum_hi(term2_a);
        add3_hi = balanced_sum_hi(term3_a);
        add0_b_hi = balanced_sum_hi(term0_b);
        add1_b_hi = balanced_sum_hi(term1_b);
        add2_b_hi = balanced_sum_hi(term2_b);
        add3_b_hi = balanced_sum_hi(term3_b);
        for (int p = 0; p < 4; p++) begin
            batch_part0_a_next[p] = 32'sd0;
            batch_part1_a_next[p] = 32'sd0;
            batch_part2_a_next[p] = 32'sd0;
            batch_part3_a_next[p] = 32'sd0;
            batch_part0_b_next[p] = 32'sd0;
            batch_part1_b_next[p] = 32'sd0;
            batch_part2_b_next[p] = 32'sd0;
            batch_part3_b_next[p] = 32'sd0;
            if ((p * 8) < S_AXIS_BYTES) begin
                batch_part0_a_next[p] = balanced_sum8(term0_a, p * 8);
                batch_part1_a_next[p] = balanced_sum8(term1_a, p * 8);
                batch_part2_a_next[p] = balanced_sum8(term2_a, p * 8);
                batch_part3_a_next[p] = balanced_sum8(term3_a, p * 8);
                batch_part0_b_next[p] = balanced_sum8(term0_b, p * 8);
                batch_part1_b_next[p] = balanced_sum8(term1_b, p * 8);
                batch_part2_b_next[p] = balanced_sum8(term2_b, p * 8);
                batch_part3_b_next[p] = balanced_sum8(term3_b, p * 8);
            end
        end
    end

    assign batch_add0_pipe = batch_part0_a[0] + batch_part0_a[1] +
                             batch_part0_a[2] + batch_part0_a[3];
    assign batch_add1_pipe = batch_part1_a[0] + batch_part1_a[1] +
                             batch_part1_a[2] + batch_part1_a[3];
    assign batch_add2_pipe = batch_part2_a[0] + batch_part2_a[1] +
                             batch_part2_a[2] + batch_part2_a[3];
    assign batch_add3_pipe = batch_part3_a[0] + batch_part3_a[1] +
                             batch_part3_a[2] + batch_part3_a[3];
    assign batch_add0_b_pipe = batch_part0_b[0] + batch_part0_b[1] +
                               batch_part0_b[2] + batch_part0_b[3];
    assign batch_add1_b_pipe = batch_part1_b[0] + batch_part1_b[1] +
                               batch_part1_b[2] + batch_part1_b[3];
    assign batch_add2_b_pipe = batch_part2_b[0] + batch_part2_b[1] +
                               batch_part2_b[2] + batch_part2_b[3];
    assign batch_add3_b_pipe = batch_part3_b[0] + batch_part3_b[1] +
                               batch_part3_b[2] + batch_part3_b[3];

    always_comb begin
        input_weight_bytes = 32'd0;
        for (int b = 0; b < S_AXIS_BYTES; b++) begin
            if (s_axis_tkeep[b] && ((weight_k + b) < k_reg)) begin
                input_weight_bytes = input_weight_bytes + 32'd1;
            end
        end
    end

    always_comb begin
        batch_input_weight_bytes = 32'd0;
        for (int b = 0; b < S_AXIS_BYTES; b++) begin
            if (s_axis_tkeep[b] && ((weight_k + b) < k_reg)) begin
                batch_input_weight_bytes = batch_input_weight_bytes + 32'd1;
            end
        end
    end

    always_comb begin
        s_axis_tready = 1'b0;
        if ((state == ST_HDR) || (state == ST_ACT) ||
            ((state == ST_WEIGHT_FAST) && !weight_input_done) ||
            (state == ST_WEIGHT_BATCH)) begin
            s_axis_tready = 1'b1;
        end
    end

    always_comb begin
        unique case (out_idx)
            2'd0: m_axis_tdata = (state == ST_OUT_BATCH) ?
                (batch_out_row[0] ? batch_acc0_b[batch_out_row >> 1] : batch_acc0_a[batch_out_row >> 1]) : acc0;
            2'd1: m_axis_tdata = (state == ST_OUT_BATCH) ?
                (batch_out_row[0] ? batch_acc1_b[batch_out_row >> 1] : batch_acc1_a[batch_out_row >> 1]) : acc1;
            2'd2: m_axis_tdata = (state == ST_OUT_BATCH) ?
                (batch_out_row[0] ? batch_acc2_b[batch_out_row >> 1] : batch_acc2_a[batch_out_row >> 1]) : acc2;
            default: m_axis_tdata = (state == ST_OUT_BATCH) ?
                (batch_out_row[0] ? batch_acc3_b[batch_out_row >> 1] : batch_acc3_a[batch_out_row >> 1]) : acc3;
        endcase
    end

    always_comb begin
        m_axis_tvalid = (state == ST_OUT_FAST) || (state == ST_OUT_BATCH);
        if (state == ST_OUT_BATCH) begin
            m_axis_tlast = (out_idx == 2'd3) &&
                           ((batch_out_row + 8'd1) == batch_rows_reg) &&
                           ((group_idx + 32'd1) == groups_reg);
        end else begin
            m_axis_tlast = (state == ST_OUT_FAST) &&
                           (out_idx == 2'd3) &&
                           ((group_idx + 32'd1) == groups_reg);
        end
    end

    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            state <= ST_IDLE;
            n_reg <= 32'd0;
            k_reg <= 32'd0;
            groups_reg <= 32'd0;
            batch_rows_reg <= 8'd1;
            map_mode <= 2'd0;
            act_count <= 32'd0;
            act_row <= 8'd0;
            weight_k <= 32'd0;
            group_idx <= 32'd0;
            out_idx <= 2'd0;
            acc0 <= 32'sd0;
            acc1 <= 32'sd0;
            acc2 <= 32'sd0;
            acc3 <= 32'sd0;
            act_rd_addr <= '0;
            act_rd_addr_b <= '0;
            weight_word_pipe0 <= '0;
            weight_word_pipe1 <= '0;
            weight_keep_pipe0 <= '0;
            weight_keep_pipe1 <= '0;
            weight_k_pipe0 <= 32'd0;
            weight_k_pipe1 <= 32'd0;
            weight_last_pipe0 <= 1'b0;
            weight_last_pipe1 <= 1'b0;
            weight_pipe0_valid <= 1'b0;
            weight_pipe1_valid <= 1'b0;
            weight_input_done <= 1'b0;
            batch_weight_word <= '0;
            batch_weight_keep <= '0;
            batch_weight_k <= 32'd0;
            batch_weight_last <= 1'b0;
            batch_calc_row <= 8'd0;
            batch_out_row <= 8'd0;
            batch_sum0_a <= 32'sd0;
            batch_sum1_a <= 32'sd0;
            batch_sum2_a <= 32'sd0;
            batch_sum3_a <= 32'sd0;
            batch_sum0_b <= 32'sd0;
            batch_sum1_b <= 32'sd0;
            batch_sum2_b <= 32'sd0;
            batch_sum3_b <= 32'sd0;
            batch_sum0_a_hi <= 32'sd0;
            batch_sum1_a_hi <= 32'sd0;
            batch_sum2_a_hi <= 32'sd0;
            batch_sum3_a_hi <= 32'sd0;
            batch_sum0_b_hi <= 32'sd0;
            batch_sum1_b_hi <= 32'sd0;
            batch_sum2_b_hi <= 32'sd0;
            batch_sum3_b_hi <= 32'sd0;
            for (int p = 0; p < 4; p++) begin
                batch_part0_a[p] <= 32'sd0;
                batch_part1_a[p] <= 32'sd0;
                batch_part2_a[p] <= 32'sd0;
                batch_part3_a[p] <= 32'sd0;
                batch_part0_b[p] <= 32'sd0;
                batch_part1_b[p] <= 32'sd0;
                batch_part2_b[p] <= 32'sd0;
                batch_part3_b[p] <= 32'sd0;
            end
            fast_sum0 <= 32'sd0;
            fast_sum1 <= 32'sd0;
            fast_sum2 <= 32'sd0;
            fast_sum3 <= 32'sd0;
            fast_sum0_hi <= 32'sd0;
            fast_sum1_hi <= 32'sd0;
            fast_sum2_hi <= 32'sd0;
            fast_sum3_hi <= 32'sd0;
            fast_sum_valid <= 1'b0;
            fast_sum_last <= 1'b0;
            batch_write_row <= 8'd0;
            batch_write_is_last <= 1'b0;
            for (int r = 0; r < BATCH_PAIRS; r++) begin
                batch_acc0_a[r] <= 32'sd0;
                batch_acc1_a[r] <= 32'sd0;
                batch_acc2_a[r] <= 32'sd0;
                batch_acc3_a[r] <= 32'sd0;
                batch_acc0_b[r] <= 32'sd0;
                batch_acc1_b[r] <= 32'sd0;
                batch_acc2_b[r] <= 32'sd0;
                batch_acc3_b[r] <= 32'sd0;
            end
            error <= 1'b0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    act_count <= 32'd0;
                    act_row <= 8'd0;
                    weight_k <= 32'd0;
                    group_idx <= 32'd0;
                    out_idx <= 2'd0;
                    acc0 <= 32'sd0;
                    acc1 <= 32'sd0;
                    acc2 <= 32'sd0;
                    acc3 <= 32'sd0;
                    weight_pipe0_valid <= 1'b0;
                    weight_pipe1_valid <= 1'b0;
                    weight_input_done <= 1'b0;
                    if (s_axis_tvalid) begin
                        state <= ST_HDR;
                    end
                end

                ST_HDR: begin
                    if (input_fire) begin
                        logic [7:0] batch_field;
                        batch_field = s_axis_tdata[111:104];

                        if ((s_axis_tdata[31:0] != MAGIC) ||
                            (s_axis_tdata[63:32] == 32'd0) ||
                            (s_axis_tdata[33:32] != 2'b00) ||
                            (s_axis_tdata[95:64] == 32'd0) ||
                            (s_axis_tdata[95:64] > MAX_K) ||
                            (batch_field > MAX_BATCH_ROWS)) begin
                            error <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            error <= 1'b0;
                            n_reg <= s_axis_tdata[63:32];
                            groups_reg <= s_axis_tdata[63:32] >> 2;
                            k_reg <= s_axis_tdata[95:64];
                            map_mode <= s_axis_tdata[97:96];
                            batch_rows_reg <= (batch_field == 8'd0) ? 8'd1 : batch_field;
                            act_count <= 32'd0;
                            act_row <= 8'd0;
                            state <= ST_ACT;
                        end
                    end
                end

                ST_ACT: begin
                    if (input_fire) begin
                        if ((act_count + S_AXIS_BYTES) >= k_reg) begin
                            if ((act_row + 8'd1) < batch_rows_reg) begin
                                act_row <= act_row + 8'd1;
                                act_count <= 32'd0;
                            end else begin
                                weight_k <= 32'd0;
                                group_idx <= 32'd0;
                                acc0 <= 32'sd0;
                                acc1 <= 32'sd0;
                                acc2 <= 32'sd0;
                                acc3 <= 32'sd0;
                                for (int r = 0; r < BATCH_PAIRS; r++) begin
                                    batch_acc0_a[r] <= 32'sd0;
                                    batch_acc1_a[r] <= 32'sd0;
                                    batch_acc2_a[r] <= 32'sd0;
                                    batch_acc3_a[r] <= 32'sd0;
                                    batch_acc0_b[r] <= 32'sd0;
                                    batch_acc1_b[r] <= 32'sd0;
                                    batch_acc2_b[r] <= 32'sd0;
                                    batch_acc3_b[r] <= 32'sd0;
                                end
                                weight_pipe0_valid <= 1'b0;
                                weight_pipe1_valid <= 1'b0;
                                weight_input_done <= 1'b0;
                                fast_sum_valid <= 1'b0;
                                fast_sum_last <= 1'b0;
                                state <= (batch_rows_reg <= 8'd1) ? ST_WEIGHT_FAST : ST_WEIGHT_BATCH;
                                act_count <= act_count + S_AXIS_BYTES;
                            end
                        end else begin
                            act_count <= act_count + S_AXIS_BYTES;
                        end
                    end
                end

                ST_WEIGHT_FAST: begin
                    weight_pipe1_valid <= weight_pipe0_valid;
                    if (weight_pipe0_valid) begin
                        weight_word_pipe1 <= weight_word_pipe0;
                        weight_keep_pipe1 <= weight_keep_pipe0;
                        weight_k_pipe1 <= weight_k_pipe0;
                        weight_last_pipe1 <= weight_last_pipe0;
                    end

                    weight_pipe0_valid <= input_fire;
                    if (input_fire) begin
                        weight_word_pipe0 <= s_axis_tdata;
                        weight_keep_pipe0 <= s_axis_tkeep;
                        weight_k_pipe0 <= weight_k;
                        act_rd_addr <= act_addr(8'd0, weight_k);
                        weight_last_pipe0 <= ((weight_k + input_weight_bytes) >= k_reg);
                        weight_k <= weight_k + input_weight_bytes;
                        if ((weight_k + input_weight_bytes) >= k_reg) begin
                            weight_input_done <= 1'b1;
                        end
                    end else begin
                        weight_last_pipe0 <= 1'b0;
                    end

                    if (fast_sum_valid) begin
                        acc0 <= acc0 + fast_sum0 + fast_sum0_hi;
                        acc1 <= acc1 + fast_sum1 + fast_sum1_hi;
                        acc2 <= acc2 + fast_sum2 + fast_sum2_hi;
                        acc3 <= acc3 + fast_sum3 + fast_sum3_hi;
                    end

                    fast_sum_valid <= weight_pipe1_valid;
                    if (weight_pipe1_valid) begin
                        fast_sum0 <= add0;
                        fast_sum1 <= add1;
                        fast_sum2 <= add2;
                        fast_sum3 <= add3;
                        fast_sum0_hi <= add0_hi;
                        fast_sum1_hi <= add1_hi;
                        fast_sum2_hi <= add2_hi;
                        fast_sum3_hi <= add3_hi;
                        fast_sum_last <= weight_last_pipe1;
                    end

                    if (fast_sum_valid && fast_sum_last) begin
                        out_idx <= 2'd0;
                        weight_pipe0_valid <= 1'b0;
                        weight_pipe1_valid <= 1'b0;
                        state <= ST_OUT_FAST;
                    end
                end

                ST_OUT_FAST: begin
                    if (output_fire) begin
                        if (out_idx == 2'd3) begin
                            if ((group_idx + 32'd1) == groups_reg) begin
                                state <= ST_IDLE;
                            end else begin
                                group_idx <= group_idx + 32'd1;
                                weight_k <= 32'd0;
                                out_idx <= 2'd0;
                                acc0 <= 32'sd0;
                                acc1 <= 32'sd0;
                                acc2 <= 32'sd0;
                                acc3 <= 32'sd0;
                                weight_pipe0_valid <= 1'b0;
                                weight_pipe1_valid <= 1'b0;
                                weight_input_done <= 1'b0;
                                fast_sum_valid <= 1'b0;
                                fast_sum_last <= 1'b0;
                                state <= ST_WEIGHT_FAST;
                            end
                        end else begin
                            out_idx <= out_idx + 2'd1;
                        end
                    end
                end

                ST_WEIGHT_BATCH: begin
                    if (input_fire) begin
                        batch_weight_word <= s_axis_tdata;
                        batch_weight_keep <= s_axis_tkeep;
                        batch_weight_k <= weight_k;
                        batch_weight_last <= ((weight_k + batch_input_weight_bytes) >= k_reg);
                        weight_k <= weight_k + batch_input_weight_bytes;
                        batch_calc_row <= 8'd0;
                        act_rd_addr <= act_addr(8'd0, weight_k);
                        act_rd_addr_b <= act_addr(8'd1, weight_k);
                        state <= ST_BATCH_READ;
                    end
                end

                ST_BATCH_READ: begin
                    if (batch_rows_reg > 8'd2) begin
                        act_rd_addr <= act_addr(8'd2, batch_weight_k);
                    end
                    if (batch_rows_reg > 8'd3) begin
                        act_rd_addr_b <= act_addr(8'd3, batch_weight_k);
                    end
                    state <= ST_BATCH_ACC;
                end

                ST_BATCH_ACC: begin
                    for (int p = 0; p < 4; p++) begin
                        batch_part0_a[p] <= batch_part0_a_next[p];
                        batch_part1_a[p] <= batch_part1_a_next[p];
                        batch_part2_a[p] <= batch_part2_a_next[p];
                        batch_part3_a[p] <= batch_part3_a_next[p];
                        batch_part0_b[p] <= batch_part0_b_next[p];
                        batch_part1_b[p] <= batch_part1_b_next[p];
                        batch_part2_b[p] <= batch_part2_b_next[p];
                        batch_part3_b[p] <= batch_part3_b_next[p];
                    end
                    batch_sum0_a_hi <= add0_hi;
                    batch_sum1_a_hi <= add1_hi;
                    batch_sum2_a_hi <= add2_hi;
                    batch_sum3_a_hi <= add3_hi;
                    batch_sum0_b_hi <= add0_b_hi;
                    batch_sum1_b_hi <= add1_b_hi;
                    batch_sum2_b_hi <= add2_b_hi;
                    batch_sum3_b_hi <= add3_b_hi;
                    batch_write_row <= batch_calc_row;
                    batch_write_is_last <=
                        ((batch_calc_row + 8'd2) >= batch_rows_reg);
                    if ((batch_calc_row + 8'd2) < batch_rows_reg) begin
                        batch_calc_row <= batch_calc_row + 8'd2;
                        if ((batch_calc_row + 8'd4) < batch_rows_reg) begin
                            act_rd_addr <= act_addr(batch_calc_row + 8'd4,
                                                    batch_weight_k);
                        end
                        if ((batch_calc_row + 8'd5) < batch_rows_reg) begin
                            act_rd_addr_b <= act_addr(batch_calc_row + 8'd5,
                                                      batch_weight_k);
                        end
                    end
                    state <= ST_BATCH_WRITE;
                end

                ST_BATCH_WRITE: begin
                    batch_acc0_a[batch_write_row >> 1] <=
                        batch_acc0_a[batch_write_row >> 1] + batch_add0_pipe + batch_sum0_a_hi;
                    batch_acc1_a[batch_write_row >> 1] <=
                        batch_acc1_a[batch_write_row >> 1] + batch_add1_pipe + batch_sum1_a_hi;
                    batch_acc2_a[batch_write_row >> 1] <=
                        batch_acc2_a[batch_write_row >> 1] + batch_add2_pipe + batch_sum2_a_hi;
                    batch_acc3_a[batch_write_row >> 1] <=
                        batch_acc3_a[batch_write_row >> 1] + batch_add3_pipe + batch_sum3_a_hi;
                    if ((batch_write_row + 8'd1) < batch_rows_reg) begin
                        batch_acc0_b[batch_write_row >> 1] <=
                            batch_acc0_b[batch_write_row >> 1] + batch_add0_b_pipe + batch_sum0_b_hi;
                        batch_acc1_b[batch_write_row >> 1] <=
                            batch_acc1_b[batch_write_row >> 1] + batch_add1_b_pipe + batch_sum1_b_hi;
                        batch_acc2_b[batch_write_row >> 1] <=
                            batch_acc2_b[batch_write_row >> 1] + batch_add2_b_pipe + batch_sum2_b_hi;
                        batch_acc3_b[batch_write_row >> 1] <=
                            batch_acc3_b[batch_write_row >> 1] + batch_add3_b_pipe + batch_sum3_b_hi;
                    end

                    if (!batch_write_is_last) begin
                        for (int p = 0; p < 4; p++) begin
                            batch_part0_a[p] <= batch_part0_a_next[p];
                            batch_part1_a[p] <= batch_part1_a_next[p];
                            batch_part2_a[p] <= batch_part2_a_next[p];
                            batch_part3_a[p] <= batch_part3_a_next[p];
                            batch_part0_b[p] <= batch_part0_b_next[p];
                            batch_part1_b[p] <= batch_part1_b_next[p];
                            batch_part2_b[p] <= batch_part2_b_next[p];
                            batch_part3_b[p] <= batch_part3_b_next[p];
                        end
                        batch_sum0_a_hi <= add0_hi;
                        batch_sum1_a_hi <= add1_hi;
                        batch_sum2_a_hi <= add2_hi;
                        batch_sum3_a_hi <= add3_hi;
                        batch_sum0_b_hi <= add0_b_hi;
                        batch_sum1_b_hi <= add1_b_hi;
                        batch_sum2_b_hi <= add2_b_hi;
                        batch_sum3_b_hi <= add3_b_hi;
                        batch_write_row <= batch_calc_row;
                        batch_write_is_last <=
                            ((batch_calc_row + 8'd2) >= batch_rows_reg);
                        if ((batch_calc_row + 8'd2) < batch_rows_reg) begin
                            batch_calc_row <= batch_calc_row + 8'd2;
                            if ((batch_calc_row + 8'd4) < batch_rows_reg) begin
                                act_rd_addr <= act_addr(batch_calc_row + 8'd4,
                                                        batch_weight_k);
                            end
                            if ((batch_calc_row + 8'd5) < batch_rows_reg) begin
                                act_rd_addr_b <= act_addr(batch_calc_row + 8'd5,
                                                          batch_weight_k);
                            end
                        end
                    end else if (batch_weight_last) begin
                        batch_out_row <= 8'd0;
                        out_idx <= 2'd0;
                        state <= ST_OUT_BATCH;
                    end else begin
                        state <= ST_WEIGHT_BATCH;
                    end
                end

                ST_OUT_BATCH: begin
                    if (output_fire) begin
                        if (out_idx == 2'd3) begin
                            if ((batch_out_row + 8'd1) == batch_rows_reg) begin
                                if ((group_idx + 32'd1) == groups_reg) begin
                                    state <= ST_IDLE;
                                end else begin
                                    group_idx <= group_idx + 32'd1;
                                    weight_k <= 32'd0;
                                    batch_out_row <= 8'd0;
                                    out_idx <= 2'd0;
                                    for (int r = 0; r < BATCH_PAIRS; r++) begin
                                        batch_acc0_a[r] <= 32'sd0;
                                        batch_acc1_a[r] <= 32'sd0;
                                        batch_acc2_a[r] <= 32'sd0;
                                        batch_acc3_a[r] <= 32'sd0;
                                        batch_acc0_b[r] <= 32'sd0;
                                        batch_acc1_b[r] <= 32'sd0;
                                        batch_acc2_b[r] <= 32'sd0;
                                        batch_acc3_b[r] <= 32'sd0;
                                    end
                                    state <= ST_WEIGHT_BATCH;
                                end
                            end else begin
                                batch_out_row <= batch_out_row + 8'd1;
                                out_idx <= 2'd0;
                            end
                        end else begin
                            out_idx <= out_idx + 2'd1;
                        end
                    end
                end

                default: begin
                    state <= ST_IDLE;
                    error <= 1'b1;
                end
            endcase
        end
    end
endmodule
