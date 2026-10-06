`timescale 1ns / 1ps

module tb_bitnet_gemv_axis;
    localparam int K = 8;
    localparam int N = 8;
    // Exercise more than two four-row hardware batches plus a partial tail.
    localparam int B = 9;
    localparam int AXIS_W = 128;
    localparam int AXIS_BYTES = AXIS_W / 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;

    logic [AXIS_W-1:0] s_tdata;
    logic [AXIS_BYTES-1:0] s_tkeep;
    logic s_tvalid;
    logic s_tready;
    logic s_tlast;

    logic [31:0] m_tdata;
    logic [3:0]  m_tkeep;
    logic m_tvalid;
    logic m_tready;
    logic m_tlast;

    logic busy;
    logic error;

    int signed activations [0:K-1];
    int signed gemm_activations [0:B-1][0:K-1];
    int signed weights [0:N-1][0:K-1];
    int signed expected_gemv [0:N-1];
    int signed expected_gemm [0:B-1][0:N-1];
    int signed got [0:(B*N)-1];
    int recv_count;
    int expected_count;

    always #5 clk = ~clk;

    bitnet_gemv_axis #(
        .S_AXIS_DATA_WIDTH(AXIS_W),
        .MAX_K(64),
        .MAX_BATCH_ROWS(16)
    ) dut (
        .ap_clk(clk),
        .ap_rst_n(rst_n),
        .s_axis_tdata(s_tdata),
        .s_axis_tkeep(s_tkeep),
        .s_axis_tvalid(s_tvalid),
        .s_axis_tready(s_tready),
        .s_axis_tlast(s_tlast),
        .m_axis_tdata(m_tdata),
        .m_axis_tkeep(m_tkeep),
        .m_axis_tvalid(m_tvalid),
        .m_axis_tready(m_tready),
        .m_axis_tlast(m_tlast),
        .busy(busy),
        .error(error)
    );

    function automatic logic [1:0] enc_weight(input int signed w);
        begin
            case (w)
                -1: enc_weight = 2'b01;
                 0: enc_weight = 2'b10;
                 1: enc_weight = 2'b11;
                default: enc_weight = 2'b10;
            endcase
        end
    endfunction

    function automatic logic [7:0] pack4w(
        input int signed w0,
        input int signed w1,
        input int signed w2,
        input int signed w3
    );
        begin
            pack4w = {enc_weight(w3), enc_weight(w2), enc_weight(w1), enc_weight(w0)};
        end
    endfunction

    function automatic logic [7:0] i8(input int signed v);
        begin
            i8 = v[7:0];
        end
    endfunction

    task automatic send_beat(
        input logic [AXIS_W-1:0] word,
        input logic [AXIS_BYTES-1:0] keep,
        input logic last
    );
        int cycles;
        begin
            cycles = 0;
            @(negedge clk);
            s_tdata = word;
            s_tkeep = keep;
            s_tlast = last;
            s_tvalid = 1'b1;
            while (!s_tready) begin
                @(negedge clk);
                cycles++;
                if (cycles > 500) begin
                    $error("send_beat timeout");
                    $fatal(1);
                end
            end
            @(posedge clk);
            @(negedge clk);
            s_tvalid = 1'b0;
            s_tlast = 1'b0;
            s_tdata = '0;
            s_tkeep = '0;
        end
    endtask

    task automatic send_header(input int batch_rows);
        logic [31:0] flags;
        begin
            flags = (batch_rows <= 1) ? 32'd0 : (32'(batch_rows) << 8);
            send_beat({flags, 32'(K), 32'(N), 32'h42544e31}, 16'hffff, 1'b0);
        end
    endtask

    task automatic send_activation_row(input int row);
        begin
            if (row < 0) begin
                send_beat({64'd0,
                           i8(activations[7]), i8(activations[6]),
                           i8(activations[5]), i8(activations[4]),
                           i8(activations[3]), i8(activations[2]),
                           i8(activations[1]), i8(activations[0])},
                          16'hffff, 1'b0);
            end else begin
                send_beat({64'd0,
                           i8(gemm_activations[row][7]), i8(gemm_activations[row][6]),
                           i8(gemm_activations[row][5]), i8(gemm_activations[row][4]),
                           i8(gemm_activations[row][3]), i8(gemm_activations[row][2]),
                           i8(gemm_activations[row][1]), i8(gemm_activations[row][0])},
                          16'hffff, 1'b0);
            end
        end
    endtask

    task automatic send_weights;
        for (int g = 0; g < N/4; g++) begin
            logic [127:0] beat;
            logic last;
            beat = '0;
            for (int k = 0; k < K; k++) begin
                beat[k*8 +: 8] = pack4w(weights[g*4 + 0][k],
                                         weights[g*4 + 1][k],
                                         weights[g*4 + 2][k],
                                         weights[g*4 + 3][k]);
            end
            last = (g == (N/4 - 1));
            send_beat(beat, 16'hffff, last);
        end
    endtask

    task automatic wait_outputs(input int count);
        int cycles;
        begin
            cycles = 0;
            while (recv_count < count) begin
                @(posedge clk);
                cycles++;
                if (cycles > 5000) begin
                    $error("output timeout got=%0d expected=%0d", recv_count, count);
                    $fatal(1);
                end
            end
            repeat (4) @(posedge clk);
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            recv_count <= 0;
        end else if (m_tvalid && m_tready) begin
            if (recv_count >= expected_count) begin
                $error("received more output words than expected");
                $fatal(1);
            end
            got[recv_count] <= $signed(m_tdata);
            if ((recv_count != (expected_count - 1)) && m_tlast) begin
                $error("unexpected TLAST at output %0d", recv_count);
                $fatal(1);
            end
            if ((recv_count == (expected_count - 1)) && !m_tlast) begin
                $error("missing TLAST on final output");
                $fatal(1);
            end
            recv_count <= recv_count + 1;
        end
    end

    initial begin
        #200000;
        $error("testbench global timeout");
        $fatal(1);
    end

    initial begin
        s_tdata = '0;
        s_tkeep = '0;
        s_tvalid = 1'b0;
        s_tlast = 1'b0;
        m_tready = 1'b1;
        expected_count = 0;
        recv_count = 0;

        activations[0] =  1;
        activations[1] = -2;
        activations[2] =  3;
        activations[3] =  4;
        activations[4] = -5;
        activations[5] =  6;
        activations[6] =  7;
        activations[7] = -8;

        for (int row = 0; row < B; row++) begin
            for (int k = 0; k < K; k++) begin
                gemm_activations[row][k] = ((row * 7 + k * 5) % 17) - 8;
            end
        end

        for (int n = 0; n < N; n++) begin
            for (int k = 0; k < K; k++) begin
                weights[n][k] = ((n + (2 * k)) % 3) - 1;
            end
        end

        weights[0][0] =  1;
        weights[0][1] = -1;
        weights[1][2] =  1;
        weights[2][3] = -1;
        weights[3][4] =  1;
        weights[4][5] = -1;
        weights[5][6] =  1;
        weights[6][7] = -1;
        weights[7][0] =  0;

        for (int n = 0; n < N; n++) begin
            expected_gemv[n] = 0;
            for (int k = 0; k < K; k++) begin
                expected_gemv[n] += activations[k] * weights[n][k];
            end
        end

        for (int row = 0; row < B; row++) begin
            for (int n = 0; n < N; n++) begin
                expected_gemm[row][n] = 0;
                for (int k = 0; k < K; k++) begin
                    expected_gemm[row][n] += gemm_activations[row][k] * weights[n][k];
                end
            end
        end

        repeat (8) @(posedge clk);
        rst_n <= 1'b1;
        repeat (4) @(posedge clk);

        recv_count = 0;
        expected_count = N;
        send_header(1);
        send_activation_row(-1);
        send_weights();
        wait_outputs(N);

        for (int n = 0; n < N; n++) begin
            if (got[n] !== expected_gemv[n]) begin
                $error("GEMV output[%0d] mismatch: got %0d expected %0d",
                       n, got[n], expected_gemv[n]);
                $fatal(1);
            end
        end

        recv_count = 0;
        expected_count = B * N;
        repeat (8) @(posedge clk);

        send_header(B);
        for (int row = 0; row < B; row++) begin
            send_activation_row(row);
        end
        send_weights();
        wait_outputs(B * N);

        for (int g = 0; g < N/4; g++) begin
            for (int row = 0; row < B; row++) begin
                for (int lane = 0; lane < 4; lane++) begin
                    int n;
                    int idx;
                    n = (g * 4) + lane;
                    idx = (g * B * 4) + (row * 4) + lane;
                    if (got[idx] !== expected_gemm[row][n]) begin
                        $error("GEMM output[row=%0d n=%0d] mismatch: got %0d expected %0d",
                               row, n, got[idx], expected_gemm[row][n]);
                        $fatal(1);
                    end
                end
            end
        end

        if (error) begin
            $error("DUT error flag asserted");
            $fatal(1);
        end

        $display("PASS: bitnet_gemv_axis produced expected GEMV and GEMM outputs");
        repeat (8) @(posedge clk);
        $finish;
    end
endmodule
