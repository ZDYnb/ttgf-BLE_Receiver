module tt_um_BLE_RX (
`ifdef GL_TEST
    input  logic VPWR,
    input  logic VGND,
`endif
    // Tiny Tapeout user interface
    input  logic [7:0] ui_in,    // dedicated inputs
    output logic [7:0] uo_out,   // dedicated outputs
    input  logic [7:0] uio_in,   // bidir inputs
    output logic [7:0] uio_out,  // bidir outputs
    output logic [7:0] uio_oe,   // bidir enables (1=drive)
    input  logic       ena,      // design selected (async to clk)
    input  logic       clk,      // system clock
    input  logic       rst_n     // async active-low reset from pad
);

    assign uio_oe = 8'b1111_1100;

    // ------------------------------------------------------------
    // Reset + enable synchronization (clean, local control signals)
    // ------------------------------------------------------------
    // Make a clean synchronous active-high reset 'rst'
    logic rst_sync1, rst_sync2;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rst_sync1 <= 1'b1;
            rst_sync2 <= 1'b1;
        end else begin
            rst_sync1 <= 1'b0;
            rst_sync2 <= rst_sync1;
        end
    end
    wire rst = rst_sync2;  // synchronous, active-high

    // Two-flop sync for 'ena'
    logic ena_ff1, ena_ff2;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            ena_ff1 <= 1'b0;
            ena_ff2 <= 1'b0;
        end else begin
            ena_ff1 <= ena;
            ena_ff2 <= ena_ff1;
        end
    end
    wire ena_sync = ena_ff2;

    // ------------------------------------------------------------
    // Boundary input registers (shorten pad -> flop paths)
    // ------------------------------------------------------------
    localparam int DATA_WIDTH = 4;

    logic [DATA_WIDTH-1:0] I_BPF_r, Q_BPF_r;
    logic [5:0]            channel_r;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            I_BPF_r   <= '0;
            Q_BPF_r   <= '0;
            channel_r <= '0;
        end else begin
            I_BPF_r   <= ui_in[3:0];   // I on low nibble
            Q_BPF_r   <= ui_in[7:4];   // Q on high nibble
            channel_r <= uio_in[1:0];  // channel from uio_in
        end
    end

    logic [5:0] channel_6bit;

    always_comb begin
        case (channel_r)
            2'b00:   channel_6bit = 6'd37;  // 2402 MHz
            2'b01:   channel_6bit = 6'd38;  // 2426 MHz
            2'b10:   channel_6bit = 6'd39;  // 2480 MHz
            default: channel_6bit = 6'd37;  // Default to channel 37
        endcase
    end
    // ------------------------------------------------------------
    // Core instance (drive with synced reset/enable + registered IO)
    // ------------------------------------------------------------
    logic demod_symbol;
    logic demod_symbol_clk;
    logic packet_detected;
    logic preamble_detected;

    ble_cdr #(
        .SAMPLE_RATE(16),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_top (
        .clk              (clk),
        .en               (ena_sync),
        .resetn           (~rst),        // core expects active-low

        .i_bpf            (I_BPF_r),
        .q_bpf            (Q_BPF_r),

        .demod_symbol     (demod_symbol),
        .demod_symbol_clk (demod_symbol_clk),
        .preamble_detected_out(preamble_detected),

        .channel          (channel_6bit),
        .packet_detected  (packet_detected)
    );

    // ------------------------------------------------------------
    // Boundary output registers (shorten flop -> pad paths)
    // ------------------------------------------------------------
    logic [7:0] uo_out_r;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            uo_out_r <= 8'h00;
        end else begin
            uo_out_r <= {1'b0, rst, ena_sync, clk, preamble_detected, packet_detected, demod_symbol_clk, demod_symbol};
        end
    end
    assign uo_out = uo_out_r;

    assign uio_out = {2'b00, I_BPF_r[3:0], 2'b00}; // bits [1:0] don't matter since they're inputs


    // Avoid unused warnings
    logic _unused = &{1'b0, uio_in[7:2]};
endmodule
