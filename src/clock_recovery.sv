module clock_recovery # (
    parameter SAMPLE_RATE = 16,
    parameter E_K_SHIFT = 2,
    parameter TAU_SHIFT = 11,
    parameter SAMPLE_POS = 2,
    parameter DATA_WIDTH = 4
) (
    input logic clk,
    input logic en,
    input logic resetn,

    input logic signed [DATA_WIDTH-1:0] i_data, q_data,
    input logic preamble_detected,

    output logic symbol_clk
);
    
    localparam PIPELINE_STAGES = 9;

    logic [$clog2(SAMPLE_RATE):0] error_calc_counter, shift_counter;
    logic signed [DATA_WIDTH-1:0] sample_at_0_i, sample_at_0_q;
    logic signed [DATA_WIDTH-1:0] sample_at_2_i, sample_at_2_q;
    logic signed [DATA_WIDTH-1:0] sample_at_16_i, sample_at_16_q;
    logic signed [DATA_WIDTH-1:0] sample_at_18_i, sample_at_18_q;
    logic [4:0] time_counter;

    // Variables to store the inputs to error calculation
    logic signed [DATA_WIDTH-1:0] i_1, q_1, i_2, q_2, i_3, q_3, i_4, q_4;

    // localparam ERROR_RES = 18 + 0;
    localparam ERROR_RES = 16; 
    localparam TAU_RES = ERROR_RES - TAU_SHIFT;
    localparam E_K_RES = ERROR_RES - E_K_SHIFT;
    localparam D_TAU_RES = $clog2(SAMPLE_RATE + 1);



    // STAGE 1: COMBINATIONAL (Sample Extraction)
    always_comb begin
        i_1 = sample_at_16_i;  // I_k[16]
        q_1 = sample_at_16_q;  // Q_k[16]
        i_2 = sample_at_0_i;   // I_k[0]
        q_2 = sample_at_0_q;   // Q_k[0]
        i_3 = sample_at_18_i;  // I_k[18]
        q_3 = sample_at_18_q;  // Q_k[18]
        i_4 = sample_at_2_i;   // I_k[2]
        q_4 = sample_at_2_q;   // Q_k[2]
    end

    // STAGE 2: Squaring (First Pipeline Register)
    // Declare Stage 2 registers
    logic signed [2*DATA_WIDTH-1:0] stage2_i_1_sqr, stage2_q_1_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage2_i_2_sqr, stage2_q_2_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage2_i_3_sqr, stage2_q_3_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage2_i_4_sqr, stage2_q_4_sqr;

    // Also pass through original values for next stage
    logic signed [DATA_WIDTH-1:0] stage2_i_1, stage2_q_1;
    logic signed [DATA_WIDTH-1:0] stage2_i_2, stage2_q_2;
    logic signed [DATA_WIDTH-1:0] stage2_i_3, stage2_q_3;
    logic signed [DATA_WIDTH-1:0] stage2_i_4, stage2_q_4;

    // Combinational logic for squaring
    logic signed [2*DATA_WIDTH-1:0] i_1_sqr, q_1_sqr, i_2_sqr, q_2_sqr;
    logic signed [2*DATA_WIDTH-1:0] i_3_sqr, q_3_sqr, i_4_sqr, q_4_sqr;

    always_comb begin
        // Compute squares (8 multipliers in parallel)
        i_1_sqr = i_1 * i_1;
        q_1_sqr = q_1 * q_1;
        i_2_sqr = i_2 * i_2;
        q_2_sqr = q_2 * q_2;
        i_3_sqr = i_3 * i_3;
        q_3_sqr = q_3 * q_3;
        i_4_sqr = i_4 * i_4;
        q_4_sqr = q_4 * q_4;
    end


    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage2_i_1_sqr <= 0;
            stage2_q_1_sqr <= 0;
            stage2_i_2_sqr <= 0;
            stage2_q_2_sqr <= 0;
            stage2_i_3_sqr <= 0;
            stage2_q_3_sqr <= 0;
            stage2_i_4_sqr <= 0;
            stage2_q_4_sqr <= 0;
            stage2_i_1 <= 0;
            stage2_q_1 <= 0;
            stage2_i_2 <= 0;
            stage2_q_2 <= 0;
            stage2_i_3 <= 0;
            stage2_q_3 <= 0;
            stage2_i_4 <= 0;
            stage2_q_4 <= 0;
        end else if (en) begin
            // Register squared values
            stage2_i_1_sqr <= i_1_sqr;
            stage2_q_1_sqr <= q_1_sqr;
            stage2_i_2_sqr <= i_2_sqr;
            stage2_q_2_sqr <= q_2_sqr;
            stage2_i_3_sqr <= i_3_sqr;
            stage2_q_3_sqr <= q_3_sqr;
            stage2_i_4_sqr <= i_4_sqr;
            stage2_q_4_sqr <= q_4_sqr;
            
            // Pass through original values
            stage2_i_1 <= i_1;
            stage2_q_1 <= q_1;
            stage2_i_2 <= i_2;
            stage2_q_2 <= q_2;
            stage2_i_3 <= i_3;
            stage2_q_3 <= q_3;
            stage2_i_4 <= i_4;
            stage2_q_4 <= q_4;
        end
    end

        // STAGE 3: First Products (i×q products)
    // Declare Stage 3 registers
    logic signed [2*DATA_WIDTH-1:0] stage3_iq_1, stage3_iq_2;
    logic signed [2*DATA_WIDTH-1:0] stage3_iq_3, stage3_iq_4;

    // Pass through squared values (we'll need them in Stage 5)
    logic signed [2*DATA_WIDTH-1:0] stage3_i_1_sqr, stage3_q_1_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage3_i_2_sqr, stage3_q_2_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage3_i_3_sqr, stage3_q_3_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage3_i_4_sqr, stage3_q_4_sqr;

    // Combinational logic for i×q products
    logic signed [2*DATA_WIDTH-1:0] iq_1, iq_2, iq_3, iq_4;

    always_comb begin
        // Compute i×q products (4 multipliers in parallel)
        iq_1 = stage2_i_1 * stage2_q_1;
        iq_2 = stage2_i_2 * stage2_q_2;
        iq_3 = stage2_i_3 * stage2_q_3;
        iq_4 = stage2_i_4 * stage2_q_4;
    end

    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage3_iq_1 <= 0;
            stage3_iq_2 <= 0;
            stage3_iq_3 <= 0;
            stage3_iq_4 <= 0;
            stage3_i_1_sqr <= 0;
            stage3_q_1_sqr <= 0;
            stage3_i_2_sqr <= 0;
            stage3_q_2_sqr <= 0;
            stage3_i_3_sqr <= 0;
            stage3_q_3_sqr <= 0;
            stage3_i_4_sqr <= 0;
            stage3_q_4_sqr <= 0;
        end else if (en) begin
            // Register i×q products
            stage3_iq_1 <= iq_1;
            stage3_iq_2 <= iq_2;
            stage3_iq_3 <= iq_3;
            stage3_iq_4 <= iq_4;
            
            stage3_i_1_sqr <= stage2_i_1_sqr;
            stage3_q_1_sqr <= stage2_q_1_sqr;
            stage3_i_2_sqr <= stage2_i_2_sqr;
            stage3_q_2_sqr <= stage2_q_2_sqr;
            stage3_i_3_sqr <= stage2_i_3_sqr;
            stage3_q_3_sqr <= stage2_q_3_sqr;
            stage3_i_4_sqr <= stage2_i_4_sqr;
            stage3_q_4_sqr <= stage2_q_4_sqr;
        end
    end

    // STAGE 4: Second Products (iq_12, iq_34)
    // Declare Stage 4 registers
    logic signed [ERROR_RES-1:0] stage4_iq_12, stage4_iq_34;
    logic signed [2*DATA_WIDTH-1:0] stage4_i_1_sqr, stage4_q_1_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage4_i_2_sqr, stage4_q_2_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage4_i_3_sqr, stage4_q_3_sqr;
    logic signed [2*DATA_WIDTH-1:0] stage4_i_4_sqr, stage4_q_4_sqr;

    // Combinational logic for second products
    logic signed [ERROR_RES-1:0] iq_12, iq_34;

    always_comb begin
        iq_12 = stage3_iq_1 * stage3_iq_2;
    end

    always_comb begin
        iq_34 = stage3_iq_3 * stage3_iq_4;
    end

    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage4_iq_12 <= 0;
            stage4_iq_34 <= 0;
            stage4_i_1_sqr <= 0;
            stage4_q_1_sqr <= 0;
            stage4_i_2_sqr <= 0;
            stage4_q_2_sqr <= 0;
            stage4_i_3_sqr <= 0;
            stage4_q_3_sqr <= 0;
            stage4_i_4_sqr <= 0;
            stage4_q_4_sqr <= 0;
        end else if (en) begin
            // Register the products
            stage4_iq_12 <= iq_12;
            stage4_iq_34 <= iq_34;
            
            // Pass through squared values
            stage4_i_1_sqr <= stage3_i_1_sqr;
            stage4_q_1_sqr <= stage3_q_1_sqr;
            stage4_i_2_sqr <= stage3_i_2_sqr;
            stage4_q_2_sqr <= stage3_q_2_sqr;
            stage4_i_3_sqr <= stage3_i_3_sqr;
            stage4_q_3_sqr <= stage3_q_3_sqr;
            stage4_i_4_sqr <= stage3_i_4_sqr;
            stage4_q_4_sqr <= stage3_q_4_sqr;
        end
    end


    // STAGE 5: Compute Differences (i² - q²)
    // Declare Stage 5 registers
    logic signed [2*DATA_WIDTH-1:0] stage5_diff_1, stage5_diff_2; 
    logic signed [2*DATA_WIDTH-1:0] stage5_diff_3, stage5_diff_4; 
    logic signed [ERROR_RES-1:0] stage5_iq_12, stage5_iq_34;

    // Combinational logic
    logic signed [2*DATA_WIDTH-1:0] diff_1, diff_2, diff_3, diff_4;

    always_comb begin
        diff_1 = stage4_i_1_sqr - stage4_q_1_sqr;  // (i₁² - q₁²)
        diff_2 = stage4_i_2_sqr - stage4_q_2_sqr;  // (i₂² - q₂²)
        diff_3 = stage4_i_3_sqr - stage4_q_3_sqr;  // (i₃² - q₃²)
        diff_4 = stage4_i_4_sqr - stage4_q_4_sqr;  // (i₄² - q₄²)
    end

    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage5_diff_1 <= 0;
            stage5_diff_2 <= 0;
            stage5_diff_3 <= 0;
            stage5_diff_4 <= 0;
            stage5_iq_12 <= 0;
            stage5_iq_34 <= 0;
        end else if (en) begin
            stage5_diff_1 <= diff_1;
            stage5_diff_2 <= diff_2;
            stage5_diff_3 <= diff_3;
            stage5_diff_4 <= diff_4;
            // Pass through iq values
            stage5_iq_12 <= stage4_iq_12;
            stage5_iq_34 <= stage4_iq_34;
        end
    end

    // STAGE 6: Multiply Differences
    // Declare Stage 6 registers
    logic signed [ERROR_RES-1:0] stage6_prod_12, stage6_prod_34;
    logic signed [ERROR_RES-1:0] stage6_iq_12, stage6_iq_34; 
    // Combinational logic
    logic signed [ERROR_RES-1:0] prod_12, prod_34;

    always_comb begin
        prod_12 = stage5_diff_1 * stage5_diff_2;  // (i₁²-q₁²)(i₂²-q₂²)
        prod_34 = stage5_diff_3 * stage5_diff_4;  // (i₃²-q₃²)(i₄²-q₄²)
    end

    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage6_prod_12 <= 0;
            stage6_prod_34 <= 0;
            stage6_iq_12 <= 0;
            stage6_iq_34 <= 0;
        end else if (en) begin
            stage6_prod_12 <= prod_12;
            stage6_prod_34 <= prod_34;
            // Pass through iq values
            stage6_iq_12 <= stage5_iq_12;
            stage6_iq_34 <= stage5_iq_34;
        end
    end

    // STAGE 7: Shift iq_12 and iq_34
    // Declare Stage 7 registers
    logic signed [ERROR_RES-1:0] stage7_prod_12, stage7_prod_34;
    logic signed [ERROR_RES-1:0] stage7_iq_12_shifted, stage7_iq_34_shifted;

    // Combinational logic
    logic signed [ERROR_RES-1:0] iq_12_shifted, iq_34_shifted;

    always_comb begin
        iq_12_shifted = stage6_iq_12 << 2;  // ×4
        iq_34_shifted = stage6_iq_34 << 2;  // ×4
    end

    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage7_prod_12 <= 0;
            stage7_prod_34 <= 0;
            stage7_iq_12_shifted <= 0;
            stage7_iq_34_shifted <= 0;
        end else if (en) begin
            stage7_prod_12 <= stage6_prod_12;
            stage7_prod_34 <= stage6_prod_34;
            stage7_iq_12_shifted <= iq_12_shifted;
            stage7_iq_34_shifted <= iq_34_shifted;
        end
    end

// STAGE 8: Addition
// Declare Stage 8 registers
    logic signed [ERROR_RES-1:0] stage8_re1, stage8_re2;

    // Combinational logic
    logic signed [ERROR_RES-1:0] re1, re2;

    always_comb begin
        // ONLY additions
        re1 = stage7_prod_12 + stage7_iq_12_shifted;
        re2 = stage7_prod_34 + stage7_iq_34_shifted;
    end

    // Register the results
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            stage8_re1 <= 0;
            stage8_re2 <= 0;
        end else if (en) begin
            stage8_re1 <= re1;
            stage8_re2 <= re2;
        end
    end


    // STAGE 9: Compute Error e_k and Tau
    logic signed [ERROR_RES-1:0] tau_int_1;
    logic signed [TAU_RES-1:0] tau_1;
    logic signed [3:0] dtau;

    // Combinational logic for error calculation
    logic signed [ERROR_RES-1:0] e_k_comb, tau_int_comb;
    logic signed [E_K_RES-1:0] e_k_shifted_comb;
    logic signed [TAU_RES-1:0] tau_comb;

    always_comb begin
        // Compute error
        e_k_comb = stage8_re1 - stage8_re2;
        
        // Shift error for tau update
        e_k_shifted_comb = e_k_comb[ERROR_RES-1:E_K_SHIFT];
        
        // Update tau integral
        tau_int_comb = tau_int_1 - ERROR_RES'($signed(e_k_shifted_comb));

        // Extract tau from integral
        tau_comb = tau_int_comb[ERROR_RES-1:TAU_SHIFT];
    end

logic signed [ERROR_RES-1:0] stage9_tau_int;
logic signed [TAU_RES-1:0] stage9_tau;

always_ff @(posedge clk or negedge resetn) begin
    if (~resetn) begin     
        stage9_tau_int <= 0;
        stage9_tau <= 0;
    end else if (en) begin
        stage9_tau_int <= tau_int_comb;
        stage9_tau <= tau_comb;
    end
end

// Control logic and main state machine
logic do_error_calc;
logic [D_TAU_RES-1:0] shift_counter_p1;
logic symbol_clk_next;

always_comb begin
    // Determine if error calculation is scheduled
    shift_counter_p1 = (shift_counter + 1);
    do_error_calc = (error_calc_counter == 1) | 
                    (shift_counter_p1[D_TAU_RES-2:0] == dtau[D_TAU_RES-2:0]);

    // Output the symbol clock
    symbol_clk_next = (shift_counter == SAMPLE_POS);
end

// Register symbol_clk output
always_ff @(posedge clk or negedge resetn) begin
    if (~resetn) begin
        symbol_clk <= 1'b0;
    end else if (en) begin
        symbol_clk <= symbol_clk_next;
    end
end

    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            tau_int_1 <= 0;
            tau_1 <= 0;
            dtau <= 0;
            shift_counter <= -PIPELINE_STAGES;  // Account for pipeline delay
            error_calc_counter <= 0;
        time_counter <= 0;
        sample_at_0_i <= 0;
        sample_at_0_q <= 0;
        sample_at_2_i <= 0;
        sample_at_2_q <= 0;
        sample_at_16_i <= 0;
        sample_at_16_q <= 0;
        sample_at_18_i <= 0;
        sample_at_18_q <= 0;
        end else if (en) begin
            if (do_error_calc) begin
                // Store tau estimates and calculate dtau
                tau_int_1 <= stage9_tau_int;  //
                tau_1 <= stage9_tau;
                dtau <= (tau_1 - stage9_tau) >>> 0;
                shift_counter <= 0;
            end else begin
                // Increment shift counter
                shift_counter <= shift_counter + 1;
            end
            
            // Decrement error calculation counter
            if (error_calc_counter != 0) begin
                error_calc_counter <= error_calc_counter - 1;
            end else if (preamble_detected) begin
                error_calc_counter <= (SAMPLE_RATE >> 1) - SAMPLE_POS;
            end
            time_counter <= (time_counter == (SAMPLE_RATE + 2)) ? 5'd0 : time_counter + 5'd1;
            // Shift samples in buffer
            case (time_counter)
                5'd0: begin
                    sample_at_0_i <= i_data;
                    sample_at_0_q <= q_data;
                end
                5'd2: begin
                    sample_at_2_i <= i_data;
                    sample_at_2_q <= q_data;
                end
                5'd16: begin
                    sample_at_16_i <= i_data;
                    sample_at_16_q <= q_data;
                end
                5'd18: begin
                    sample_at_18_i <= i_data;
                    sample_at_18_q <= q_data;
                end
                default: begin
                    // Do nothing
                end
            endcase
        end
    end

endmodule
