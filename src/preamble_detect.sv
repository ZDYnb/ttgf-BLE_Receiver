module preamble_detect # (
    parameter SAMPLE_RATE = 16,
    parameter TRANSITION_ERROR = 1
) (
    input logic clk,
    input logic en,
    input logic resetn,

    input logic data_bit,
    output logic preamble_detected
);

    localparam PREAMBLE_LEN = 8;
    localparam BUFFER_LEN = (PREAMBLE_LEN - 1) * SAMPLE_RATE + 2 * TRANSITION_ERROR - 1;
    localparam NUM_CHECKS = 7;  // Number of transition checks
    
    // Buffer to store when matched filter output transitions occur
    logic [BUFFER_LEN-1:0] transition_buffer;
    logic last_bit;
    
    logic [NUM_CHECKS-1:0] stage1_valid_transitions;
    logic stage1_current_transition;
    
    // Combinational logic for computing valid transitions
    logic [NUM_CHECKS-1:0] valid_transitions;
    logic current_transition;
    int i, j;

    // STAGE 1: COMBINATIONAL (Compute all checks in parallel)
    //
    always_comb begin
        // Current transition
        current_transition = data_bit ^ last_bit;
        
        // Compute all 7 valid_transition checks independently
        for (i = 0; i < NUM_CHECKS; i = i + 1) begin
            valid_transitions[i] = 1'b0;
            // Check within TRANSITION_ERROR window
            for (j = -TRANSITION_ERROR; j <= TRANSITION_ERROR; j = j + 1) begin
                valid_transitions[i] = valid_transitions[i] ^ 
                    transition_buffer[(i+1) * SAMPLE_RATE - 1 + j];
            end
        end
    end

    // STAGE 2: Reduction AND
    always_comb begin
        // Start with current transition
        preamble_detected = stage1_current_transition;
        
        // AND all the registered valid_transition results
        for (i = 0; i < NUM_CHECKS; i = i + 1) begin
            preamble_detected = preamble_detected & stage1_valid_transitions[i];
        end
    end

    // SEQUENTIAL LOGIC (Pipeline registers + buffer)
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            // Reset pipeline stages
            stage1_valid_transitions <= 0;
            stage1_current_transition <= 0;
            
            // Reset buffer and state
            transition_buffer <= {BUFFER_LEN{1'b0}};
            last_bit <= 0;
        end else if (en) begin
            stage1_valid_transitions <= valid_transitions;
            stage1_current_transition <= current_transition;
            
            transition_buffer <= {transition_buffer[BUFFER_LEN-2:0], current_transition};
            last_bit <= data_bit;
        end
    end

endmodule