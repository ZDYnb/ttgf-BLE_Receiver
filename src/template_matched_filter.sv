module matched_filter #(
    parameter SAMPLE_RATE = #{(samples_per_symbol)},
    parameter DATA_WIDTH = #{(adc_width)}
) (
    input logic clk,
    input logic en,
    input logic resetn,

    input logic signed [DATA_WIDTH-1:0] i_data, q_data,
    output logic demodulated_bit
);

    localparam PIPELINE_STAGES = 1;

    localparam TEMPLATE_WIDTH = #{(template_width)};
    localparam PROD_WIDTH = DATA_WIDTH + TEMPLATE_WIDTH;
    localparam PROD_SUM_WIDTH = $clog2(SAMPLE_RATE) + PROD_WIDTH;
    localparam SQR_WIDTH = 2 * PROD_SUM_WIDTH;
    localparam SCORE_WIDTH = SQR_WIDTH + 1;

/*
    #{(template_table)}
*/

    // Define buffer for input data
    logic signed [SAMPLE_RATE-1:0][DATA_WIDTH-1:0] i_buffer, q_buffer;
    always_ff @(posedge clk or negedge resetn) begin
        if (~resetn) begin
            i_buffer <= 0;
            q_buffer <= 0;
        end else if (en) begin
            i_buffer <= {i_buffer[SAMPLE_RATE-2:0], i_data};
            q_buffer <= {q_buffer[SAMPLE_RATE-2:0], q_data};
        end
    end
    
    // Define score calculation variables
    logic signed [SCORE_WIDTH-1:0] low_score, high_score;
    logic signed [PROD_SUM_WIDTH-1:0] low_i_i_prod_sum, low_i_q_prod_sum, low_q_i_prod_sum, low_q_q_prod_sum, high_i_i_prod_sum, high_i_q_prod_sum, high_q_i_prod_sum, high_q_q_prod_sum;
    logic signed [SQR_WIDTH-1:0] low_i_i_sqr, low_i_q_sqr, low_q_i_sqr, low_q_q_sqr, high_i_i_sqr, high_i_q_sqr, high_q_i_sqr, high_q_q_sqr;

    always_comb begin
        /*verilator lint_off WIDTH*/
        #{(template_product_sum)}
        /*verilator lint_on WIDTH*/

        // Calculate the low score by squaring and summing the products
        low_i_i_sqr = low_i_i_prod_sum * low_i_i_prod_sum;
        low_i_q_sqr = low_i_q_prod_sum * low_i_q_prod_sum;
        low_q_i_sqr = low_q_i_prod_sum * low_q_i_prod_sum;
        low_q_q_sqr = low_q_q_prod_sum * low_q_q_prod_sum;
        low_score = low_i_i_sqr + low_i_q_sqr + low_q_i_sqr + low_q_q_sqr;

        // Calculate the low score by squaring and summing the products
        high_i_i_sqr = high_i_i_prod_sum * high_i_i_prod_sum;
        high_i_q_sqr = high_i_q_prod_sum * high_i_q_prod_sum;
        high_q_i_sqr = high_q_i_prod_sum * high_q_i_prod_sum;
        high_q_q_sqr = high_q_q_prod_sum * high_q_q_prod_sum;
        high_score = high_i_i_sqr + high_i_q_sqr + high_q_i_sqr + high_q_q_sqr;

        // Determine the bit
        demodulated_bit = high_score > low_score;
    end

endmodule
