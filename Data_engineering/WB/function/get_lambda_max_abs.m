function [lambda_1_new, lambda_2_new, lambda_3_new, lambda_4_new, lambda_5_new] = get_lambda_max_abs( lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_1_value, lambda_2_value, lambda_3_value, lambda_4_value, lambda_5_value)

    lambda_1_new = lambda_1 ./ lambda_1_value;
    lambda_2_new = lambda_2 ./ lambda_2_value;
    lambda_3_new = lambda_3 ./ lambda_3_value;
    lambda_4_new = lambda_4 ./ lambda_4_value;
    lambda_5_new = lambda_5 ./ lambda_5_value;

end

