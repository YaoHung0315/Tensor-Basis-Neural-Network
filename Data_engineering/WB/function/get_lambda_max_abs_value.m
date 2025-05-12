function [lambda_1_value, lambda_2_value, lambda_3_value, lambda_4_value, lambda_5_value] = get_lambda_max_abs_value( lambda_1, lambda_2, lambda_3, lambda_4, lambda_5 )
    lambda_1_value = max(abs(lambda_1));
    lambda_2_value = max(abs(lambda_2));
    lambda_3_value = max(abs(lambda_3));
    lambda_4_value = max(abs(lambda_4));
    lambda_5_value = max(abs(lambda_5));
end

