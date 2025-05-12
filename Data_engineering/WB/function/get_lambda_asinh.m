function [lambda_data_new] = get_lambda_asinh( lambda_data )

    lambda_data_new(:,1) = asinh(lambda_data(:,1));
    lambda_data_new(:,2) = asinh(lambda_data(:,2));
    lambda_data_new(:,3) = asinh(lambda_data(:,3));
    lambda_data_new(:,4) = asinh(lambda_data(:,4));
    lambda_data_new(:,5) = asinh(lambda_data(:,5));
    lambda_data_new(:,6) = asinh(lambda_data(:,6));
    lambda_data_new(:,7) = asinh(lambda_data(:,7));
    
end

