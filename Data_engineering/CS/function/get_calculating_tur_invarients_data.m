function [ lambda_1, lambda_2, lambda_3, lambda_4, lambda_5 ] = get_calculating_tur_invarients_data(  S_tensor, R_tensor )

len = length(S_tensor);
lambda_1 = zeros(len,1);
lambda_2 = zeros(len,1);
lambda_3 = zeros(len,1);
lambda_4 = zeros(len,1);
lambda_5 = zeros(len,1);

for i = 1:len

    S = S_tensor{i};
    R = R_tensor{i};
    [ lambda ] = get_invariants_2D( S, R );
    lambda_1(i) = lambda(1);
    lambda_2(i) = lambda(2);
    lambda_3(i) = lambda(3);
    lambda_4(i) = lambda(4);
    lambda_5(i) = lambda(5);

    clear S R lambda;

end
    
end

