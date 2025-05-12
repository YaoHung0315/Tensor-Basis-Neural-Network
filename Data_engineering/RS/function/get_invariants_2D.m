function [ lambda ] = get_invariants_2D( S, R )

lambda_1 = trace(S*S);
lambda_2 = trace(S*S*S);
lambda_3 = trace(R*R);
lambda_4 = trace(R*R*S);
lambda_5 = trace(R*R*S*S);

lambda = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5];

end

