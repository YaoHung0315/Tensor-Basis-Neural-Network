function [S_tensor_mean, S_tensor_std, R_tensor_mean, R_tensor_std] = get_ori_SR_adj_feature_z_score_value( S_tensor, R_tensor )

S_TENSOR_mean = zeros(3,3);
S_TENSOR_std = zeros(3,3);
R_TENSOR_mean = zeros(3,3);
R_TENSOR_std = zeros(3,3);
len = length(S_tensor(:));

for row = 1:3
    for col = 1:3
        temp_values = zeros(len, 1);
        for i = 1:len
            temp_values(i) = S_tensor{i}(row, col);
        end
        S_TENSOR_mean(row, col) = mean(temp_values);
        S_TENSOR_std(row, col) = std(temp_values);
    end
end

for row = 1:3
    for col = 1:3
        temp_values = zeros(len, 1);
        for i = 1:len
            temp_values(i) = R_tensor{i}(row, col);
        end
        R_TENSOR_mean(row, col) = mean(temp_values);
        R_TENSOR_std(row, col) = std(temp_values);
    end
end

S_tensor_mean = S_TENSOR_mean(1,2);
S_tensor_std = S_TENSOR_std(1,2);
R_tensor_mean = S_TENSOR_mean(1,2);
R_tensor_std = S_TENSOR_std(1,2);

end
