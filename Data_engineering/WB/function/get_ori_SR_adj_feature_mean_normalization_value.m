function [S_tensor_max, S_tensor_min, S_tensor_mean, R_tensor_max, R_tensor_min, R_tensor_mean] = get_ori_SR_adj_feature_mean_normalization_value( S_tensor, R_tensor )

S_TENSOR_max = zeros(3,3);
S_TENSOR_min = zeros(3,3);
S_TENSOR_mean = zeros(3,3);
R_TENSOR_max = zeros(3,3);
R_TENSOR_min = zeros(3,3);
R_TENSOR_mean = zeros(3,3);
len = length(S_tensor(:));

for row = 1:3
    for col = 1:3
        temp_values = zeros(len, 1);
        for i = 1:len
            temp_values(i) = S_tensor{i}(row, col);
        end
        S_TENSOR_mean(row, col) = mean(temp_values);
        S_TENSOR_max(row, col) = max(temp_values);
        S_TENSOR_min(row, col) = min(temp_values);
    end
end

for row = 1:3
    for col = 1:3
        temp_values = zeros(len, 1);
        for i = 1:len
            temp_values(i) = R_tensor{i}(row, col);
        end
        R_TENSOR_mean(row, col) = mean(temp_values);
        R_TENSOR_max(row, col) = max(temp_values);
        R_TENSOR_min(row, col) = min(temp_values);
    end
end

S_tensor_max = S_TENSOR_max(1,2);
S_tensor_min = S_TENSOR_min(1,2);
S_tensor_mean = S_TENSOR_mean(1,2);
R_tensor_max = S_TENSOR_max(1,2);
R_tensor_min = S_TENSOR_min(1,2);
R_tensor_mean = S_TENSOR_mean(1,2);

end

