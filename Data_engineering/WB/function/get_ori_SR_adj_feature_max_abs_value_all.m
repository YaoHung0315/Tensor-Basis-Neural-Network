function [S_tensor_max, S_tensor_min, R_tensor_max, R_tensor_min] = get_ori_SR_adj_feature_max_abs_value_test( S_tensor, R_tensor )

S_TENSOR_max = zeros(3,3);
S_TENSOR_min = zeros(3,3);
R_TENSOR_max = zeros(3,3);
R_TENSOR_min = zeros(3,3);
len = length(S_tensor(:));

for row = 1:3
    for col = 1:3
        temp_values = zeros(len, 1);
        for i = 1:len
            temp_values(i) = S_tensor{i}(row, col);
        end
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
        R_TENSOR_max(row, col) = max(temp_values);
        R_TENSOR_min(row, col) = min(temp_values);
    end
end

S_tensor_max = S_TENSOR_max;
S_tensor_min = S_TENSOR_min;
R_tensor_max = R_TENSOR_max;
R_tensor_min = R_TENSOR_min;

end

