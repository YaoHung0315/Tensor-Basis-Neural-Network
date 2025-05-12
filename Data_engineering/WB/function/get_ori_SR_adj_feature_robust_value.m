function [S_tensor_median, S_tensor_iqr, R_tensor_median, R_tensor_iqr] = get_ori_SR_adj_feature_robust_value(S_tensor, R_tensor)

S_TENSOR_median = zeros(3,3);
S_TENSOR_iqr = zeros(3,3);
R_TENSOR_median = zeros(3,3);
R_TENSOR_iqr = zeros(3,3);
len = length(S_tensor(:));

for row = 1:3
    for col = 1:3
        temp_values_S = zeros(len, 1);
        temp_values_R = zeros(len, 1);
        for i = 1:len
            temp_values_S(i) = S_tensor{i}(row, col);
            temp_values_R(i) = R_tensor{i}(row, col);
        end
        S_TENSOR_median(row, col) = median(temp_values_S);
        S_TENSOR_iqr(row, col) = iqr(temp_values_S);
        
        R_TENSOR_median(row, col) = median(temp_values_R);
        R_TENSOR_iqr(row, col) = iqr(temp_values_R);
    end
end

S_tensor_median = S_TENSOR_median(1,2);
S_tensor_iqr = S_TENSOR_iqr(1,2);
R_tensor_median = R_TENSOR_median(1,2);
R_tensor_iqr = R_TENSOR_iqr(1,2);

end
