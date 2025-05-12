function [S_tensor_adj, R_tensor_adj] = get_ori_SR_adj_feature_mean_normalization( S_tensor, R_tensor, S_tensor_max, S_tensor_min, S_tensor_mean, R_tensor_max, R_tensor_min, R_tensor_mean );

    [nx, ny] = size(S_tensor);

    ori_cell = S_tensor(:);
    nor_cell = cell(size(ori_cell));
    for j = 1:length(ori_cell)
        tensor = ori_cell{j};
        nor_tensor = zeros(size(tensor));
        for row = 1:3
            for col = 1:3
                min_val = S_tensor_min;
                max_val = S_tensor_max;
                mean_val = S_tensor_mean;
                nor_tensor(row, col) = (tensor(row, col) - mean_val) / (max_val - min_val);

                if isnan(nor_tensor(row, col))
                    nor_tensor(row, col) = 0;
                end

            end
        end
        nor_cell{j} = nor_tensor;
    end
    S_tensor_adj = reshape(nor_cell, nx, ny);


    ori_cell = R_tensor(:);
    nor_cell = cell(size(ori_cell));
    for j = 1:length(ori_cell)
        tensor = ori_cell{j};
        nor_tensor = zeros(size(tensor));
        for row = 1:3
            for col = 1:3
                min_val = R_tensor_min;
                max_val = R_tensor_max;
                mean_val = R_tensor_mean;
                nor_tensor(row, col) = (tensor(row, col) - mean_val) / (max_val - min_val);

                if isnan(nor_tensor(row, col))
                    nor_tensor(row, col) = 0;
                end

            end
        end
        nor_cell{j} = nor_tensor;
    end
    R_tensor_adj = reshape(nor_cell, nx, ny);

    
end

