function [S_tensor_adj, R_tensor_adj] = get_ori_SR_adj_feature_z_score(S_tensor, R_tensor, S_tensor_mean, S_tensor_std, R_tensor_mean, R_tensor_std)

    [nx, ny] = size(S_tensor);

    ori_cell = S_tensor(:);
    nor_cell = cell(size(ori_cell));
    for j = 1:length(ori_cell)
        tensor = ori_cell{j};
        nor_tensor = zeros(size(tensor));
        for row = 1:3
            for col = 1:3
                mean_val = S_tensor_mean;
                std_val = S_tensor_std;
                nor_tensor(row, col) = (tensor(row, col) - mean_val) / std_val;

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
                mean_val = R_tensor_mean;
                std_val = R_tensor_std;
                nor_tensor(row, col) = (tensor(row, col) - mean_val) / std_val;

                if isnan(nor_tensor(row, col))
                    nor_tensor(row, col) = 0;
                end
            end
        end
        nor_cell{j} = nor_tensor;
    end
    R_tensor_adj = reshape(nor_cell, nx, ny);

end
