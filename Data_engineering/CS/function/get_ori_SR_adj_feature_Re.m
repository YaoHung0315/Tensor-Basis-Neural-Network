function [S_tensor_adj, R_tensor_adj] = get_ori_SR_adj_feature_Re(S_tensor, R_tensor, Re_tau)

    [nx, ny] = size(S_tensor);

    ori_cell = S_tensor(:);
    nor_cell = cell(size(ori_cell));
    for j = 1:length(ori_cell)
        tensor = ori_cell{j};
        nor_tensor = zeros(size(tensor));
        for row = 1:3
            for col = 1:3
                val = Re_tau;
                nor_tensor(row, col) = (tensor(row, col)/val);

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
                val = Re_tau;
                nor_tensor(row, col) = (tensor(row, col)/val);

                if isnan(nor_tensor(row, col))
                    nor_tensor(row, col) = 0;
                end
            end
        end
        nor_cell{j} = nor_tensor;
    end
    R_tensor_adj = reshape(nor_cell, nx, ny);

end
