function [ data] = get_double_dot_data( tensor, Strain_rate_tensor, nx, ny )

data = zeros(nx, ny);

for i = 1:nx
    for j = 1:ny

        matrix1 = tensor{i, j};
        matrix2 = Strain_rate_tensor{i, j};
        double_dot_product = trace(matrix1 * matrix2');
        data(i, j) = double_dot_product;
    end
end

end

