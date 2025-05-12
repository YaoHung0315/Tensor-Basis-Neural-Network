function [ output_cell ] = select_special_point_cell( input_cell, X, Y, slice_x, slice_y );

matrix11 = zeros(size(input_cell,1), size(input_cell,2));
matrix12 = zeros(size(input_cell,1), size(input_cell,2));
matrix21 = zeros(size(input_cell,1), size(input_cell,2));
matrix22 = zeros(size(input_cell,1), size(input_cell,2));
for i = 1:size(input_cell,1)
    for j = 1:size(input_cell,2)
        matrix11(i, j) = input_cell{i, j}(1, 1);
        matrix12(i, j) = input_cell{i, j}(1, 2);
        matrix21(i, j) = input_cell{i, j}(2, 1);
        matrix22(i, j) = input_cell{i, j}(2, 2);
    end
end
matrix11 = griddata( X, Y, matrix11, slice_x, slice_y, 'cubic');
matrix12 = griddata( X, Y, matrix12, slice_x, slice_y, 'cubic');
matrix21 = griddata( X, Y, matrix21, slice_x, slice_y, 'cubic');
matrix22 = griddata( X, Y, matrix22, slice_x, slice_y, 'cubic');


output_cell = cell(size(slice_x,1), 1);
for i = 1:size(slice_x)
    data1 = matrix11(i);
    data2 = matrix12(i);
    data3 = matrix21(i);
    data4 = matrix22(i);
    
    combinedData = [data1; data2; data3; data4];
    
    output_cell{i} = combinedData;
end

end