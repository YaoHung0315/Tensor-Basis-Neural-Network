function [ output_matrix ] = select_special_point_matrix( input_matrix, slice_x, slice_y, x_point, y_point );

matrix = griddata( slice_x, slice_y, input_matrix, x_point, y_point, 'cubic');
logi = isnan(matrix);
matrix(logi) = griddata( slice_x, slice_y, input_matrix, x_point(logi), y_point(logi), 'nearest');
output_matrix = matrix;

end
