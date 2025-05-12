function [ T_matrix_data_new ] = get_T_matrix_asinh( T_matrix_data )


len = length(T_matrix_data);
T_matrix_data_new = cell(len, 1);
for i = 1:len
    T_matrix_data_new{i}(:,1) = asinh(T_matrix_data{i}(:,1));

    T_matrix_data_new{i}(:,2) = asinh(T_matrix_data{i}(:,2));

    T_matrix_data_new{i}(:,3) = asinh(T_matrix_data{i}(:,3));

    T_matrix_data_new{i}(:,4) = asinh(T_matrix_data{i}(:,4));

    T_matrix_data_new{i}(:,5) = asinh(T_matrix_data{i}(:,5));

    T_matrix_data_new{i}(:,6) = asinh(T_matrix_data{i}(:,6));

    T_matrix_data_new{i}(:,7) = asinh(T_matrix_data{i}(:,7));

    T_matrix_data_new{i}(:,8) = asinh(T_matrix_data{i}(:,8));

    T_matrix_data_new{i}(:,9) = asinh(T_matrix_data{i}(:,9));

    T_matrix_data_new{i}(:,10) = asinh(T_matrix_data{i}(:,10));

end

end