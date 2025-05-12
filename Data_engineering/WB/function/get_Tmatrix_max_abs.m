function [T_matrix_1_new, T_matrix_2_new, T_matrix_3_new, T_matrix_4_new, T_matrix_5_new, T_matrix_6_new, T_matrix_7_new, T_matrix_8_new, T_matrix_9_new, T_matrix_10_new] = ...
    get_Tmatrix_max_abs( T_matrix_1, T_matrix_2, T_matrix_3, T_matrix_4, T_matrix_5, T_matrix_6, T_matrix_7, T_matrix_8, T_matrix_9, T_matrix_10, ...
    T_matrix_1_value, T_matrix_2_value, T_matrix_3_value, T_matrix_4_value, T_matrix_5_value, T_matrix_6_value, T_matrix_7_value, T_matrix_8_value, T_matrix_9_value, T_matrix_10_value)

    len = length(T_matrix_1);
    T_matrix_1_new = cell(len, 1);
    for i = 1:len
        T_matrix_1_new{i} = T_matrix_1{i} ./ T_matrix_1_value;
    end

    len = length(T_matrix_2);
    T_matrix_2_new = cell(len, 1);
    for i = 1:len
        T_matrix_2_new{i} = T_matrix_2{i} ./ T_matrix_2_value;
    end

    len = length(T_matrix_3);
    T_matrix_3_new = cell(len, 1);
    for i = 1:len
        T_matrix_3_new{i} = T_matrix_3{i} ./ T_matrix_3_value;
    end

    len = length(T_matrix_4);
    T_matrix_4_new = cell(len, 1);
    for i = 1:len
        T_matrix_4_new{i} = T_matrix_4{i} ./ T_matrix_4_value;
    end

    len = length(T_matrix_5);
    T_matrix_5_new = cell(len, 1);
    for i = 1:len
        T_matrix_5_new{i} = T_matrix_5{i} ./ T_matrix_5_value;
    end

    len = length(T_matrix_6);
    T_matrix_6_new = cell(len, 1);
    for i = 1:len
        T_matrix_6_new{i} = T_matrix_6{i} ./ T_matrix_6_value;
    end

    len = length(T_matrix_7);
    T_matrix_7_new = cell(len, 1);
    for i = 1:len
        T_matrix_7_new{i} = T_matrix_7{i} ./ T_matrix_7_value;
    end

    len = length(T_matrix_8);
    T_matrix_8_new = cell(len, 1);
    for i = 1:len
        T_matrix_8_new{i} = T_matrix_8{i} ./ T_matrix_8_value;
    end

    len = length(T_matrix_9);
    T_matrix_9_new = cell(len, 1);
    for i = 1:len
        T_matrix_9_new{i} = T_matrix_9{i} ./ T_matrix_9_value;
    end

    len = length(T_matrix_10);
    T_matrix_10_new = cell(len, 1);
    for i = 1:len
        T_matrix_10_new{i} = T_matrix_10{i} ./ T_matrix_10_value;
    end
end