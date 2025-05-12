function [T_matrix_1_value, T_matrix_2_value, T_matrix_3_value, T_matrix_4_value, T_matrix_5_value, T_matrix_6_value, T_matrix_7_value, T_matrix_8_value, T_matrix_9_value, T_matrix_10_value] = ...
    get_Tmatrix_max_abs_value( T_matrix_1, T_matrix_2, T_matrix_3, T_matrix_4, T_matrix_5, T_matrix_6, T_matrix_7, T_matrix_8, T_matrix_9, T_matrix_10 )

    len = length(T_matrix_1);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_1{i}(1,2) );
    end
    T_matrix_1_value = max(temp_values);
    
    len = length(T_matrix_2);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_2{i}(1,2) );
    end
    T_matrix_2_value = max(temp_values);
    
    len = length(T_matrix_3);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_3{i}(1,2) );
    end
    T_matrix_3_value = max(temp_values);
    
    len = length(T_matrix_4);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_4{i}(2,2) );
    end
    T_matrix_4_value = max(temp_values);

    len = length(T_matrix_5);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_5{i}(1,2) );
    end
    T_matrix_5_value = max(temp_values);

    len = length(T_matrix_6);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_6{i}(1,2) );
    end
    T_matrix_6_value = max(temp_values);

    len = length(T_matrix_7);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_7{i}(1,2) );
    end
    T_matrix_7_value = max(temp_values);
    
    len = length(T_matrix_8);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_8{i}(1,2) );
    end
    T_matrix_8_value = max(temp_values);

    len = length(T_matrix_9);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_9{i}(1,2) );
    end
    T_matrix_9_value = max(temp_values);

    len = length(T_matrix_10);
    temp_values = zeros(len,1);
    for i = 1:len
        temp_values(i) = abs( T_matrix_10{i}(1,2) );
    end
    T_matrix_10_value = max(temp_values);
end