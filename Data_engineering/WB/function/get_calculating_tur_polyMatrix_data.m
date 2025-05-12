function [ T_matrix_1, T_matrix_2, T_matrix_3, T_matrix_4, T_matrix_5, T_matrix_6, T_matrix_7, T_matrix_8, T_matrix_9, T_matrix_10 ] = ...
get_calculating_tur_polyMatrix_data(  S_tensor, R_tensor )

len = length(S_tensor);
T_matrix_1  =  cell(len, 1);
T_matrix_2  =  cell(len, 1);
T_matrix_3  =  cell(len, 1);
T_matrix_4  =  cell(len, 1);
T_matrix_5  =  cell(len, 1);
T_matrix_6  =  cell(len, 1);
T_matrix_7  =  cell(len, 1);
T_matrix_8  =  cell(len, 1);
T_matrix_9  =  cell(len, 1);
T_matrix_10 =  cell(len, 1);

for i = 1:len

        S = S_tensor{i};
        R = R_tensor{i};
        [ T_matrix ] = get_polymatrix_2D( S, R );
        T_matrix_1{i}  = T_matrix{1};
        T_matrix_2{i}  = T_matrix{2};
        T_matrix_3{i}  = T_matrix{3};
        T_matrix_4{i}  = T_matrix{4};
        T_matrix_5{i}  = T_matrix{5};
        T_matrix_6{i}  = T_matrix{6};
        T_matrix_7{i}  = T_matrix{7};
        T_matrix_8{i}  = T_matrix{8};
        T_matrix_9{i}  = T_matrix{9};
        T_matrix_10{i} = T_matrix{10};
        
        clear S R T_matrix;

end

end



