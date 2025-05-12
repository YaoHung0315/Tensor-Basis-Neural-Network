function [ x_data_point, y_data_point ] = set_ML_senior_point( x_data, y_data, step_h);

% The index of the mask -----
% mask2c --- 'c' for coarse
% vec_i1 = 1:16:size(U,1)-1;
% vec_j1 = 1:64:size(U,2)-1;
% vec_i2 = 1:8:98;  % y-direction
% vec_j2 = 1800:32:2400;  % x-direction

% vec_i1 = [1:16:size(U,1)-1 size(U,1)];
% vec_j1 = [1:16:size(U,2)-1 size(U,2)];
% vec_i2 = 1:8:81;  % y-direction
% vec_j2 = 817:8:2300;  % x-direction

% % vec_i1 = 1:16:size(U,1)-1; %16
% % vec_j1 = 1:16:size(U,2)-1;
% vec_i1 = [1:16:size(U,1) size(U,1)]; %16
% vec_j1 = [1:16:size(U,2)];
% % vec_i2 = 20:8:101;  % y-direction
% % vec_j2 = 2000:8:2400;  % x-direction
% vec_i2 = 20:50:101;  % y-direction
% vec_j2 = 2000:500:2400;  % x-direction

% logi = slice_xp_c(:)<28 & slice_xp_c(:)>12 & slice_yp_c(:)>0 & slice_yp_c(:)<step_h;
% slice_xp_c(logi)           = [];
% slice_yp_c(logi)           = [];
% lambda_data_c(logi,:)        = [];
% T_matrix_data_c(logi,:)      = [];
% anisotropic_tensor_c(logi,:) = [];
% disp(['Total data points are Np = ' num2str(size(lambda_data_c,1)) '.'])


vec_i1 = [1:16:size(x_data,1) size(x_data,1)];
vec_j1 = [1:16:size(x_data,2)-1 size(x_data,2)];
vec_i2 = 1:8:81;  % y-direction
vec_j2 = 817:8:2300;  % x-direction


tmpx1= x_data(vec_i1,vec_j1);
tmpx2= x_data(vec_i2,vec_j2);
x_data_point = [tmpx1(:) ; tmpx2(:)];

tmpy1= y_data(vec_i1,vec_j1);
tmpy2= y_data(vec_i2,vec_j2);
y_data_point = [tmpy1(:) ; tmpy2(:)];


logi = x_data_point(:)<28 & x_data_point(:)>12 & y_data_point(:)>0 & y_data_point(:)<step_h;
x_data_point(logi)           = [];
y_data_point(logi)           = [];


end
