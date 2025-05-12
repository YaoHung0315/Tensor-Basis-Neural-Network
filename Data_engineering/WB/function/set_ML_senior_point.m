function [ x_data_point, y_data_point ] = set_ML_senior_point( x_data, y_data );

Nx = 10;
Ny = 10;
vec_i1=  [1:Nx:size(x_data,1) size(x_data,1)];
vec_j1 = [1:Ny:size(x_data,2) size(x_data,2)];
vec_i2 = [1:(Nx-2):size(x_data,1) size(x_data,1)];
vec_j2 = [1:2:46];

tmpx1= x_data(vec_i1,vec_j1);
tmpx2= x_data(vec_i2,vec_j2);
x_data_point = [tmpx1(:) ; tmpx2(:)];

tmpy1= y_data(vec_i1,vec_j1);
tmpy2= y_data(vec_i2,vec_j2);
y_data_point = [tmpy1(:) ; tmpy2(:)];

end
