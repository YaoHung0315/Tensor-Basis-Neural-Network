function [ x_data_point, y_data_point ] = set_ML_senior_point( x_data, y_data );


vec_i1 = 2:8:size(x_data,1)-1;
vec_i1 = [vec_i1 size(x_data,1)-2];
vec_j1 = 2:8:size(x_data,2)-1;
vec_j1 = [vec_j1 size(x_data,2)-2];

vec_i2 = 512:4:size(x_data,1)-1;  % x-direction
vec_j2 = 2:2:45;   % y-direction


tmpx1= x_data(vec_i1,vec_j1);
tmpx2= x_data(vec_i2,vec_j2);
x_data_point = [tmpx1(:) ; tmpx2(:)];

tmpy1= y_data(vec_i1,vec_j1);
tmpy2= y_data(vec_i2,vec_j2);
y_data_point = [tmpy1(:) ; tmpy2(:)];

end
