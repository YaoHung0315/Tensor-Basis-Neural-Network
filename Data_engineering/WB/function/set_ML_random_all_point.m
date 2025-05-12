function [x_data_point, y_data_point] = set_ML_random_all_point(x_data, y_data)

x_data = x_data(:);
y_data = y_data(:);

rng(1); % random seed
nn = length(x_data);
numToSelect = ceil(nn/10);
Indices = randperm(nn, numToSelect);
x_select_data = x_data(Indices);
y_select_data = y_data(Indices);
x_data_point = x_select_data(:);
y_data_point = y_select_data(:);


end