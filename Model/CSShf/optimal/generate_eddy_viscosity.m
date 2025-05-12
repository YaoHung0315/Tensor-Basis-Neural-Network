%%
clc; clear all; close all;

dir_pwd = pwd;
pathParts = strsplit(dir_pwd, filesep);
if length(pathParts) > 1
    newPathParts = pathParts(1:end-1);
else
    newPathParts = pathParts;
end
newPathPart = strjoin(newPathParts, filesep);

path = newPathPart;
dir_root_label = [path '/output/label'];
dir_root_pred =  [path '/output/pred'];
dir_root_loc = [path '/data/'];
dir_root_s =   [path '/fig'];

label_dir = [dir_root_label '/label_datasheet_testing.mat'];
pred_sum_dir = [dir_root_pred '/pred_sum_datasheet_testing.mat'];
Case_dir = [dir_root_loc '/ML_TBNN_testing_caseinf.mat'];

load(label_dir);
load(pred_sum_dir);
load(Case_dir);

x_point = x_point{1};
y_point = y_point{1};
x = x{1};
y = y{1};
S_tensor = S_tensor{1};
S_tensor_adj = S_tensor_adj{1};

R_tensor = R_tensor{1};
R_tensor_adj = R_tensor_adj{1};

Ux = Ux;
Uy = Uy;
Vx = Vx;
Vy = Vy;
uu = uu;
uv = uv;
vv = vv;


%% double dot, eddy viscosity individual

label_tensor = cell(size(label_data, 1), 1);
pred_sum_tensor = cell(size(pred_sum_data, 1), 1);
for i = 1:size(label_data, 1)
    label_tensor{i} = reshape(label_data(i, :), [3, 3]);
    pred_sum_tensor{i} = reshape(pred_sum_data(i, :), [3, 3]);

end
bij_LES = label_data;
bij_TBNN = pred_sum_data;

Sij = zeros(length(S_tensor), 9);
Sij_adj = zeros(length(S_tensor_adj), 9);
Rij = zeros(length(R_tensor), 9);
Rij_adj = zeros(length(R_tensor_adj), 9);
for i = 1:size(S_tensor, 1)
    Sij(i,:) = S_tensor{i}(:);
    Sij_adj(i,:) = S_tensor_adj{i}(:);
    Rij(i,:) = R_tensor{i}(:);
    Rij_adj(i,:) = R_tensor_adj{i}(:);
end


S_matrix = zeros(length(S_tensor), 1);
S_matrix_adj = zeros(length(S_tensor_adj), 1);
label_matrix = zeros(length(label_tensor), 1);
pred_sum_matrix = zeros(length(pred_sum_data), 1);
for i = 1:length(S_tensor)
    matrix = S_tensor{i};
    S_matrix(i) = sqrt(trace(matrix * matrix'));
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = label_tensor{i};
    label_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_sum_tensor{i};
    pred_sum_matrix(i) = trace(matrix1 * matrix2');
end

label_eddy_double_dot = -(label_matrix)./(S_matrix.*S_matrix);
pred_sum_eddy_double_dot = -(pred_sum_matrix)./(S_matrix.*S_matrix);

eddy_double_LES = label_eddy_double_dot;
eddy_double_TBNN = pred_sum_eddy_double_dot;


eddy_LES = size(length(S_tensor_adj), 1);
eddy_TBNN = size(length(S_tensor_adj), 1);
for i = 1:length(S_tensor)
    for j = 1:9
        eddy_LES(i,j) = bij_LES(i,j)/Sij(i);
        eddy_TBNN(i,j) = bij_TBNN(i,j)/Sij(i);
    end
end


cd(dir_root_loc)
save('ML_Optimal_method.mat', 'eddy_double_LES', 'eddy_double_TBNN', 'eddy_LES', 'eddy_TBNN','Sij', 'Sij_adj', 'Rij', 'Rij_adj', ...
     'Ux', 'Uy', 'Vx', 'Vy', 'uu', 'uv', 'vv', 'bij_LES', 'bij_TBNN', 'x_point', 'y_point', 'x', 'y');



