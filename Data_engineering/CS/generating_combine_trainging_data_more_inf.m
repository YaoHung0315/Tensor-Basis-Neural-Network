clc; clear all; close all;

dir_pwd = pwd;
pathParts = strsplit(dir_pwd, filesep);
if length(pathParts) > 2
    newPathParts = pathParts(1:end-2);
else
    newPathParts = pathParts;
end
newPathPart = strjoin(newPathParts, filesep);

dir_root_l =   [newPathPart '/Data_ML_TBNN/CS/'];
dir_root_l =   [dir_root_l 'R2023_point/Data_ori_SR_adj_feature_max_abs_single/'];
dir_root_s =   [newPathPart '/ML_TBNN/new/CS/CSShf/'];
dir_root_s =   [dir_root_s 'data'];



%% copy data
n = 1;

SaveCaseName{n} = ['CSRe710H089I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H121I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H131I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H077I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H105I8'];  n=n+1;
SaveCaseName{n} = ['CSRe307H105I8'];  n=n+1;
SaveCaseName{n} = ['CSRe500H105I8'];  n=n+1;
SaveCaseName{n} = ['CSRe1004H105I8'];  n=n+1;

if ~exist(dir_root_s, 'dir')
    mkdir(dir_root_s);
end

for i = 1:length(SaveCaseName)
    srcFile = fullfile(dir_root_l, [SaveCaseName{i}, '.mat']);
    destFile = fullfile(dir_root_s, [SaveCaseName{i}, '.mat']); 
    copyfile(srcFile, destFile);
end




%% comebine training data
clear SaveCaseName
n = 1;

SaveCaseName{n} = ['CSRe710H089I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H121I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H131I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H077I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe710H105I8'];  n=n+1;
SaveCaseName{n} = ['CSRe307H105I8'];  n=n+1;
SaveCaseName{n} = ['CSRe500H105I8'];  n=n+1;
SaveCaseName{n} = ['CSRe1004H105I8'];  n=n+1;

anisotropic_tensor_data_com = [];
T_matrix_data_com = [];
lambda_data_com = [];
R_tensor_com = {};
R_tensor_adj_com = {};
S_tensor_com = {};
S_tensor_adj_com = {};
x_point_com = {};
y_point_com = {};
Lx_com = {};
nu_com = {};
Re_tau_com = {};
x_com = {};
y_com = {};
S_tensor_max_com = {};
R_tensor_max_com = {};
Ux_com = {};
Uy_com = {};
Vx_com = {};
Vy_com = {};
uu_com = {};
uv_com = {};
vv_com = {};

for i = 1:length(SaveCaseName)
    load([dir_root_l, SaveCaseName{i}]);
    anisotropic_tensor_data_com = [anisotropic_tensor_data_com; anisotropic_tensor_data];
    T_matrix_data_com           = [T_matrix_data_com; T_matrix_data];
    lambda_data_com             = [lambda_data_com; lambda_data];
    R_tensor_com{i}             = R_tensor;
    R_tensor_adj_com{i}         = R_tensor_adj;
    S_tensor_com{i}             = S_tensor;
    S_tensor_adj_com{i}         = S_tensor_adj;
    x_point_com{i}              = x_point;
    y_point_com{i}              = y_point;
    Lx_com{i}                   = Lx;
    nu_com{i}                   = nu;
    Re_tau_com{i}               = Re_tau;
    x_com{i}                    = x;
    y_com{i}                    = y;
    S_tensor_max_com{i}         = S_tensor_max;
    R_tensor_max_com{i}         = R_tensor_max;
    Ux_com{i}                   = Ux;
    Uy_com{i}                   = Uy;
    Vx_com{i}                   = Vx;
    Vy_com{i}                   = Vy;
    uu_com{i}                   = uu;
    uv_com{i}                   = uv;
    vv_com{i}                   = vv;
end

anisotropic_tensor_data = anisotropic_tensor_data_com;
T_matrix_data = T_matrix_data_com;
lambda_data = lambda_data_com;
R_tensor = R_tensor_com;
R_tensor_adj = R_tensor_adj_com;
S_tensor = S_tensor_com;
S_tensor_adj = S_tensor_adj_com;
x_point = x_point_com;
y_point = y_point_com;
Lx = Lx_com;
nu = nu_com;
Re_tau = Re_tau_com;
x = x_com;
y = y_com;
S_tensor_max = S_tensor_max_com;
R_tensor_max = R_tensor_max_com;
Ux = Ux_com;
Uy = Uy_com;
Vx = Vx_com;
Vy = Vy_com;
uu = uu_com;
uv = uv_com;
vv = vv_com;


cd(dir_root_s)
save('ML_TBNN_training', 'anisotropic_tensor_data', 'T_matrix_data', 'lambda_data');
save('ML_TBNN_training_caseinf', 'R_tensor', 'R_tensor_adj', 'S_tensor', 'S_tensor_adj', 'S_tensor_max', 'R_tensor_max', ...
     'Ux','Uy','Vx','Vy','uu','vv','uv','x_point', 'y_point', 'Lx', 'nu', 'Re_tau', 'x', 'y');




%% comebine testing data
clear SaveCaseName
n = 1;

% SaveCaseName{n} = ['CSRe710H089I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe710H121I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe710H131I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe710H077I8'];  n=n+1;
SaveCaseName{n} = ['CSRe710H105I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe307H105I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe500H105I8'];  n=n+1;
% SaveCaseName{n} = ['CSRe1004H105I8'];  n=n+1;

anisotropic_tensor_data_com = [];
T_matrix_data_com = [];
lambda_data_com = [];
R_tensor_com = {};
R_tensor_adj_com = {};
S_tensor_com = {};
S_tensor_adj_com = {};
x_point_com = {};
y_point_com = {};
Lx_com = {};
nu_com = {};
Re_tau_com = {};
x_com = {};
y_com = {};
S_tensor_max_com = {};
R_tensor_max_com = {};
Ux_com = {};
Uy_com = {};
Vx_com = {};
Vy_com = {};
uu_com = {};
uv_com = {};
vv_com = {};

for i = 1:length(SaveCaseName)
    load([dir_root_l, SaveCaseName{i}]);
    anisotropic_tensor_data_com = [anisotropic_tensor_data_com; anisotropic_tensor_data];
    T_matrix_data_com           = [T_matrix_data_com; T_matrix_data];
    lambda_data_com             = [lambda_data_com; lambda_data];
    R_tensor_com{i}             = R_tensor;
    R_tensor_adj_com{i}         = R_tensor_adj;
    S_tensor_com{i}             = S_tensor;
    S_tensor_adj_com{i}         = S_tensor_adj;
    x_point_com{i}              = x_point;
    y_point_com{i}              = y_point;
    Lx_com{i}                   = Lx;
    nu_com{i}                   = nu;
    Re_tau_com{i}               = Re_tau;
    x_com{i}                    = x;
    y_com{i}                    = y;
    S_tensor_max_com{i}         = S_tensor_max;
    R_tensor_max_com{i}         = R_tensor_max;
    Ux_com{i}                   = Ux;
    Uy_com{i}                   = Uy;
    Vx_com{i}                   = Vx;
    Vy_com{i}                   = Vy;
    uu_com{i}                   = uu;
    uv_com{i}                   = uv;
    vv_com{i}                   = vv;
end

anisotropic_tensor_data = anisotropic_tensor_data_com;
T_matrix_data = T_matrix_data_com;
lambda_data = lambda_data_com;
R_tensor = R_tensor_com;
R_tensor_adj = R_tensor_adj_com;
S_tensor = S_tensor_com;
S_tensor_adj = S_tensor_adj_com;
x_point = x_point_com;
y_point = y_point_com;
Lx = Lx_com;
nu = nu_com;
Re_tau = Re_tau_com;
x = x_com;
y = y_com;
S_tensor_max = S_tensor_max_com;
R_tensor_max = R_tensor_max_com;
Ux = Ux_com;
Uy = Uy_com;
Vx = Vx_com;
Vy = Vy_com;
uu = uu_com;
uv = uv_com;
vv = vv_com;

cd(dir_root_s)
save('ML_TBNN_testing', 'anisotropic_tensor_data', 'T_matrix_data', 'lambda_data');
save('ML_TBNN_testing_caseinf', 'R_tensor', 'R_tensor_adj', 'S_tensor', 'S_tensor_adj', 'S_tensor_max', 'R_tensor_max', ...
     'Ux','Uy','Vx','Vy','uu','vv','uv','x_point', 'y_point', 'Lx', 'nu', 'Re_tau', 'x', 'y');

