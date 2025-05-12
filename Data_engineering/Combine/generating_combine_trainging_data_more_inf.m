clc; clear all; close all;

dir_pwd = pwd;
pathParts = strsplit(dir_pwd, filesep);
if length(pathParts) > 2
    newPathParts = pathParts(1:end-2);
else
    newPathParts = pathParts;
end
newPathPart = strjoin(newPathParts, filesep);

dir_root_l_CS =   [newPathPart '/Data_ML_TBNN/CS/'];
dir_root_l_CS =   [dir_root_l_CS 'R2023_point/Data_ori_SR_adj_feature_max_abs_single/'];

dir_root_l_RS =   [newPathPart '/Data_ML_TBNN/RS/'];
dir_root_l_RS =   [dir_root_l_RS 'R2023_point/Data_ori_SR_adj_feature_max_abs_single/'];

dir_root_l_WB =   [newPathPart '/Data_ML_TBNN/WB/'];
dir_root_l_WB =   [dir_root_l_WB 'R2023_point/Data_ori_SR_adj_feature_max_abs_single/'];

dir_root_s_Combin =   [newPathPart '/ML_TBNN/Combine/R2023_point/Data_ori_SR_adj_feature_max_abs_single/CombineCSH/'];
dir_root_s_Combin =   [dir_root_s_Combin 'data'];




%% copy data
if ~exist(dir_root_s_Combin, 'dir')
    mkdir(dir_root_s_Combin);
end


n = 1;
SaveCaseName_CS{n} = ['CSRe710H089I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H121I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H131I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H077I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe307H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe500H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe1004H105I8'];  n=n+1;

for i = 1:length(SaveCaseName_CS)
    srcFile = fullfile(dir_root_l_CS, [SaveCaseName_CS{i}, '.mat']);
    destFile = fullfile(dir_root_s_Combin, [SaveCaseName_CS{i}, '.mat']); 
    copyfile(srcFile, destFile);
end


n = 1;
SaveCaseName_RS{n} = ['RSRe1162H333I7'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1273H333I7'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1325H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1375H333I7'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1325H167I5'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1325H250I5'];  n=n+1;

for i = 1:length(SaveCaseName_RS)
    srcFile = fullfile(dir_root_l_RS, [SaveCaseName_RS{i}, '.mat']);
    destFile = fullfile(dir_root_s_Combin, [SaveCaseName_RS{i}, '.mat']); 
    copyfile(srcFile, destFile);
end


% n = 1;
% SaveCaseName_WB{n} = ['WBRe307H059I7'];  n=n+1; % WBRe307H118
% SaveCaseName_WB{n} = ['WBRe198H059I7'];  n=n+1; % WBRe198H118
% SaveCaseName_WB{n} = ['WBRe251H030I7'];  n=n+1; % WBRe251H089
% SaveCaseName_WB{n} = ['WBRe251H089I7'];  n=n+1; % WBRe251H148
% SaveCaseName_WB{n} = ['WBRe251H059I7'];  n=n+1; % WBRe251H118
% SaveCaseName_WB{n} = ['WBRe710H059I7'];  n=n+1; % WBRe710H118
% SaveCaseName_WB{n} = ['WBRe500H059I7'];  n=n+1; % WBRe500H118
% SaveCaseName_WB{n} = ['WBRe1004H059I7']; n=n+1; % WBRe1004H118
% SaveCaseName_WB{n} = ['WBRe710H030I7'];  n=n+1; % WBRe710H148
% SaveCaseName_WB{n} = ['WBRe710H089I7'];  n=n+1; % WBRe710H089
% SaveCaseName_WB{n} = ['WBRe251H177I7'];  n=n+1; % WBRe251H177
% 
% for i = 1:length(SaveCaseName_WB)
%     srcFile = fullfile(dir_root_l_WB, [SaveCaseName_WB{i}, '.mat']);
%     destFile = fullfile(dir_root_s_Combin, [SaveCaseName_WB{i}, '.mat']); 
%     copyfile(srcFile, destFile);
% end






%% comebine training data
clear SaveCaseName_CS
n = 1;
SaveCaseName_CS{n} = ['CSRe710H089I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H121I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H131I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H077I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe710H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe307H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe500H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe1004H105I8'];  n=n+1;


clear SaveCaseName_RS
n = 1;
SaveCaseName_RS{n} = ['RSRe1162H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1273H333I7'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1325H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1375H333I7'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1325H167I5'];  n=n+1;
SaveCaseName_RS{n} = ['RSRe1325H250I5'];  n=n+1;


% clear SaveCaseName_WB
% n = 1;
% SaveCaseName_WB{n} = ['WBRe307H059I7'];  n=n+1; % WBRe307H118
% SaveCaseName_WB{n} = ['WBRe198H059I7'];  n=n+1; % WBRe198H118
% SaveCaseName_WB{n} = ['WBRe251H030I7'];  n=n+1; % WBRe251H089
% SaveCaseName_WB{n} = ['WBRe251H089I7'];  n=n+1; % WBRe251H148
% SaveCaseName_WB{n} = ['WBRe251H059I7'];  n=n+1; % WBRe251H118
% SaveCaseName_WB{n} = ['WBRe710H059I7'];  n=n+1; % WBRe710H118
% SaveCaseName_WB{n} = ['WBRe500H059I7'];  n=n+1; % WBRe500H118
% SaveCaseName_WB{n} = ['WBRe1004H059I7']; n=n+1; % WBRe1004H118
% SaveCaseName_WB{n} = ['WBRe710H030I7'];  n=n+1; % WBRe710H148
% SaveCaseName_WB{n} = ['WBRe710H089I7'];  n=n+1; % WBRe710H089
% SaveCaseName_WB{n} = ['WBRe251H177I7'];  n=n+1; % WBRe251H177


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


n_CS = length(SaveCaseName_CS);
n_RS = length(SaveCaseName_RS);
% n_WB = length(SaveCaseName_WB);
for i = 1:n_CS
    load([dir_root_l_CS, SaveCaseName_CS{i}]);
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


for i = 1:n_RS
    load([dir_root_l_RS, SaveCaseName_RS{i}]);
    anisotropic_tensor_data_com = [anisotropic_tensor_data_com; anisotropic_tensor_data];
    T_matrix_data_com           = [T_matrix_data_com; T_matrix_data];
    lambda_data_com             = [lambda_data_com; lambda_data];
    R_tensor_com{i+n_CS}             = R_tensor;
    R_tensor_adj_com{i+n_CS}         = R_tensor_adj;
    S_tensor_com{i+n_CS}             = S_tensor;
    S_tensor_adj_com{i+n_CS}         = S_tensor_adj;
    x_point_com{i+n_CS}              = x_point;
    y_point_com{i+n_CS}              = y_point;
    Lx_com{i+n_CS}                   = Lx;
    nu_com{i+n_CS}                   = nu;
    Re_tau_com{i+n_CS}               = Re_tau;
    x_com{i+n_CS}                    = x;
    y_com{i+n_CS}                    = y;
    S_tensor_max_com{i+n_CS}         = S_tensor_max;
    R_tensor_max_com{i+n_CS}         = R_tensor_max;
    Ux_com{i+n_CS}                   = Ux;
    Uy_com{i+n_CS}                   = Uy;
    Vx_com{i+n_CS}                   = Vx;
    Vy_com{i+n_CS}                   = Vy;
    uu_com{i+n_CS}                   = uu;
    uv_com{i+n_CS}                   = uv;
    vv_com{i+n_CS}                   = vv;
end


% for i = 1:n_WB
%     load([dir_root_l_WB, SaveCaseName_WB{i}]);
%     anisotropic_tensor_data_com = [anisotropic_tensor_data_com; anisotropic_tensor_data];
%     T_matrix_data_com           = [T_matrix_data_com; T_matrix_data];
%     lambda_data_com             = [lambda_data_com; lambda_data];
%     R_tensor_com{i+n_CS+n_WB}             = R_tensor;
%     R_tensor_adj_com{i+n_CS+n_WB}         = R_tensor_adj;
%     S_tensor_com{i+n_CS+n_WB}             = S_tensor;
%     S_tensor_adj_com{i+n_CS+n_WB}         = S_tensor_adj;
%     x_point_com{i+n_CS+n_WB}              = x_point;
%     y_point_com{i+n_CS+n_WB}              = y_point;
%     Lx_com{i+n_CS+n_WB}                   = Lx;
%     nu_com{i+n_CS+n_WB}                   = nu;
%     Re_tau_com{i+n_CS+n_WB}               = Re_tau;
%     x_com{i+n_CS+n_WB}                    = x;
%     y_com{i+n_CS+n_WB}                    = y;
%     S_tensor_max_com{i+n_CS+n_WB}         = S_tensor_max;
%     R_tensor_max_com{i+n_CS+n_WB}         = R_tensor_max;
%     Ux_com{i+n_CS+n_WB}                   = Ux;
%     Uy_com{i+n_CS+n_WB}                   = Uy;
%     Vx_com{i+n_CS+n_WB}                   = Vx;
%     Vy_com{i+n_CS+n_WB}                   = Vy;
%     uu_com{i+n_CS+n_WB}                   = uu;
%     uv_com{i+n_CS+n_WB}                   = uv;
%     vv_com{i+n_CS+n_WB}                   = vv;
% end


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


cd(dir_root_s_Combin)
save('ML_TBNN_training', 'anisotropic_tensor_data', 'T_matrix_data', 'lambda_data');
save('ML_TBNN_training_caseinf', 'R_tensor', 'R_tensor_adj', 'S_tensor', 'S_tensor_adj', 'S_tensor_max', 'R_tensor_max', ...
     'Ux','Uy','Vx','Vy','uu','vv','uv','x_point', 'y_point', 'Lx', 'nu', 'Re_tau', 'x', 'y');

n_CS = 0;
n_RS = 0;




%% comebine testing data
clear SaveCaseName_CS
n = 1;
% SaveCaseName_CS{n} = ['CSRe710H089I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe710H121I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe710H131I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe710H077I8'];  n=n+1;
SaveCaseName_CS{n} = ['CSRe710H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe307H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe500H105I8'];  n=n+1;
% SaveCaseName_CS{n} = ['CSRe1004H105I8'];  n=n+1;


% clear SaveCaseName_RS
% n = 1;
% SaveCaseName_RS{n} = ['RSRe1162H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1273H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1325H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1375H333I7'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1325H167I5'];  n=n+1;
% SaveCaseName_RS{n} = ['RSRe1325H250I5'];  n=n+1;


% clear SaveCaseName_WB
% n = 1;
% SaveCaseName_WB{n} = ['WBRe307H059I7'];  n=n+1; % WBRe307H118
% SaveCaseName_WB{n} = ['WBRe198H059I7'];  n=n+1; % WBRe198H118
% SaveCaseName_WB{n} = ['WBRe251H030I7'];  n=n+1; % WBRe251H089
% SaveCaseName_WB{n} = ['WBRe251H089I7'];  n=n+1; % WBRe251H148
% SaveCaseName_WB{n} = ['WBRe251H059I7'];  n=n+1; % WBRe251H118
% SaveCaseName_WB{n} = ['WBRe710H059I7'];  n=n+1; % WBRe710H118
% SaveCaseName_WB{n} = ['WBRe500H059I7'];  n=n+1; % WBRe500H118
% SaveCaseName_WB{n} = ['WBRe1004H059I7']; n=n+1; % WBRe1004H118
% SaveCaseName_WB{n} = ['WBRe710H030I7'];  n=n+1; % WBRe710H148
% SaveCaseName_WB{n} = ['WBRe710H089I7'];  n=n+1; % WBRe710H089
% SaveCaseName_WB{n} = ['WBRe251H177I7'];  n=n+1; % WBRe251H177


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


n_CS = length(SaveCaseName_CS);
% n_RS = length(SaveCaseName_RS);
% n_WB = length(SaveCaseName_WB);
for i = 1:n_CS
    load([dir_root_l_CS, SaveCaseName_CS{i}]);
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


% for i = 1:n_RS
%     load([dir_root_l_RS, SaveCaseName_RS{i}]);
%     anisotropic_tensor_data_com = [anisotropic_tensor_data_com; anisotropic_tensor_data];
%     T_matrix_data_com           = [T_matrix_data_com; T_matrix_data];
%     lambda_data_com             = [lambda_data_com; lambda_data];
%     R_tensor_com{i+n_CS}             = R_tensor;
%     R_tensor_adj_com{i+n_CS}         = R_tensor_adj;
%     S_tensor_com{i+n_CS}             = S_tensor;
%     S_tensor_adj_com{i+n_CS}         = S_tensor_adj;
%     x_point_com{i+n_CS}              = x_point;
%     y_point_com{i+n_CS}              = y_point;
%     Lx_com{i+n_CS}                   = Lx;
%     nu_com{i+n_CS}                   = nu;
%     Re_tau_com{i+n_CS}               = Re_tau;
%     x_com{i+n_CS}                    = x;
%     y_com{i+n_CS}                    = y;
%     S_tensor_max_com{i+n_CS}         = S_tensor_max;
%     R_tensor_max_com{i+n_CS}         = R_tensor_max;
%     Ux_com{i+n_CS}                   = Ux;
%     Uy_com{i+n_CS}                   = Uy;
%     Vx_com{i+n_CS}                   = Vx;
%     Vy_com{i+n_CS}                   = Vy;
%     uu_com{i+n_CS}                   = uu;
%     uv_com{i+n_CS}                   = uv;
%     vv_com{i+n_CS}                   = vv;
% end


% for i = 1:n_WB
%     load([dir_root_l_WB, SaveCaseName_WB{i}]);
%     anisotropic_tensor_data_com = [anisotropic_tensor_data_com; anisotropic_tensor_data];
%     T_matrix_data_com           = [T_matrix_data_com; T_matrix_data];
%     lambda_data_com             = [lambda_data_com; lambda_data];
%     R_tensor_com{i+n_CS+n_WB}             = R_tensor;
%     R_tensor_adj_com{i+n_CS+n_WB}         = R_tensor_adj;
%     S_tensor_com{i+n_CS+n_WB}             = S_tensor;
%     S_tensor_adj_com{i+n_CS+n_WB}         = S_tensor_adj;
%     x_point_com{i+n_CS+n_WB}              = x_point;
%     y_point_com{i+n_CS+n_WB}              = y_point;
%     Lx_com{i+n_CS+n_WB}                   = Lx;
%     nu_com{i+n_CS+n_WB}                   = nu;
%     Re_tau_com{i+n_CS+n_WB}               = Re_tau;
%     x_com{i+n_CS+n_WB}                    = x;
%     y_com{i+n_CS+n_WB}                    = y;
%     S_tensor_max_com{i+n_CS+n_WB}         = S_tensor_max;
%     R_tensor_max_com{i+n_CS+n_WB}         = R_tensor_max;
%     Ux_com{i+n_CS+n_WB}                   = Ux;
%     Uy_com{i+n_CS+n_WB}                   = Uy;
%     Vx_com{i+n_CS+n_WB}                   = Vx;
%     Vy_com{i+n_CS+n_WB}                   = Vy;
%     uu_com{i+n_CS+n_WB}                   = uu;
%     uv_com{i+n_CS+n_WB}                   = uv;
%     vv_com{i+n_CS+n_WB}                   = vv;
% end


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


cd(dir_root_s_Combin)
save('ML_TBNN_testing', 'anisotropic_tensor_data', 'T_matrix_data', 'lambda_data');
save('ML_TBNN_testing_caseinf', 'R_tensor', 'R_tensor_adj', 'S_tensor', 'S_tensor_adj', 'S_tensor_max', 'R_tensor_max', ...
     'Ux','Uy','Vx','Vy','uu','vv','uv','x_point', 'y_point', 'Lx', 'nu', 'Re_tau', 'x', 'y');

