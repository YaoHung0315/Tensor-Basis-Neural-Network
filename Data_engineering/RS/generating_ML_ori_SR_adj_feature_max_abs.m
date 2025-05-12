%%
clc; clear all; close all;

dir_pwd = pwd;
pathParts = strsplit(dir_pwd, filesep);
if length(pathParts) > 2
    newPathParts = pathParts(1:end-2);
else
    newPathParts = pathParts;
end
newPathPart = strjoin(newPathParts, filesep);

dir_root_l_r = [newPathPart '/Data_original_flow_field/Data_rawdata/RS/'];
dir_root_l_d = [newPathPart '/Data_original_flow_field/Data_distribution_psi_phi/RS/'];
dir_root_c =   [newPathPart '/Code_ML_TBNN/RS/function/'];
dir_root_s =   [newPathPart '/Data_ML_TBNN/RS/'];
dir_root_s =   [dir_root_s 'R2023_point_1/Data_ori_SR_adj_feature_max_abs'];


simulation_dir
addpath(dir_root_c);


%% Transform Raw Data to ori_SR ori_feature
for i = 1:length(SaveCaseName)
    if ~exist(dir_root_s, 'dir')
        mkdir(dir_root_s);
    end

    raw_data_file_name = Casename_r{i};
    psi_data_file_name = Casename_psi{i};
    phi_data_file_name = Casename_phi{i};

    [lambda_data, T_matrix_data, anisotropic_tensor_data, S_tensor, R_tensor, S_tensor_adj, R_tensor_adj, S_tensor_max, R_tensor_max, ...
     Ux, Uy, Vx, Vy, uu, vv, uv, x_point, y_point, x, y, Re_tau, Lx, nu] = ...
     get_flow_field_to_ML_TBNN(dir_root_l_r, dir_root_l_d, raw_data_file_name, psi_data_file_name, phi_data_file_name);

    cd(dir_root_s);
    save([SaveCaseName{i}],'lambda_data','T_matrix_data','anisotropic_tensor_data','S_tensor','R_tensor','S_tensor_adj','R_tensor_adj','S_tensor_max','R_tensor_max', ...
                           'Ux','Uy','Vx','Vy','uu','vv','uv','x_point','y_point','x','y','Re_tau','Lx','nu','-v7.3');
    
    figure;
    scatter(x_point, y_point, 0.5, 'filled', 'k');
    box on;
    axis equal;
    set(gca, 'TickLabelInterpreter', 'latex');
    title(SaveCaseName{i}, 'Interpreter', 'latex');
    xlim([min(x_point(:)) max(x_point(:))]);
    ylim([min(y_point(:)) max(y_point(:))]);
    xlabel('$x$', 'Interpreter', 'latex');
    ylabel('$y$', 'Interpreter', 'latex');

    outputFileName = [SaveCaseName{i}, '_select_point.jpg'];
    print('-djpeg', '-r600', outputFileName);
    close all;

end




function [lambda_data, T_matrix_data, anisotropic_tensor_data, S_tensor, R_tensor, S_tensor_adj, R_tensor_adj, S_tensor_max, R_tensor_max, ...
          Ux, Uy, Vx, Vy, uu, vv, uv, x_point, y_point, x, y, Re_tau, Lx, nu] = ...
          get_flow_field_to_ML_TBNN(dir_root_l_r, dir_root_l_d, Casename_r, Casename_psi, Casename_phi)

tic

X  = []; Y  = []; Z  = [];
U  = []; V  = []; W  = [];
uu = []; uv = []; uw = [];
vu = []; vv = []; vw = [];
wu = []; wv = []; ww = [];

cd(dir_root_l_r);
load(Casename_r);

Uz = zeros(size(Ux)); Vz = zeros(size(Ux)); 
Wx = zeros(size(Ux)); Wy = zeros(size(Ux)); Wz = zeros(size(Ux)); 
uw = zeros(size(uu)); vw = zeros(size(uu)); 
wu = zeros(size(uu)); wv = zeros(size(uu)); ww = zeros(size(uu));
Z  = []; W  = [];
vu = uv;

Ux = Ux(2:end-1,2:end-1);
Uy = Uy(2:end-1,2:end-1);
Uz = Uz(2:end-1,2:end-1);

Vx = Vx(2:end-1,2:end-1);
Vy = Vy(2:end-1,2:end-1);
Vz = Vz(2:end-1,2:end-1);

Wx = Wx(2:end-1,2:end-1);
Wy = Wy(2:end-1,2:end-1);
Wz = Wz(2:end-1,2:end-1);


nu    = 1e-6;         % viscosity
Lx    = 0.01;         % characteristic length
Utau  = Re_tau*nu/Lx; % characteristic velocity
scl_L = 1/Lx;         % dimensionless velocity length
scl_U = 1/Utau;       % dimensionless velocity scale
x = X(2:end-1,2:end-1)*scl_L;
y = Y(2:end-1,2:end-1)*scl_L;

nx = size(x,1);
ny = size(x,2);
Re_tau = Re_tau;
Lx = Lx;
nu = nu;

step_h = 1; % change by case
target_points = 4000;
[x_point, y_point] = set_ML_triangle_point_R2023_1( x, y, target_points, step_h);
% [x_point, y_point] = set_ML_senior_point( x, y, step_h);


%%%%% calculate strain_rate_tensor, rotation_rate_tensor, anisotropic_tensor
[X, Y, Z, U, V, W] = get_physics_2frictoinscale(X, Y, Z, U, V, W, scl_U, scl_L);


% [Ux, Vx, Wx, Uy, Vy, Wy, Uz, Vz, Wz] = get_velocity_gradient_2D(X, Y, Z, U, V, W);
[Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz] = physics_gradU_2dimensionless(Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz, Lx, Utau);


[uu, vv, ww, uv, uw, vw] = get_physicsf_2frictoinscale(uu, vv, ww, uv, uw, vw, scl_U);
[uu, vv, ww, uv, uw, vw] = get_physics_cutboundary(uu, vv, ww, uv, uw, vw);

[ Ux ] = select_special_point_matrix( Ux, x, y, x_point, y_point);
[ Vx ] = select_special_point_matrix( Vx, x, y, x_point, y_point);
[ Wx ] = select_special_point_matrix( Wx, x, y, x_point, y_point);
[ Uy ] = select_special_point_matrix( Uy, x, y, x_point, y_point);
[ Vy ] = select_special_point_matrix( Vy, x, y, x_point, y_point);
[ Wy ] = select_special_point_matrix( Wy, x, y, x_point, y_point);
[ Uz ] = select_special_point_matrix( Uz, x, y, x_point, y_point);
[ Vz ] = select_special_point_matrix( Vz, x, y, x_point, y_point);
[ Wz ] = select_special_point_matrix( Wz, x, y, x_point, y_point);

[ uu ] = select_special_point_matrix( uu, x, y, x_point, y_point);
[ vv ] = select_special_point_matrix( vv, x, y, x_point, y_point);
[ ww ] = select_special_point_matrix( ww, x, y, x_point, y_point);
[ uv ] = select_special_point_matrix( uv, x, y, x_point, y_point);
[ uw ] = select_special_point_matrix( uw, x, y, x_point, y_point);
[ vw ] = select_special_point_matrix( vw, x, y, x_point, y_point);

[S_tensor, R_tensor] = get_calculating_tur_SR_data(Ux, Vx, Wx, Uy, Vy, Wy, Uz, Vz, Wz);
[S_tensor_max, S_tensor_min, R_tensor_max, R_tensor_min] = get_ori_SR_adj_feature_max_abs_value( S_tensor, R_tensor );
[S_tensor_adj, R_tensor_adj] = get_ori_SR_adj_feature_max_abs( S_tensor, R_tensor, S_tensor_max, S_tensor_min, R_tensor_max, R_tensor_min );

[Reynolds_stress] = get_Reynolds_stress( uu, uv, uw, vu, vv, vw, wu, wv, ww ); % calculate Reynolds_stress
[anisotropic_tensor] = get_anisotropic_tensor( Reynolds_stress ); % calculate anisotropic Reynolds_stress


%%%% calculate psi phi
cd(dir_root_l_d);
load(Casename_psi);
slice_psi = griddata(tri_position(:,1), tri_position(:,2), tri_psi, x_point, y_point, 'cubic');
logi = isnan(slice_psi);
slice_psi(logi) = griddata(tri_position(:,1), tri_position(:,2), tri_psi, x_point(logi), y_point(logi), 'nearest');

load(Casename_phi);
slice_phi = griddata(tri_position(:,1), tri_position(:,2), tri_psi, x_point, y_point, 'cubic');
logi = isnan(slice_phi);
slice_phi(logi) = griddata(tri_position(:,1), tri_position(:,2), tri_psi, x_point(logi), y_point(logi), 'nearest');


% calculate lambda_data
[lambda_1, lambda_2, lambda_3, lambda_4, lambda_5] = get_calculating_tur_invarients_data( S_tensor_adj, R_tensor_adj);
lambda_6 = slice_psi;
lambda_7 = slice_phi;
lambda_data = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7];

% calculate T_matrix_data
[T_matrix_1, T_matrix_2, T_matrix_3, T_matrix_4, T_matrix_5, T_matrix_6, T_matrix_7, T_matrix_8, T_matrix_9, T_matrix_10] = get_calculating_tur_polyMatrix_data( S_tensor_adj, R_tensor_adj );
T_matrix_data = cell(length(T_matrix_1), 1);
for i = 1:length(T_matrix_1)
    combinedMatrix = zeros(9, 10);
    combinedMatrix(:, 1) = T_matrix_1{i}(:);
    combinedMatrix(:, 2) = T_matrix_2{i}(:);
    combinedMatrix(:, 3) = T_matrix_3{i}(:);
    combinedMatrix(:, 4) = T_matrix_4{i}(:);
    combinedMatrix(:, 5) = T_matrix_5{i}(:);
    combinedMatrix(:, 6) = T_matrix_6{i}(:);
    combinedMatrix(:, 7) = T_matrix_7{i}(:);
    combinedMatrix(:, 8) = T_matrix_8{i}(:);
    combinedMatrix(:, 9) = T_matrix_9{i}(:);
    combinedMatrix(:, 10) = T_matrix_10{i}(:);
    combinedMatrix([3, 6, 7, 8, 9], :) = 0; %%%% not training data
    T_matrix_data{i} = combinedMatrix;
end

% calculate anisotropic_tensor_data
anisotropic_tensor_data = zeros(length(anisotropic_tensor), 9);
for i = 1:length(anisotropic_tensor)
    entry = anisotropic_tensor{i};
    anisotropic_tensor_data(i, :) = entry(:)';
end
anisotropic_tensor_data(:, [3, 6, 7, 8, 9]) = 0; %%%% not training data

disp(['finish calculated, t = ', num2str(toc)])

end



function [Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz] = physics_gradU_2dimensionless(Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz, scle_L, scle_U)
Ratio = scle_L/scle_U;
Ux = Ux.*Ratio;
Uy = Uy.*Ratio;
Uz = Uz.*Ratio;
Vx = Vx.*Ratio;
Vy = Vy.*Ratio;
Vz = Vz.*Ratio;
Wx = Wx.*Ratio;
Wy = Wy.*Ratio;
Wz = Wz.*Ratio;
end








