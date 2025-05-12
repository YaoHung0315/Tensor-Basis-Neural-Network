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
dir_root_s =   [dir_root_s 'Data_ori_SR_adj_feature_max_abs_single'];

simulation_dir
addpath(dir_root_c);


%%
for i = 1:length(SaveCaseName)

    fullFileName = fullfile(dir_root_s, SaveCaseName{i});
    if ~exist(fullFileName, 'dir')
        mkdir(fullFileName);
    end

    cd(dir_root_s);
    load(SaveCaseName{i}); % load feature
    load(Casename_r{i}); % load rawdata
    cd(fullFileName);


    %%%% parameters setting
    % nu    = 1e-6;         % viscosity
    % Lx    = 0.0508;       % characteristic length
    Utau  = Re_tau*nu/Lx % characteristic velocity
    scl_L = 1/Lx;         % dimensionless velocity length
    scl_U = 1/Utau;       % dimensionless velocity scale

    %%%%% dimensionless flow field value
    X = X*scl_L;
    Y = Y*scl_L;
    U = U*scl_U;
    V = V*scl_U;
    uu = uu*scl_U^2;
    vv = vv*scl_U^2;
    uv = uv*scl_U^2;

    % load(Casename_psi{i});
    % slice_psi = griddata(tri_position(:,1), tri_position(:,2), tri_psi, X, Y, 'cubic'); % load streamfunction
    % logi = isnan(slice_psi);
    % slice_psi(logi) = griddata(tri_position(:,1), tri_position(:,2), tri_psi, X(logi), Y(logi), 'nearest');
    % psi = slice_psi;
    % 
    % load(Casename_phi{i});
    % slice_phi = griddata(tri_position(:,1), tri_position(:,2), tri_psi, X, Y, 'cubic'); % load velocity potential
    % logi = isnan(slice_phi);
    % slice_phi(logi) = griddata(tri_position(:,1), tri_position(:,2), tri_psi, X(logi), Y(logi), 'nearest');
    % phi = slice_phi;


    dataStruct = struct('data', {U, V, uu, vv, uv}, 'name', {'U', 'V', 'uu', 'vv', 'uv'});

    for ii = 1:length(dataStruct)
        currentData = dataStruct(ii).data;
        currentName = dataStruct(ii).name;

        figure;
        p = pcolor(X, Y, currentData);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;

        box on;
        axis equal;
        xlabel('$x$', 'Interpreter', 'latex');
        ylabel('$y$', 'Interpreter', 'latex');
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName], 'Interpreter', 'latex');
        xlim([min(X(:)) max(X(:))])
        ylim([min(Y(:)) max(Y(:))])  
        % caxis([0 14])

        fileName = [currentName, '.jpg'];
        print('-djpeg', '-r600', fileName);        
        close(gcf);
    end
    cd ..
    cd ..

end
