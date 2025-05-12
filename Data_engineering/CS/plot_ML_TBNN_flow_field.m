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

dir_root_l_r = [newPathPart '/Data_original_flow_field/Data_rawdata/CS/'];
dir_root_l_d = [newPathPart '/Data_original_flow_field/Data_distribution_psi_phi/CS/'];
dir_root_c =   [newPathPart '/Code_ML_TBNN/CS/function/'];
dir_root_s =   [newPathPart '/Data_ML_TBNN/CS/'];
dir_root_s =   [dir_root_s 'Data_ori_SR_adj_feature_max_abs_single'];

simulation_dir
addpath(dir_root_c);



for i = 1:length(SaveCaseName)

    fullFileName = fullfile(dir_root_s, SaveCaseName{i});
    if ~exist(fullFileName, 'dir')
        mkdir(fullFileName);
    end

    cd(dir_root_s);
    load(SaveCaseName{i}); % load feature
    cd(fullFileName);


    lambda_1 = lambda_data(:,1);
    lambda_2 = lambda_data(:,2);
    lambda_3 = lambda_data(:,3);
    lambda_4 = lambda_data(:,4);
    lambda_5 = lambda_data(:,5);
    lambda_6 = lambda_data(:,6);
    lambda_7 = lambda_data(:,7);

    nn = length(T_matrix_data);
    T_matrix_1 = cell(nn, 1);
    for i = 1:nn
        vector_T = T_matrix_data{i};
        T_matrix_1{i} = reshape(vector_T(:,1), [3, 3]);
        T_matrix_2{i} = reshape(vector_T(:,2), [3, 3]);
        T_matrix_3{i} = reshape(vector_T(:,3), [3, 3]);
        T_matrix_4{i} = reshape(vector_T(:,4), [3, 3]);
        T_matrix_5{i} = reshape(vector_T(:,5), [3, 3]);
        T_matrix_6{i} = reshape(vector_T(:,6), [3, 3]);
        T_matrix_7{i} = reshape(vector_T(:,7), [3, 3]);
        T_matrix_8{i} = reshape(vector_T(:,8), [3, 3]);
        T_matrix_9{i} = reshape(vector_T(:,9), [3, 3]);
        T_matrix_10{i} = reshape(vector_T(:,10), [3, 3]);
        vector_a = anisotropic_tensor_data(i,:);
        anisotropic_tensor{i} = reshape(vector_a, [3, 3]);
    end


    eddy_viscosity = cell(nn,1);
    for i = 1:nn
        eddy_viscosity{i} = -(anisotropic_tensor{i}./S_tensor{i});
    end

    eddy_viscosity_double_dot = zeros(nn,1);
    for i = 1:nn
        matrix1 = anisotropic_tensor{i};
        matrix2 = S_tensor{i};
        eddy_viscosity_double_dot(i) = -(trace(matrix1 * matrix2')./trace(matrix2 * matrix2'));
    end




    %%%% tensor
    dataStruct = struct( ...
        'data', {T_matrix_1, T_matrix_2, T_matrix_3, T_matrix_4, T_matrix_5, T_matrix_6, T_matrix_7, T_matrix_8, T_matrix_9, T_matrix_10 ...
        anisotropic_tensor, R_tensor, S_tensor, R_tensor_adj, S_tensor_adj, eddy_viscosity}, ...
        'name', {'Tmatrix1', 'Tmatrix2', 'Tmatrix3', 'Tmatrix4', 'Tmatrix5', 'Tmatrix6', 'Tmatrix7', 'Tmatrix8', 'Tmatrix9', 'Tmatrix10'...
        'anisotropic tenosr', 'R tensor', 'S tnesor', 'R tensor adjust', 'S tnesor adjust', 'eddy viscosity'});



    % %%%% tensor
    % dataStruct = struct( ...
    %     'data', { R_tensor, S_tensor, R_tensor_adj, S_tensor_adj }, ...
    %     'name', { 'R tensor', 'S tnesor', 'R tensor adjust', 'S tnesor adjust' });


    for ii = 1:length(dataStruct)
        currentData = dataStruct(ii).data;
        currentName = dataStruct(ii).name;

        matrix11 = zeros(nn, 1);
        matrix12 = zeros(nn, 1);
        matrix22 = zeros(nn, 1);
        for i = 1:nn
            matrix11(i) = currentData{i}(1, 1);
            matrix12(i) = currentData{i}(1, 2);
            matrix22(i) = currentData{i}(2, 2);
        end

        matrix11 = griddata(x_point, y_point, matrix11, x, y, 'cubic');
        logi = isnan(matrix11);
        matrix11(logi) = griddata(x, y, matrix11, x(logi), y(logi), 'nearest');

        matrix12 = griddata(x_point, y_point, matrix12, x, y, 'cubic');
        logi = isnan(matrix12);
        matrix12(logi) = griddata(x, y, matrix12, x(logi), y(logi), 'nearest');

        matrix22 = griddata(x_point, y_point, matrix22, x, y, 'cubic');
        logi = isnan(matrix22);
        matrix22(logi) = griddata(x, y, matrix22, x(logi), y(logi), 'linear');


        t = tiledlayout("vertical");
        t.TileSpacing = 'compact';
        t.Padding = 'compact';


        nexttile;
        p = pcolor(x, y, matrix11);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;

        box on;
        axis equal;
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName, '\ 11'], 'Interpreter', 'latex');
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])


        nexttile;
        p = pcolor(x, y, matrix12);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;

        box on;
        axis equal;
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName, '\ 12'], 'Interpreter', 'latex');
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])


        nexttile;
        p = pcolor(x, y, matrix22);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;

        box on;
        axis equal;
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName, '\ 22'], 'Interpreter', 'latex');
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])

        xlabel(t,'$x$', 'Interpreter', 'latex');
        ylabel(t,'$y$', 'Interpreter', 'latex');

        fileName = [currentName, '.jpg'];
        print('-djpeg', '-r600', fileName);
        close(gcf);
    end





    %%%% tensor
    dataStruct = struct( ...
        'data', { anisotropic_tensor, eddy_viscosity }, ...
        'name', { 'anisotropic tenosr', 'eddy viscosity' });


    for ii = 1:length(dataStruct)
        currentData = dataStruct(ii).data;
        currentName = dataStruct(ii).name;

        matrix11 = zeros(nn, 1);
        matrix12 = zeros(nn, 1);
        matrix22 = zeros(nn, 1);
        for i = 1:nn
            matrix11(i) = currentData{i}(1, 1);
            matrix12(i) = currentData{i}(1, 2);
            matrix22(i) = currentData{i}(2, 2);
        end

        matrix11 = griddata(x_point, y_point, matrix11, x, y, 'cubic');
        logi = isnan(matrix11);
        matrix11(logi) = griddata(x, y, matrix11, x(logi), y(logi), 'nearest');

        matrix12 = griddata(x_point, y_point, matrix12, x, y, 'cubic');
        logi = isnan(matrix12);
        matrix12(logi) = griddata(x, y, matrix12, x(logi), y(logi), 'nearest');

        matrix22 = griddata(x_point, y_point, matrix22, x, y, 'cubic');
        logi = isnan(matrix22);
        matrix22(logi) = griddata(x, y, matrix22, x(logi), y(logi), 'linear');


        t = tiledlayout("vertical");
        t.TileSpacing = 'compact';
        t.Padding = 'compact';


        nexttile;
        p = pcolor(x, y, matrix11);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;
        clim([prctile(matrix11(:), 5), prctile(matrix11(:), 95)])


        box on;
        axis equal;
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName, '\ 11'], 'Interpreter', 'latex');
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])


        nexttile;
        p = pcolor(x, y, matrix12);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;
        clim([prctile(matrix12(:), 5), prctile(matrix12(:), 95)])

        box on;
        axis equal;
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName, '\ 12'], 'Interpreter', 'latex');
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])


        nexttile;
        p = pcolor(x, y, matrix22);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';
        colormap(jet);
        colorbar;
        clim([prctile(matrix22(:), 5), prctile(matrix22(:), 95)])

        box on;
        axis equal;
        set(gca, 'TickLabelInterpreter', 'latex');
        title([currentName, '\ 22'], 'Interpreter', 'latex');
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])

        xlabel(t,'$x$', 'Interpreter', 'latex');
        ylabel(t,'$y$', 'Interpreter', 'latex');

        fileName = [currentName, '.jpg'];
        print('-djpeg', '-r600', fileName);
        close(gcf);
    end




    %%%% scalar
    dataStruct = struct( ...
        'data', {lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, eddy_viscosity_double_dot}, ...
        'name', {'lambda1', 'lambda2', 'lambda3', 'lambda4', 'lambda5', 'lambda6', 'lambda7' ,'eddy viscosity double dot'});

    % dataStruct = struct( ...
    %     'data', {eddy_viscosity_double_dot}, ...
    %     'name', {'eddy viscosity double dot'});


    for ii = 1:length(dataStruct)
        currentData = dataStruct(ii).data;
        currentName = dataStruct(ii).name;
        currentData = griddata(x_point, y_point, currentData, x, y, 'cubic');

        figure;
        p = pcolor(x, y, currentData);
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
        xlim([min(x(:)) max(x(:))])
        ylim([min(y(:)) max(y(:))])
        clim([prctile(currentData(:), 5), prctile(currentData(:), 95)])

        fileName = [currentName, '.jpg'];
        print('-djpeg', '-r600', fileName);
        close(gcf);
    end


    cd ..
    cd ..
end
