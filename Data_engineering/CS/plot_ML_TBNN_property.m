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


%%
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
    T_matrix_2 = cell(nn, 1);
    T_matrix_3 = cell(nn, 1);
    T_matrix_4 = cell(nn, 1);
    T_matrix_5 = cell(nn, 1);
    T_matrix_6 = cell(nn, 1);
    T_matrix_7 = cell(nn, 1);
    T_matrix_8 = cell(nn, 1);
    T_matrix_9 = cell(nn, 1);
    T_matrix_10 = cell(nn, 1);
    anisotropic_tensor = cell(nn, 1);
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




    lambdaStruct = struct( ...
        'data', {lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7}, ...
        'name', {'lambda1', 'lambda2', 'lambda3', 'lambda4', 'lambda5', 'lambda6', 'lambda7'});

    % lambda distribution(histogram)
    for ii = 1:length(lambdaStruct)
        currentLambda = lambdaStruct(ii).data;
        currentName = lambdaStruct(ii).name;

        figure;
        histogram(currentLambda(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        fileName = [currentName '_distribution.jpg'];
        print('-djpeg', '-r600', fileName);
        close(gcf);
    end

    % lambda boxplot
    figure;
    allLambdaData = [];
    for ii = 1:length(lambdaStruct)
        currentLambda = lambdaStruct(ii).data(:);
        allLambdaData = [allLambdaData, currentLambda];
    end
    boxplot(allLambdaData, 'Labels', {lambdaStruct.name});
    ylabel('value', 'Interpreter', 'latex');

    title('lambda boxplot', 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');
    set(gca, 'YScale', 'log')
    fileName = 'lambda boxplot.jpg';
    print('-djpeg', '-r600', fileName);
    close(gcf);

    % lambda Correlation Map
    figure;
    featureMatrix = zeros(nn,1);
    for i = 1:nn
        featureMatrix(i) = anisotropic_tensor{i}(1, 2);
    end

    data_matrix = [allLambdaData, featureMatrix(:)];

    correlation_matrix = corr(data_matrix);
    imagesc(abs(correlation_matrix));

    colorbar;
    colormap(jet);
    title('lambda correlation map', 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));
    xticklabels([arrayfun(@(x) ['label', num2str(x)], 1:size(allLambdaData, 2), 'UniformOutput', false), {'anisotropy tensor 12'}]);
    yticklabels([arrayfun(@(x) ['label', num2str(x)], 1:size(allLambdaData, 2), 'UniformOutput', false), {'anisotropy tensor 12'}]);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1])

    fileName = ['lambda_correlation_map.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);
    



    TmatrixStruct = struct( ...
        'data', {T_matrix_1, T_matrix_2, T_matrix_3, T_matrix_4, T_matrix_5, T_matrix_6, T_matrix_7, T_matrix_8, T_matrix_9, T_matrix_10}, ...
        'name', {'Tmatrix1', 'Tmatrix2', 'Tmatrix3', 'Tmatrix4', 'Tmatrix5', 'Tmatrix6', 'Tmatrix7', 'Tmatrix8', 'Tmatrix9', 'Tmatrix10'});

    % T matrix distribution(histogram)
    for ii = 1:length(TmatrixStruct)
        currentData = TmatrixStruct(ii).data;
        currentName = TmatrixStruct(ii).name;

        matrix11 = zeros(nn, 1);
        matrix12 = zeros(nn, 1);
        matrix22 = zeros(nn, 1);
        for i = 1:nn
            matrix11(i) = currentData{i}(1, 1);
            matrix12(i) = currentData{i}(1, 2);
            matrix22(i) = currentData{i}(2, 2);
        end

        figure;
        t = tiledlayout("vertical");
        t.TileSpacing = 'compact';
        t.Padding = 'compact';

        nexttile;
        histogram(matrix11(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' 11 distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        nexttile;
        histogram(matrix12(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' 12 distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        nexttile;
        histogram(matrix22(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' 22 distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        fileName = [currentName, '_distribution.jpg'];
        print('-djpeg', '-r600', fileName);
        close(gcf);
    end


    % T matrix boxplot
    figure;
    allTmatrixdata1 = [];
    allTmatrixdata2 = [];
    allTmatrixdata5 = [];
    for ii = 1:length(TmatrixStruct)
        currentTmatrix = TmatrixStruct(ii).data(:);
        currentTmatrix1 = zeros(length(currentTmatrix(:)),1);
        currentTmatrix2 = zeros(length(currentTmatrix(:)),1);
        currentTmatrix5 = zeros(length(currentTmatrix(:)),1);
        for i = 1:length(currentTmatrix)
            currentTmatrix1(i) = currentTmatrix{i}(1, 1);
            currentTmatrix2(i) = currentTmatrix{i}(1, 2);
            currentTmatrix5(i) = currentTmatrix{i}(2, 2);
        end
        allTmatrixdata1 = [allTmatrixdata1, currentTmatrix1];
        allTmatrixdata2 = [allTmatrixdata2, currentTmatrix2];
        allTmatrixdata5 = [allTmatrixdata5, currentTmatrix5];
    end


    boxplot(allTmatrixdata1, 'Labels', {TmatrixStruct.name});
    ylabel('value', 'Interpreter', 'latex');
    title('lambda 11 boxplot', 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');
    set(gca, 'YScale', 'log')

    fileName = 'Tmatrix_11_boxplot.jpg';
    print('-djpeg', '-r600', fileName);
    close(gcf);


    boxplot(allTmatrixdata2, 'Labels', {TmatrixStruct.name});
    ylabel('value', 'Interpreter', 'latex');
    title('lambda 12 boxplot', 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');
    set(gca, 'YScale', 'log')

    fileName = 'Tmatrix_12_boxplot.jpg';
    print('-djpeg', '-r600', fileName);
    close(gcf);


    boxplot(allTmatrixdata5, 'Labels', {TmatrixStruct.name});
    ylabel('value', 'Interpreter', 'latex');
    title('lambda 22 boxplot', 'Interpreter', 'latex');
    set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');
    set(gca, 'YScale', 'log')

    fileName = 'Tmatrix_22_boxplot.jpg';
    print('-djpeg', '-r600', fileName);
    close(gcf);


    % T matrix Correlation Map
    figure;
    featureMatrix1 = zeros(nn,1);
    featureMatrix2 = zeros(nn,1);
    featureMatrix5 = zeros(nn,1);
    for i = 1:nn
        featureMatrix1(i) = anisotropic_tensor{i}(1, 1);
        featureMatrix2(i) = anisotropic_tensor{i}(1, 2);
        featureMatrix5(i) = anisotropic_tensor{i}(2, 2);
    end

    data_matrix1 = [allTmatrixdata1, featureMatrix1(:)];
    data_matrix2 = [allTmatrixdata2, featureMatrix2(:)];
    data_matrix5 = [allTmatrixdata5, featureMatrix5(:)];


    correlation_matrix = corr(data_matrix1);
    imagesc(abs(correlation_matrix));

    colorbar;
    colormap(jet);
    title('Tmatrix 11 correlation map', 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));
    xticklabels([arrayfun(@(x) ['Tmatrix ', num2str(x)], 1:size(allTmatrixdata5, 2), 'UniformOutput', false), {'anisotropy tensor 11'}]);
    yticklabels([arrayfun(@(x) ['Tmatrix ', num2str(x)], 1:size(allTmatrixdata5, 2), 'UniformOutput', false), {'anisotropy tensor 11'}]);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1])
    colormap(flip(gray))

    fileName = ['Tmatrix_11_correlation_map.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);


    correlation_matrix = corr(data_matrix2);
    imagesc(abs(correlation_matrix));

    colorbar;
    colormap(jet);
    title('Tmatrix 12 correlation map', 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));
    xticklabels([arrayfun(@(x) ['Tmatrix ', num2str(x)], 1:size(allTmatrixdata5, 2), 'UniformOutput', false), {'anisotropy tensor 12'}]);
    yticklabels([arrayfun(@(x) ['Tmatrix ', num2str(x)], 1:size(allTmatrixdata5, 2), 'UniformOutput', false), {'anisotropy tensor 12'}]);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1])
    colormap(flip(gray))

    fileName = ['Tmatrix_12_correlation_map.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);


    correlation_matrix = corr(data_matrix5);
    imagesc(abs(correlation_matrix));

    colorbar;
    colormap(jet);
    title('Tmatrix 22 correlation map', 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));
    xticklabels([arrayfun(@(x) ['Tmatrix ', num2str(x)], 1:size(allTmatrixdata5, 2), 'UniformOutput', false), {'anisotropy tensor 22'}]);
    yticklabels([arrayfun(@(x) ['Tmatrix ', num2str(x)], 1:size(allTmatrixdata5, 2), 'UniformOutput', false), {'anisotropy tensor 22'}]);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1])
    colormap(flip(gray))

    fileName = ['Tmatrix_22_correlation_map.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);



    TmatrixStruct = struct( ...
        'data', {anisotropic_tensor, R_tensor, S_tensor, R_tensor_adj, S_tensor_adj}, ...
        'name', {'anisotropic tenosr', 'R tensor', 'S tnesor', 'R tensor adj', 'S tnesor adj'});

    % other data distribution(histogram)
    for ii = 1:length(TmatrixStruct)
        currentData = TmatrixStruct(ii).data;
        currentName = TmatrixStruct(ii).name;

        matrix11 = zeros(nn,1);
        matrix12 = zeros(nn,1);
        matrix22 = zeros(nn,1);

        for i = 1:nn
            matrix11(i) = currentData{i}(1, 1);
            matrix12(i) = currentData{i}(1, 2);
            matrix22(i) = currentData{i}(2, 2);
        end

        figure;
        t = tiledlayout("vertical");
        t.TileSpacing = 'compact';
        t.Padding = 'compact';

        nexttile;
        histogram(matrix11(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' 11 distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        nexttile;
        histogram(matrix12(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' 12 distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        nexttile;
        histogram(matrix22(:), 100);
        xlabel('value', 'Interpreter', 'latex');
        ylabel('probability', 'Interpreter', 'latex');
        title([currentName ' 22 distribution'], 'Interpreter', 'latex');
        set(gca, 'FontName', 'Times', 'FontSize', 14, 'TickLabelInterpreter', 'latex');

        fileName = [currentName, '_distribution.jpg'];
        print('-djpeg', '-r600', fileName);
        close(gcf);
    end

    cd ..
    cd ..
end