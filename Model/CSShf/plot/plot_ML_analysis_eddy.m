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
g10_dir = [dir_root_pred '/g10_datasheet_testing.mat'];
pred_sum_dir = [dir_root_pred '/pred_sum_datasheet_testing.mat'];
pred_1_dir = [dir_root_pred '/pred_1_datasheet_testing.mat'];
pred_2_dir = [dir_root_pred '/pred_2_datasheet_testing.mat'];
pred_3_dir = [dir_root_pred '/pred_3_datasheet_testing.mat'];
pred_4_dir = [dir_root_pred '/pred_4_datasheet_testing.mat'];
pred_5_dir = [dir_root_pred '/pred_5_datasheet_testing.mat'];
pred_6_dir = [dir_root_pred '/pred_6_datasheet_testing.mat'];
pred_7_dir = [dir_root_pred '/pred_7_datasheet_testing.mat'];
pred_8_dir = [dir_root_pred '/pred_8_datasheet_testing.mat'];
pred_9_dir = [dir_root_pred '/pred_9_datasheet_testing.mat'];
pred_10_dir = [dir_root_pred '/pred_10_datasheet_testing.mat'];

Case_dir = [dir_root_loc '/ML_TBNN_testing_caseinf.mat'];

load(label_dir);
load(g10_dir);
load(pred_sum_dir);
load(pred_1_dir);
load(pred_2_dir);
load(pred_3_dir);
load(pred_4_dir);
load(pred_5_dir);
load(pred_6_dir);
load(pred_7_dir);
load(pred_8_dir);
load(pred_9_dir);
load(pred_10_dir);
load(Case_dir);

x_point = x_point{1};
y_point = y_point{1};
x = x{1};
y = y{1};
S_tensor = S_tensor{1};



%% double dot, eddy viscosity individual

label_tensor = cell(size(label_data, 1), 1);
pred_sum_tensor = cell(size(pred_sum_data, 1), 1);
pred_1_tensor = cell(size(pred_1_data, 1), 1);
pred_2_tensor = cell(size(pred_2_data, 1), 1);
pred_3_tensor = cell(size(pred_3_data, 1), 1);
pred_4_tensor = cell(size(pred_4_data, 1), 1);
pred_5_tensor = cell(size(pred_5_data, 1), 1);
pred_6_tensor = cell(size(pred_6_data, 1), 1);
pred_7_tensor = cell(size(pred_7_data, 1), 1);
pred_8_tensor = cell(size(pred_8_data, 1), 1);
pred_9_tensor = cell(size(pred_9_data, 1), 1);
pred_10_tensor = cell(size(pred_10_data, 1), 1);

for i = 1:size(label_data, 1)
    label_tensor{i} = reshape(label_data(i, :), [3, 3]);
    pred_sum_tensor{i} = reshape(pred_sum_data(i, :), [3, 3]);
    pred_1_tensor{i} = reshape(pred_1_data(i, :), [3, 3]);
    pred_2_tensor{i} = reshape(pred_2_data(i, :), [3, 3]);
    pred_3_tensor{i} = reshape(pred_3_data(i, :), [3, 3]);
    pred_4_tensor{i} = reshape(pred_4_data(i, :), [3, 3]);
    pred_5_tensor{i} = reshape(pred_5_data(i, :), [3, 3]);
    pred_6_tensor{i} = reshape(pred_6_data(i, :), [3, 3]);
    pred_7_tensor{i} = reshape(pred_7_data(i, :), [3, 3]);
    pred_8_tensor{i} = reshape(pred_8_data(i, :), [3, 3]);
    pred_9_tensor{i} = reshape(pred_9_data(i, :), [3, 3]);
    pred_10_tensor{i} = reshape(pred_10_data(i, :), [3, 3]);
end

S_matrix = zeros(length(S_tensor), 1);
label_matrix = zeros(length(label_tensor), 1);
pred_sum_matrix = zeros(length(pred_sum_data), 1);
pred_1_matrix = zeros(length(pred_1_data), 1);
pred_2_matrix = zeros(length(pred_2_data), 1);
pred_3_matrix = zeros(length(pred_3_data), 1);
pred_4_matrix = zeros(length(pred_4_data), 1);
pred_5_matrix = zeros(length(pred_5_data), 1);
pred_6_matrix = zeros(length(pred_6_data), 1);
pred_7_matrix = zeros(length(pred_7_data), 1);
pred_8_matrix = zeros(length(pred_8_data), 1);
pred_9_matrix = zeros(length(pred_9_data), 1);
pred_10_matrix = zeros(length(pred_10_data), 1);

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

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1_tensor{i};
    pred_1_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_2_tensor{i};
    pred_2_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_3_tensor{i};
    pred_3_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_4_tensor{i};
    pred_4_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_5_tensor{i};
    pred_5_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_6_tensor{i};
    pred_6_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_7_tensor{i};
    pred_7_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_8_tensor{i};
    pred_8_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_9_tensor{i};
    pred_9_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_10_tensor{i};
    pred_10_matrix(i) = trace(matrix1 * matrix2');
end
label_eddy_double_dot = -(label_matrix)./(S_matrix.*S_matrix);
pred_sum_eddy_double_dot = -(pred_sum_matrix)./(S_matrix.*S_matrix);
pred_1_eddy_double_dot = -(pred_1_matrix)./(S_matrix.*S_matrix);
pred_2_eddy_double_dot = -(pred_2_matrix)./(S_matrix.*S_matrix);
pred_3_eddy_double_dot = -(pred_3_matrix)./(S_matrix.*S_matrix);
pred_4_eddy_double_dot = -(pred_4_matrix)./(S_matrix.*S_matrix);
pred_5_eddy_double_dot = -(pred_5_matrix)./(S_matrix.*S_matrix);
pred_6_eddy_double_dot = -(pred_6_matrix)./(S_matrix.*S_matrix);
pred_7_eddy_double_dot = -(pred_7_matrix)./(S_matrix.*S_matrix);
pred_8_eddy_double_dot = -(pred_8_matrix)./(S_matrix.*S_matrix);
pred_9_eddy_double_dot = -(pred_9_matrix)./(S_matrix.*S_matrix);
pred_10_eddy_double_dot = -(pred_10_matrix)./(S_matrix.*S_matrix);







%% bij individaul double dot, eddy viscosity accuracy
cd(dir_root_s)

columns = [1];
for i = 1:length(columns)

    R2_values = zeros(1, 11);

    actual = label_eddy_double_dot(:, columns(i));
    pred_data = {pred_sum_eddy_double_dot, pred_1_eddy_double_dot, pred_2_eddy_double_dot, pred_3_eddy_double_dot, pred_4_eddy_double_dot, ...
                 pred_5_eddy_double_dot, pred_6_eddy_double_dot, pred_7_eddy_double_dot, pred_8_eddy_double_dot, pred_9_eddy_double_dot, pred_10_eddy_double_dot};
    for j = 1:length(pred_data)
        predicted = pred_data{j}(:, columns(i));
        SS_res = sum((actual - predicted).^2);
        SS_tot = sum((actual - mean(actual)).^2);
        R2_values(j) = 1 - (SS_res / SS_tot);
    end

    bar(R2_values);
    title(['accuracy R^2 bij double dot eddy'], 'Interpreter', 'latex');
    set(gca, 'xticklabel', {'eddysum', 'eddy1', 'eddy2', 'eddy3', 'eddy4', 'eddy5', 'eddy6', 'eddy7', 'eddy8', 'eddy9', 'eddy10'});
    xlabel('Output', 'Interpreter', 'latex');
    ylabel('R^2 Value', 'Interpreter', 'latex');
    grid on;

    fileName = ['accuracy_Pred_double_dot_eddy.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

end


%% bij individaul double dot, eddy viscosity correlation map

cd(dir_root_s)

columns = [1];
for i = 1:length(columns)

    actual = label_eddy_double_dot(:, columns(i));
    pred_data = {pred_sum_eddy_double_dot, pred_1_eddy_double_dot, pred_2_eddy_double_dot, pred_3_eddy_double_dot, pred_4_eddy_double_dot, ...
                 pred_5_eddy_double_dot, pred_6_eddy_double_dot, pred_7_eddy_double_dot, pred_8_eddy_double_dot, pred_9_eddy_double_dot, pred_10_eddy_double_dot};

    data_matrix = zeros(size(actual, 1), length(pred_data) + 1);
    data_matrix(:, end) = actual;

    for j = 1:length(pred_data)
        data_matrix(:, j) = pred_data{j}(:, columns(i));
    end

    correlation_matrix = corr(data_matrix);

    imagesc(abs(correlation_matrix));
    colorbar;
    colormap(jet);
    title(['correlation bij double dot eddy'], 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));

    pred_labels = arrayfun(@(x) ['eddy', num2str(x)], 1:length(pred_data)-1, 'UniformOutput', false);
    all_labels = [{'eddysum'}, pred_labels, {'label eddy'}];

    xticklabels(all_labels);
    yticklabels(all_labels);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1]);


    fileName = ['correlation_Pred_double_dot_eddy.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

end

%% plot individaul data bij double dot, eddy viscosity
cd(dir_root_s)

columns = [1];
colors = {'k-o', 'r-o', 'g', 'b', 'm', 'c', 'y', [0.5 0 0], [0 0.5 0], [0 0 0.5], [0.5 0.5 0], [0.3 0.3 0.3]};
for i = 1:length(columns)
    actual = label_eddy_double_dot(:, columns(i));
    nn = length(actual);
    
    predicted_sum = pred_sum_eddy_double_dot(:, columns(i));
    predicted_1 = pred_1_eddy_double_dot(:, columns(i));
    predicted_2 = pred_2_eddy_double_dot(:, columns(i));
    predicted_3 = pred_3_eddy_double_dot(:, columns(i));
    predicted_4 = pred_4_eddy_double_dot(:, columns(i));
    predicted_5 = pred_5_eddy_double_dot(:, columns(i));
    predicted_6 = pred_6_eddy_double_dot(:, columns(i));
    predicted_7 = pred_7_eddy_double_dot(:, columns(i));
    predicted_8 = pred_8_eddy_double_dot(:, columns(i));
    predicted_9 = pred_9_eddy_double_dot(:, columns(i));
    predicted_10 = pred_10_eddy_double_dot(:, columns(i));

    plot(1:5:nn, actual(1:5:nn), colors{1}, 'LineWidth', 1.0, 'MarkerSize',2); hold on;
    plot(1:5:nn, predicted_sum(1:5:nn), colors{2}, 'LineWidth', 0.5, 'MarkerSize',2); hold on;
    plot(1:5:nn, predicted_1(1:5:nn), colors{3}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_2(1:5:nn), colors{4}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_3(1:5:nn), colors{5}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_4(1:5:nn), colors{6}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_5(1:5:nn), colors{7}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_6(1:5:nn), 'color', colors{8}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_7(1:5:nn), 'color', colors{9}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_8(1:5:nn), 'color', colors{10}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_9(1:5:nn), 'color', colors{11}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_10(1:5:nn), 'color', colors{12}, 'LineWidth', 0.5); hold on;

    xlim([0 nn]);

    title(['comparison bij double dot eddy'], 'Interpreter', 'latex');
    xlabel('Sample', 'Interpreter', 'latex');
    ylabel('Value', 'Interpreter', 'latex');
    legend({'label eddy', 'eddysum', 'eddy1', 'eddy2', 'eddy3', 'eddy4', 'eddy5', 'eddy6', 'eddy7', 'eddy8', 'eddy9', 'eddy10'}, 'Interpreter', 'latex');
    grid on;
    
    fileName = ['comparison_Pred_double_dot_eddy.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);
end





%% double dot, eddy viscosity summation

pred_1to1_tensor = cell(size(pred_1_data, 1), 1);
pred_1to2_tensor = cell(size(pred_2_data, 1), 1);
pred_1to3_tensor = cell(size(pred_3_data, 1), 1);
pred_1to4_tensor = cell(size(pred_4_data, 1), 1);
pred_1to5_tensor = cell(size(pred_5_data, 1), 1);
pred_1to6_tensor = cell(size(pred_6_data, 1), 1);
pred_1to7_tensor = cell(size(pred_7_data, 1), 1);
pred_1to8_tensor = cell(size(pred_8_data, 1), 1);
pred_1to9_tensor = cell(size(pred_9_data, 1), 1);
pred_1to10_tensor = cell(size(pred_10_data, 1), 1);

for i = 1:size(label_data, 1)
    pred_1to1_tensor{i} = reshape(pred_1_data(i, :), [3, 3]);
    pred_1to2_tensor{i} = pred_1to1_tensor{i} + reshape(pred_2_data(i, :), [3, 3]);
    pred_1to3_tensor{i} = pred_1to2_tensor{i} + reshape(pred_3_data(i, :), [3, 3]);
    pred_1to4_tensor{i} = pred_1to3_tensor{i} + reshape(pred_4_data(i, :), [3, 3]);
    pred_1to5_tensor{i} = pred_1to4_tensor{i} + reshape(pred_5_data(i, :), [3, 3]);
    pred_1to6_tensor{i} = pred_1to5_tensor{i} + reshape(pred_6_data(i, :), [3, 3]);
    pred_1to7_tensor{i} = pred_1to6_tensor{i} + reshape(pred_7_data(i, :), [3, 3]);
    pred_1to8_tensor{i} = pred_1to7_tensor{i} + reshape(pred_8_data(i, :), [3, 3]);
    pred_1to9_tensor{i} = pred_1to8_tensor{i} + reshape(pred_9_data(i, :), [3, 3]);
    pred_1to10_tensor{i} = pred_1to9_tensor{i} + reshape(pred_10_data(i, :), [3, 3]);
end

S_matrix = zeros(length(S_tensor), 1);
pred_1to1_matrix = zeros(length(pred_1_data), 1);
pred_1to2_matrix = zeros(length(pred_2_data), 1);
pred_1to3_matrix = zeros(length(pred_3_data), 1);
pred_1to4_matrix = zeros(length(pred_4_data), 1);
pred_1to5_matrix = zeros(length(pred_5_data), 1);
pred_1to6_matrix = zeros(length(pred_6_data), 1);
pred_1to7_matrix = zeros(length(pred_7_data), 1);
pred_1to8_matrix = zeros(length(pred_8_data), 1);
pred_1to9_matrix = zeros(length(pred_9_data), 1);
pred_1to10_matrix = zeros(length(pred_10_data), 1);

for i = 1:length(S_tensor)
    matrix = S_tensor{i};
    S_matrix(i) = sqrt(trace(matrix * matrix'));
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to1_tensor{i};
    pred_1to1_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to2_tensor{i};
    pred_1to2_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to3_tensor{i};
    pred_1to3_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to4_tensor{i};
    pred_1to4_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to5_tensor{i};
    pred_1to5_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to6_tensor{i};
    pred_1to6_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to7_tensor{i};
    pred_1to7_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to8_tensor{i};
    pred_1to8_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to9_tensor{i};
    pred_1to9_matrix(i) = trace(matrix1 * matrix2');
end

for i = 1:length(S_tensor)
    matrix1 = S_tensor{i};
    matrix2 = pred_1to10_tensor{i};
    pred_1to10_matrix(i) = trace(matrix1 * matrix2');
end
pred_1to1_eddy_double_dot = -(pred_1to1_matrix)./(S_matrix.*S_matrix);
pred_1to2_eddy_double_dot = -(pred_1to2_matrix)./(S_matrix.*S_matrix);
pred_1to3_eddy_double_dot = -(pred_1to3_matrix)./(S_matrix.*S_matrix);
pred_1to4_eddy_double_dot = -(pred_1to4_matrix)./(S_matrix.*S_matrix);
pred_1to5_eddy_double_dot = -(pred_1to5_matrix)./(S_matrix.*S_matrix);
pred_1to6_eddy_double_dot = -(pred_1to6_matrix)./(S_matrix.*S_matrix);
pred_1to7_eddy_double_dot = -(pred_1to7_matrix)./(S_matrix.*S_matrix);
pred_1to8_eddy_double_dot = -(pred_1to8_matrix)./(S_matrix.*S_matrix);
pred_1to9_eddy_double_dot = -(pred_1to9_matrix)./(S_matrix.*S_matrix);
pred_1to10_eddy_double_dot = -(pred_1to10_matrix)./(S_matrix.*S_matrix);



%% bij individaul double dot, eddy viscosity accuracy
cd(dir_root_s)

columns = [1];
for i = 1:length(columns)

    R2_values = zeros(1, 11);

    actual = label_eddy_double_dot(:, columns(i));
    pred_data = {pred_sum_eddy_double_dot, pred_1to1_eddy_double_dot, pred_1to2_eddy_double_dot, pred_1to3_eddy_double_dot, pred_1to4_eddy_double_dot, ...
                 pred_1to5_eddy_double_dot, pred_1to6_eddy_double_dot, pred_1to7_eddy_double_dot, pred_1to8_eddy_double_dot, pred_1to9_eddy_double_dot, pred_1to10_eddy_double_dot};
    for j = 1:length(pred_data)
        predicted = pred_data{j}(:, columns(i));
        SS_res = sum((actual - predicted).^2);
        SS_tot = sum((actual - mean(actual)).^2);
        R2_values(j) = 1 - (SS_res / SS_tot);
    end

    bar(R2_values);
    title(['accuracy R^2 bij double dot eddy summation'], 'Interpreter', 'latex');
    set(gca, 'xticklabel', {'eddysum', 'eddy1to1', 'eddy1to2', 'eddy1to3', 'eddy1to4', 'eddy1to5', 'eddy1to6', 'eddy1to7', 'eddy1to8', 'eddy1to9', 'eddy1to10'});
    xlabel('Output', 'Interpreter', 'latex');
    ylabel('R^2 Value', 'Interpreter', 'latex');
    grid on;

    fileName = ['accuracy_Pred_double_dot_eddy_summation.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

end


%% bij individaul double dot, eddy viscosity correlation map

cd(dir_root_s)

columns = [1];
for i = 1:length(columns)

    actual = label_eddy_double_dot(:, columns(i));
    pred_data = {pred_sum_eddy_double_dot, pred_1to1_eddy_double_dot, pred_1to2_eddy_double_dot, pred_1to3_eddy_double_dot, pred_1to4_eddy_double_dot, ...
                 pred_1to5_eddy_double_dot, pred_1to6_eddy_double_dot, pred_1to7_eddy_double_dot, pred_1to8_eddy_double_dot, pred_1to9_eddy_double_dot, pred_1to10_eddy_double_dot};

    data_matrix = zeros(size(actual, 1), length(pred_data) + 1);
    data_matrix(:, end) = actual;

    for j = 1:length(pred_data)
        data_matrix(:, j) = pred_data{j}(:, columns(i));
    end

    correlation_matrix = corr(data_matrix);

    imagesc(abs(correlation_matrix));
    colorbar;
    colormap(jet);
    title(['correlation bij double dot eddy summation'], 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));

    pred_labels = arrayfun(@(x) ['eddy1to', num2str(x)], 1:length(pred_data)-1, 'UniformOutput', false);
    all_labels = [{'eddysum'}, pred_labels, {'label eddy'}];

    xticklabels(all_labels);
    yticklabels(all_labels);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1]);


    fileName = ['correlation_Pred_double_dot_eddy_summation.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

end

%% plot individaul data bij double dot, eddy viscosity
cd(dir_root_s)

columns = [1];
colors = {'k-o', 'r-o', 'g', 'b', 'm', 'c', 'y', [0.5 0 0], [0 0.5 0], [0 0 0.5], [0.5 0.5 0], [0.3 0.3 0.3]};
for i = 1:length(columns)
    actual = label_eddy_double_dot(:, columns(i));
    nn = length(actual);
    
    predicted_sum = pred_sum_eddy_double_dot(:, columns(i));
    predicted_1 = pred_1to1_eddy_double_dot(:, columns(i));
    predicted_2 = pred_1to2_eddy_double_dot(:, columns(i));
    predicted_3 = pred_1to3_eddy_double_dot(:, columns(i));
    predicted_4 = pred_1to4_eddy_double_dot(:, columns(i));
    predicted_5 = pred_1to5_eddy_double_dot(:, columns(i));
    predicted_6 = pred_1to6_eddy_double_dot(:, columns(i));
    predicted_7 = pred_1to7_eddy_double_dot(:, columns(i));
    predicted_8 = pred_1to8_eddy_double_dot(:, columns(i));
    predicted_9 = pred_1to9_eddy_double_dot(:, columns(i));
    predicted_10 = pred_1to10_eddy_double_dot(:, columns(i));

    plot(1:5:nn, actual(1:5:nn), colors{1}, 'LineWidth', 1.0, 'MarkerSize',2); hold on;
    plot(1:5:nn, predicted_sum(1:5:nn), colors{2}, 'LineWidth', 0.5, 'MarkerSize',2); hold on;
    plot(1:5:nn, predicted_1(1:5:nn), colors{3}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_2(1:5:nn), colors{4}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_3(1:5:nn), colors{5}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_4(1:5:nn), colors{6}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_5(1:5:nn), colors{7}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_6(1:5:nn), 'color', colors{8}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_7(1:5:nn), 'color', colors{9}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_8(1:5:nn), 'color', colors{10}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_9(1:5:nn), 'color', colors{11}, 'LineWidth', 0.5); hold on;
    plot(1:5:nn, predicted_10(1:5:nn), 'color', colors{12}, 'LineWidth', 0.5); hold on;

    xlim([0 nn]);

    title(['comparison bij double dot eddy summation'], 'Interpreter', 'latex');
    xlabel('Sample', 'Interpreter', 'latex');
    ylabel('Value', 'Interpreter', 'latex');
    legend({'label eddy', 'eddysum', 'eddy1', 'eddy2', 'eddy3', 'eddy4', 'eddy5', 'eddy6', 'eddy7', 'eddy8', 'eddy9', 'eddy10'}, 'Interpreter', 'latex');
    grid on;
    
    fileName = ['comparison_Pred_double_dot_eddy_summation.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);
end



