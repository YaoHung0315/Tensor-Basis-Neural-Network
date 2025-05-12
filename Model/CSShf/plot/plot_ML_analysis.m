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

%% bij accuracy
cd(dir_root_s)

columns = [1 2 5];
for i = 1:length(columns)

    R2_values = zeros(1, 11);

    actual = label_data(:, columns(i));
    pred_data = {pred_sum_data, pred_1_data, pred_2_data, pred_3_data, pred_4_data, ...
                 pred_5_data, pred_6_data, pred_7_data, pred_8_data, pred_9_data, pred_10_data};
    for j = 1:length(pred_data)
        predicted = pred_data{j}(:, columns(i));
        SS_res = sum((actual - predicted).^2);
        SS_tot = sum((actual - mean(actual)).^2);
        R2_values(j) = 1 - (SS_res / SS_tot);
    end

    bar(R2_values);
    title(['accuracy R^2 bij ' num2str(columns(i))], 'Interpreter', 'latex');
    set(gca, 'xticklabel', {'Predsum', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'Pred5', 'Pred6', 'Pred7', 'Pred8', 'Pred9', 'Pred10'});
    xlabel('Output', 'Interpreter', 'latex');
    ylabel('R^2 Value', 'Interpreter', 'latex');
    grid on;

    fileName = ['accuracy_Pred_' num2str(columns(i)) '.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

end


%% bij correlation map

cd(dir_root_s)

columns = [1 2 5];
for i = 1:length(columns)


    pred_data = {pred_sum_data, pred_1_data, pred_2_data, pred_3_data, pred_4_data, ...
        pred_5_data, pred_6_data, pred_7_data, pred_8_data, pred_9_data, pred_10_data};

    data_matrix = zeros(size(label_data, 1), length(pred_data) + 1);
    data_matrix(:, end) = label_data(:, columns(i));

    for j = 1:length(pred_data)
        data_matrix(:, j) = pred_data{j}(:, columns(i));
    end

    correlation_matrix = corr(data_matrix);

    imagesc(abs(correlation_matrix));
    colorbar;
    colormap(jet);
    title(['correlation bij ' num2str(columns(i))], 'Interpreter', 'latex');
    xlabel('Features and Label', 'Interpreter', 'latex');
    ylabel('Features and Label', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');

    xticks(1:size(correlation_matrix, 2));
    yticks(1:size(correlation_matrix, 1));

    pred_labels = arrayfun(@(x) ['Pred', num2str(x)], 1:length(pred_data)-1, 'UniformOutput', false);
    all_labels = [{'Predsum'}, pred_labels, {['label bij ' num2str(columns(i))]}];

    xticklabels(all_labels);
    yticklabels(all_labels);
    xtickangle(45);
    ytickangle(45);
    caxis([0 1]);


    fileName = ['correlation_Pred_' num2str(columns(i)) '.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

end

%% plot data bij
cd(dir_root_s)

columns = [1 2 5];
colors = {'k-o', 'r-o', 'g', 'b', 'm', 'c', 'y', [0.5 0 0], [0 0.5 0], [0 0 0.5], [0.5 0.5 0], [0.3 0.3 0.3]};
for i = 1:length(columns)
    actual = label_data(:, columns(i));
    nn = length(actual);
    
    predicted_sum = pred_sum_data(:, columns(i));
    predicted_1 = pred_1_data(:, columns(i));
    predicted_2 = pred_2_data(:, columns(i));
    predicted_3 = pred_3_data(:, columns(i));
    predicted_4 = pred_4_data(:, columns(i));
    predicted_5 = pred_5_data(:, columns(i));
    predicted_6 = pred_6_data(:, columns(i));
    predicted_7 = pred_7_data(:, columns(i));
    predicted_8 = pred_8_data(:, columns(i));
    predicted_9 = pred_9_data(:, columns(i));
    predicted_10 = pred_10_data(:, columns(i));

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

    title(['comparison bij ' num2str(columns(i))], 'Interpreter', 'latex');
    xlabel('Sample', 'Interpreter', 'latex');
    ylabel('Value', 'Interpreter', 'latex');
    legend({'label bij', 'Predsum', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'Pred5', 'Pred6', 'Pred7', 'Pred8', 'Pred9', 'Pred10'}, 'Interpreter', 'latex');
    grid on;
    
    fileName = ['comparison_bij_' num2str(columns(i)) '.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);
end






