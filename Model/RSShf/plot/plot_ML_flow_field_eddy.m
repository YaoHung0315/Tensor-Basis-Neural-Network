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
R_tensor = R_tensor{1};

S_tensor_adj = S_tensor_adj{1};
R_tensor_adj = R_tensor_adj{1};



%% every tensor, eddy viscosity
% 
% S_scale_data = zeros(length(S_tensor_adj), 9);
% R_scale_data = zeros(length(R_tensor_adj), 9);
% S_data = zeros(length(S_tensor), 9);
% R_data = zeros(length(R_tensor), 9);
% for i = 1:length(S_tensor_adj)
%     entry1 = S_tensor_adj{i};
%     S_scale_data(i, :) = entry1(:)';
% 
%     entry2 = R_tensor_adj{i};
%     R_scale_data(i, :) = entry2(:)';
% 
%     entry3 = S_tensor{i};
%     S_data(i, :) = entry3(:)';
% 
%     entry4 = R_tensor{i};
%     R_data(i, :) = entry4(:)';
% end
% 
% label_eddy = -label_data./S_data;
% pred_eddy = -pred_sum_data./S_data;
% 
% columns = [1 2 5];
% for i = 1:length(columns)
%     % label_eddy
%     figure;
%     label_interp = griddata(x_point(:), y_point(:), label_eddy(:,columns(i)), x, y, 'cubic');
%     p_label = pcolor(x, y, label_interp);
%     p_label.FaceColor = 'interp';
%     p_label.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Label eddy ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
% 
%     caxis([prctile(label_eddy(:, columns(i)), 5) prctile(label_eddy(:, columns(i)), 95)]);
% 
%     cd(dir_root_s);
%     fileName = ['Label_eddy_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
%     % pred_eddy
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_eddy(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred eddy ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
% 
%     caxis([prctile(pred_eddy(:, columns(i)), 5) prctile(pred_eddy(:, columns(i)), 95)]);
% 
%     cd(dir_root_s);
%     fileName = ['Pred_sum_eddy_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% end



%% double dot, eddy viscosity

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



% label_eddy_double_dot
figure;
label_interp = griddata(x_point(:), y_point(:), label_eddy_double_dot(:), x, y, 'cubic');
p_label = pcolor(x, y, label_interp);
p_label.FaceColor = 'interp';
p_label.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Label eddy double dot'], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(label_eddy_double_dot, 5) prctile(label_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['label_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_sum_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_sum_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred sum eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_sum_eddy_double_dot, 5) prctile(pred_sum_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_sum_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_1_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1_eddy_double_dot, 5) prctile(pred_1_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_2_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_2_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 2 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_2_eddy_double_dot, 5) prctile(pred_2_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_2_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_3_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_3_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 3 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_3_eddy_double_dot, 5) prctile(pred_3_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_3_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_4_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_4_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 4 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_4_eddy_double_dot, 5) prctile(pred_4_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_4_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_5_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_5_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 5 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_5_eddy_double_dot, 5) prctile(pred_5_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_5_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_6_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_6_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 6 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_6_eddy_double_dot, 5) prctile(pred_6_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_6_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_7_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_7_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 7 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_7_eddy_double_dot, 5) prctile(pred_7_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_7_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_8_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_8_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 8 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_8_eddy_double_dot, 5) prctile(pred_8_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_8_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_9_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_9_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 9 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_9_eddy_double_dot, 5) prctile(pred_9_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_9_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_10_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_10_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 10 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_10_eddy_double_dot, 5) prctile(pred_10_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_10_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);





%% double dot, eddy viscosity

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


% label_eddy_double_dot
figure;
label_interp = griddata(x_point(:), y_point(:), label_eddy_double_dot(:), x, y, 'cubic');
p_label = pcolor(x, y, label_interp);
p_label.FaceColor = 'interp';
p_label.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Label eddy double dot'], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(label_eddy_double_dot, 5) prctile(label_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['label_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_sum_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_sum_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred sum eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_sum_eddy_double_dot, 5) prctile(pred_sum_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_sum_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_1_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1_eddy_double_dot, 5) prctile(pred_1_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_2_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_2_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 2 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_2_eddy_double_dot, 5) prctile(pred_2_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_2_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_3_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_3_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 3 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_3_eddy_double_dot, 5) prctile(pred_3_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_3_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_4_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_4_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 4 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_4_eddy_double_dot, 5) prctile(pred_4_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_4_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_5_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_5_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 5 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_5_eddy_double_dot, 5) prctile(pred_5_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_5_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_6_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_6_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 6 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_6_eddy_double_dot, 5) prctile(pred_6_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_6_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_7_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_7_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 7 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_7_eddy_double_dot, 5) prctile(pred_7_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_7_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_8_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_8_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 8 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_8_eddy_double_dot, 5) prctile(pred_8_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_8_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_9_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_9_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 9 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_9_eddy_double_dot, 5) prctile(pred_9_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_9_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_10_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_10_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 10 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_10_eddy_double_dot, 5) prctile(pred_10_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_10_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);





%% double dot, eddy viscosity

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


% pred_1to1_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to1_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to1 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to1_eddy_double_dot, 5) prctile(pred_1to1_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to1_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_1to2_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to2_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to2 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to2_eddy_double_dot, 5) prctile(pred_1to2_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to2_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_1to3_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to3_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to3 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to3_eddy_double_dot, 5) prctile(pred_1to3_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to3_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_4_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to4_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to4 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to4_eddy_double_dot, 5) prctile(pred_1to4_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to4_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_5_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to5_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to5 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to5_eddy_double_dot, 5) prctile(pred_1to5_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to5_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_6_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to6_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to6 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to6_eddy_double_dot, 5) prctile(pred_1to6_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to6_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_7_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to7_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to7 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to7_eddy_double_dot, 5) prctile(pred_1to7_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to7_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_8_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to8_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to8 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to8_eddy_double_dot, 5) prctile(pred_1to8_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to8_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_9_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to9_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to9 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to9_eddy_double_dot, 5) prctile(pred_1to9_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to9_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);


% pred_10_eddy_double_dot
figure;
pred_interp = griddata(x_point(:), y_point(:), pred_1to10_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
colorbar;
box on;
axis equal;
set(gca, 'TickLabelInterpreter', 'latex');
title(['Pred 1to10 eddy double dot' ], 'Interpreter', 'latex');
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);

caxis([prctile(pred_1to10_eddy_double_dot, 5) prctile(pred_1to10_eddy_double_dot, 95)]);
% caxis([0 0.3]);

cd(dir_root_s);
fileName = ['pred_1to10_eddy_double_dot.jpg'];
print('-djpeg', '-r600', fileName);
close(gcf);

