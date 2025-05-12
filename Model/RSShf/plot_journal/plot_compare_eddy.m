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
dir_root_s =   [path '/fig_journal'];

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
R_tensor = R_tensor{1};

S_tensor_adj = S_tensor_adj{1};
R_tensor_adj = R_tensor_adj{1};




%% double dot, eddy viscosity

label_tensor = cell(size(label_data, 1), 1);
pred_sum_tensor = cell(size(pred_sum_data, 1), 1);


for i = 1:size(label_data, 1)
    label_tensor{i} = reshape(label_data(i, :), [3, 3]);
    pred_sum_tensor{i} = reshape(pred_sum_data(i, :), [3, 3]);

end

S_matrix = zeros(length(S_tensor), 1);
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


fig = figure('Units', 'pixels', 'Position', [100 100 1000 500], 'PaperPositionMode', 'auto');

t = tiledlayout(2,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile;
label_interp = griddata(x_point(:), y_point(:), label_eddy_double_dot(:), x, y, 'cubic');
p_label = pcolor(x, y, label_interp);
p_label.FaceColor = 'interp';
p_label.EdgeColor = 'none';
colormap(jet);
box on;
set(gca, 'TickLabelInterpreter', 'latex', ...
         'FontSize', 20, ...
         'XTickLabel', []);
title('LES, eddy viscosity', 'Interpreter', 'latex', 'FontSize', 20);
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);
caxis([prctile(label_eddy_double_dot, 5) prctile(label_eddy_double_dot, 95)]);


nexttile;
pred_interp = griddata(x_point(:), y_point(:), pred_sum_eddy_double_dot(:), x, y, 'cubic');
p_pred = pcolor(x, y, pred_interp);
p_pred.FaceColor = 'interp';
p_pred.EdgeColor = 'none';
colormap(jet);
c = colorbar;
c.TickLabelInterpreter = 'latex';
c.FontSize = 20;
box on;
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
title('TBNN, eddy viscosity', 'Interpreter', 'latex', 'FontSize', 20);
xlim([min(x(:)) max(x(:))]);
ylim([min(y(:)) max(y(:))]);
caxis([prctile(label_eddy_double_dot, 5) prctile(label_eddy_double_dot, 95)]);

cd(dir_root_s);
print(fig, 'compare_eddy_double_dot', '-djpeg', '-r600');
close(fig);