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


%% g10 

columns = [1 2 3 4 5 6 7 8 9 10];
for i = 1:length(columns)
    % label_eddy
    figure;
    label_interp = griddata(x_point(:), y_point(:), g10_data(:,columns(i)), x, y, 'cubic');
    p_label = pcolor(x, y, label_interp);
    p_label.FaceColor = 'interp';
    p_label.EdgeColor = 'none';
    colormap(jet);
    colorbar;
    box on;
    axis equal;
    set(gca, 'TickLabelInterpreter', 'latex');
    title(['G ' num2str(columns(i))], 'Interpreter', 'latex');
    xlim([min(x(:)) max(x(:))]);
    ylim([min(y(:)) max(y(:))]);

    caxis([prctile(g10_data(:, columns(i)), 5) prctile(g10_data(:, columns(i)), 95)]);

    cd(dir_root_s);
    fileName = ['flow_G_' num2str(columns(i)) '.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);
end



%%
columns = [1 2 5];
for i = 1:length(columns)
    % label_data
    figure;
    label_interp = griddata(x_point(:), y_point(:), label_data(:, columns(i)), x, y, 'cubic');
    p_label = pcolor(x, y, label_interp);
    p_label.FaceColor = 'interp';
    p_label.EdgeColor = 'none';
    colormap(jet);
    colorbar;
    box on;
    axis equal;
    set(gca, 'TickLabelInterpreter', 'latex');
    title(['Label ' num2str(columns(i))], 'Interpreter', 'latex');
    xlim([min(x(:)) max(x(:))]);
    ylim([min(y(:)) max(y(:))]);
    cd(dir_root_s);
    fileName = ['flow_Label_' num2str(columns(i)) '.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

    % pred_sum_data
    figure;
    pred_interp = griddata(x_point(:), y_point(:), pred_sum_data(:, columns(i)), x, y, 'cubic');
    p_pred = pcolor(x, y, pred_interp);
    p_pred.FaceColor = 'interp';
    p_pred.EdgeColor = 'none';
    colormap(jet);
    colorbar;
    box on;
    axis equal;
    set(gca, 'TickLabelInterpreter', 'latex');
    title(['Pred sum ' num2str(columns(i))], 'Interpreter', 'latex');
    xlim([min(x(:)) max(x(:))]);
    ylim([min(y(:)) max(y(:))]);
    cd(dir_root_s);
    fileName = ['flow_Pred_sum_' num2str(columns(i)) '.jpg'];
    print('-djpeg', '-r600', fileName);
    close(gcf);

% 
%     % pred_1_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_1_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_2_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_2_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 2 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_2_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_3_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_3_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 3 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_3_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_4_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_4_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 4 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_4_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_5_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_5_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 5 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_5_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
%     % pred_6_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_6_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 6 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_6_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_7_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_7_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 7 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_7_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_8_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_8_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 8 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_8_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_9_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_9_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 9 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_9_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_10_data
%     figure;
%     pred_interp = griddata(x_point(:), y_point(:), pred_10_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 10 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_10_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
end
% 
% 
% 
% 
% %%
% columns = [1 2 5];
% for i = 1:length(columns)
% 
%     % pred_1_data
%     figure;
%     pred_data(:, columns(i)) = pred_1_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to1 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to1_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_2_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_2_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to2 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to2_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_3_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_3_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to3 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to3_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_4_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_4_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to4 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to4_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_5_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_5_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to5 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to5_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_6_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_6_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to6 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to6_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_7_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_7_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to7 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to7_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_8_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_8_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to8 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to8_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
%     % pred_9_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_9_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to9 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to9_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% 
%     % pred_10_data
%     figure;
%     pred_data(:, columns(i)) = pred_data(:, columns(i))+pred_10_data(:, columns(i));
%     pred_interp = griddata(x_point(:), y_point(:), pred_data(:, columns(i)), x, y, 'cubic');
%     p_pred = pcolor(x, y, pred_interp);
%     p_pred.FaceColor = 'interp';
%     p_pred.EdgeColor = 'none';
%     colormap(jet);
%     colorbar;
%     box on;
%     axis equal;
%     set(gca, 'TickLabelInterpreter', 'latex');
%     title(['Pred 1to10 ' num2str(columns(i))], 'Interpreter', 'latex');
%     xlim([min(x(:)) max(x(:))]);
%     ylim([min(y(:)) max(y(:))]);
%     cd(dir_root_s);
%     fileName = ['flow_Pred_1to10_' num2str(columns(i)) '.jpg'];
%     print('-djpeg', '-r600', fileName);
%     close(gcf);
% 
% 
% end
