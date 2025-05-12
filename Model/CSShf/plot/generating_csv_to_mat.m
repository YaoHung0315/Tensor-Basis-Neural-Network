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
dir_root_label = [newPathPart '/output/label'];
dir_root_pred = [newPathPart '/output/pred'];
dir_root_c = [newPathPart '/function'];


cd(path)
label_dir = [dir_root_label '/label_data.csv'];
label_data = readtable(label_dir);
label_data = table2array(label_data);
label_data = label_data(2:end,:);
cd(dir_root_label)
save('label_datasheet_testing.mat', 'label_data');

cd(path)
g10_dir = [dir_root_pred '/output_g.csv'];
g10_data = readtable(g10_dir);
g10_data = table2array(g10_data);
g10_data = g10_data(2:end,:);
cd(dir_root_pred)
save('g10_datasheet_testing.mat', 'g10_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred_sum.csv'];
pred_sum_data = readtable(pred_dir);
pred_sum_data = table2array(pred_sum_data);
pred_sum_data = pred_sum_data(2:end,:);
cd(dir_root_pred)
save('pred_sum_datasheet_testing.mat', 'pred_sum_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred1.csv'];
pred_1_data = readtable(pred_dir);
pred_1_data = table2array(pred_1_data);
pred_1_data = pred_1_data(2:end,:);
cd(dir_root_pred)
save('pred_1_datasheet_testing.mat', 'pred_1_data');


cd(path)
pred_dir = [dir_root_pred '/output_pred2.csv'];
pred_2_data = readtable(pred_dir);
pred_2_data = table2array(pred_2_data);
pred_2_data = pred_2_data(2:end,:);
cd(dir_root_pred)
save('pred_2_datasheet_testing.mat', 'pred_2_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred3.csv'];
pred_3_data = readtable(pred_dir);
pred_3_data = table2array(pred_3_data);
pred_3_data = pred_3_data(2:end,:);
cd(dir_root_pred)
save('pred_3_datasheet_testing.mat', 'pred_3_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred4.csv'];
pred_4_data = readtable(pred_dir);
pred_4_data = table2array(pred_4_data);
pred_4_data = pred_4_data(2:end,:);
cd(dir_root_pred)
save('pred_4_datasheet_testing.mat', 'pred_4_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred5.csv'];
pred_5_data = readtable(pred_dir);
pred_5_data = table2array(pred_5_data);
pred_5_data = pred_5_data(2:end,:);
cd(dir_root_pred)
save('pred_5_datasheet_testing.mat', 'pred_5_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred6.csv'];
pred_6_data = readtable(pred_dir);
pred_6_data = table2array(pred_6_data);
pred_6_data = pred_6_data(2:end,:);
cd(dir_root_pred)
save('pred_6_datasheet_testing.mat', 'pred_6_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred7.csv'];
pred_7_data = readtable(pred_dir);
pred_7_data = table2array(pred_7_data);
pred_7_data = pred_7_data(2:end,:);
cd(dir_root_pred)
save('pred_7_datasheet_testing.mat', 'pred_7_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred8.csv'];
pred_8_data = readtable(pred_dir);
pred_8_data = table2array(pred_8_data);
pred_8_data = pred_8_data(2:end,:);
cd(dir_root_pred)
save('pred_8_datasheet_testing.mat', 'pred_8_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred9.csv'];
pred_9_data = readtable(pred_dir);
pred_9_data = table2array(pred_9_data);
pred_9_data = pred_9_data(2:end,:);
cd(dir_root_pred)
save('pred_9_datasheet_testing.mat', 'pred_9_data');

cd(path)
pred_dir = [dir_root_pred '/output_pred10.csv'];
pred_10_data = readtable(pred_dir);
pred_10_data = table2array(pred_10_data);
pred_10_data = pred_10_data(2:end,:);
cd(dir_root_pred)
save('pred_10_datasheet_testing.mat', 'pred_10_data');
cd ..
