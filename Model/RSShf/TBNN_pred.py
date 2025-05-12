import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network import Net


# Define the dataset class
class turDataset(Dataset):
    def __init__(self, mat_path, num_lambda, num_t):
        data = sio.loadmat(mat_path)
        self.lambda_data = torch.tensor(data['lambda_data'][:, :num_lambda], dtype=torch.float32)
        self.anisotropic_tensor_data = torch.tensor(data['anisotropic_tensor_data'], dtype=torch.float32)
        t_cell = data['T_matrix_data'].flatten()
        # Extract the first 'num_t' columns from each cell (each cell is a 9x10 array)
        self.T_matrix_data = [torch.tensor(np.squeeze(x)[:, 0:num_t], dtype=torch.float32) for x in t_cell]

    def __len__(self):
        return len(self.lambda_data)

    def __getitem__(self, index):
        return {
            'lambda_data': self.lambda_data[index],
            'anisotropic_tensor_data': self.anisotropic_tensor_data[index],
            'T_matrix_data': self.T_matrix_data[index]
        }

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Read hyperparameters
hyper_parameters_path = './output/hyper_parameters.csv'
hyper_parameters_df = pd.read_csv(hyper_parameters_path)

width = hyper_parameters_df['width'][0]
layer = hyper_parameters_df['layer'][0]
max_epochs = hyper_parameters_df['max_epochs'][0]
learning_rate = hyper_parameters_df['learning_rate'][0]
optimizer_name = hyper_parameters_df['optimizer_name'][0]
patience = hyper_parameters_df['patience'][0]
k_fold = hyper_parameters_df['k_fold'][0]
num_lambda = hyper_parameters_df['num_lambda'][0]
num_t = hyper_parameters_df['num_t'][0]

# Initialize weights
weight_for_g = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
tmp_weight_g = torch.Tensor(weight_for_g).unsqueeze(0).unsqueeze(1).permute(1, 0, 2)

# Initialize model and dataset for prediction
batchSize = 1
test_mat_path = './data/ML_TBNN_testing.mat'
test_dataset = turDataset(mat_path=test_mat_path, num_lambda=num_lambda, num_t=num_t)
validation_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

# Directly set parameters
model_save_path = 'output/model_fold.pkl'
out_dir = model_save_path.split('/')[0]
split_path = model_save_path.split('/')[1].split('_')

# Folder paths for predictions and labels
pred_path = os.path.join(out_dir, "pred/")
label_path = os.path.join(out_dir, "label/")
os.makedirs(pred_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)

# Extract file name from test dataset path
test_dataset_name = os.path.splitext(os.path.basename(test_mat_path))[0]


# Load each fold's model
models = []
for fold in range(k_fold):
    fold_model_save_path = model_save_path.replace('.pkl', f'{fold+1}.pkl')
    net = Net(width, layer, num_lambda, num_t).to(device)
    net.load_state_dict(torch.load(fold_model_save_path, map_location=device))
    net.to(device)
    net.eval()
    models.append(net)


# Prediction and saving results
output_pred1 = []
output_pred2 = []
output_pred3 = []
output_pred4 = []
output_pred5 = []
output_pred6 = []
output_pred7 = []
output_pred8 = []
output_pred9 = []
output_pred10 = []
label_data = []
output_g = []
output_pred_sum = []

# Perform predictions
for j, data in enumerate(validation_loader, 0):
    lambda_data, T_matrix_data, anisotropic_tensor_data = (
        data['lambda_data'], data['T_matrix_data'], data['anisotropic_tensor_data']
    )

    input_tensor = lambda_data.to(device)  # lambda_data
    label = anisotropic_tensor_data.to(device)  # ground truth

    predictions = [[] for _ in range(num_t)]
    output_v = []

    for model in models:
        output = model(input_tensor)  # lambda_data => G
        output = output.unsqueeze(1)  # expand dim for matmul
        t_tensor = T_matrix_data.to(device).permute(0, 2, 1)  # T_matrix_data

        for i in range(num_t):
            slice_input = t_tensor[:, i, :]
            slice_output = output[:, :, i]
            predict = slice_output.matmul(slice_input)
            predict = predict.squeeze(1)
            predictions[i].append(predict)

        output_v.append(output.squeeze(1))  # ensure proper shape

    # Average predictions from all models for each of the num_t outputs
    average_predicts = [torch.mean(torch.stack(preds), dim=0) for preds in predictions]
    tmp_g = torch.mean(torch.stack(output_v), dim=0)

    # Append predictions to corresponding lists
    for i, avg_predict in enumerate(average_predicts):
        if i == 0:
            output_pred1.extend(avg_predict.cpu().detach().tolist())
        elif i == 1:
            output_pred2.extend(avg_predict.cpu().detach().tolist())
        elif i == 2:
            output_pred3.extend(avg_predict.cpu().detach().tolist())
        elif i == 3:
            output_pred4.extend(avg_predict.cpu().detach().tolist())
        elif i == 4:
            output_pred5.extend(avg_predict.cpu().detach().tolist())
        elif i == 5:
            output_pred6.extend(avg_predict.cpu().detach().tolist())
        elif i == 6:
            output_pred7.extend(avg_predict.cpu().detach().tolist())
        elif i == 7:
            output_pred8.extend(avg_predict.cpu().detach().tolist())
        elif i == 8:
            output_pred9.extend(avg_predict.cpu().detach().tolist())
        elif i == 9:
            output_pred10.extend(avg_predict.cpu().detach().tolist())

    # Sum the num_t output predictions
    sum_preds = sum(average_predicts)
    
    output_pred_sum.extend(sum_preds.cpu().detach().tolist())
    label_data.extend(label.cpu().detach().tolist())
    output_g.extend(tmp_g.cpu().detach().tolist())


pred_paths = [globals()[f"output_pred{i+1}"] for i in range(num_t)]
pred_path_names = [f"output_pred{i+1}" for i in range(num_t)]


for i, pred in enumerate(pred_paths):
    pd.DataFrame(pred).to_csv(pred_path + f'{pred_path_names[i]}.csv', index=False)
pd.DataFrame(output_pred_sum).to_csv(pred_path + 'output_pred_sum.csv', index=False)
pd.DataFrame(label_data).to_csv(label_path + 'label_data.csv', index=False)
pd.DataFrame(output_g).to_csv(pred_path + 'output_g.csv', index=False)

# Calculate evaluation metrics
def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def calculate_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def calculate_r2(predictions, targets):
    ss_tot = np.sum((targets - np.mean(targets))**2)
    ss_res = np.sum((targets - predictions)**2)
    return 1 - ss_res / ss_tot

label_data_np = np.array(label_data)

for i, output_pred in enumerate(pred_paths, 1):
    output_pred_np = np.array(output_pred)
    mae = calculate_mae(output_pred_np, label_data_np)
    rmse = calculate_rmse(output_pred_np, label_data_np)
    r2 = calculate_r2(output_pred_np, label_data_np)

    print(f'Output Pred {i}:')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')

    with open(pred_path + f'metrics_output_pred{i}.txt', 'w') as f:
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'R²: {r2:.4f}\n')

    for feature_idx in [0, 1, 4]:
        plt.figure(figsize=(10, 5))
        plt.plot(output_pred_np[:500, feature_idx], label=f'Predicted {i}', marker='o')
        plt.plot(label_data_np[:500, feature_idx], label='True', marker='x')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'Predicted vs True Values (Output Pred {i} Feature {feature_idx + 1})')
        plt.legend()
        plt.grid(True)
        plt.savefig(pred_path + f'pred_vs_true_output_pred{i}_feature{feature_idx + 1}.png')
        plt.close()

# Perform the same operations for output_pred_sum
output_pred_sum_np = np.array(output_pred_sum)
mae = calculate_mae(output_pred_sum_np, label_data_np)
rmse = calculate_rmse(output_pred_sum_np, label_data_np)
r2 = calculate_r2(output_pred_sum_np, label_data_np)

print(f'Output Pred Sum:')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

with open(pred_path + 'metrics_output_pred_sum.txt', 'w') as f:
    f.write(f'MAE: {mae:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'R²: {r2:.4f}\n')

for feature_idx in [0, 1, 4]:
    plt.figure(figsize=(10, 5))
    plt.plot(output_pred_sum_np[:500, feature_idx], label='Predicted Sum', marker='o')
    plt.plot(label_data_np[:500, feature_idx], label='True', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Predicted vs True Values (Output Pred Sum Feature {feature_idx + 1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(pred_path + f'pred_vs_true_output_pred_sum_feature{feature_idx + 1}.png')
    plt.close()

print("finish {}.{}.{}.{}.{}".format(layer, width, max_epochs, batchSize, learning_rate))

