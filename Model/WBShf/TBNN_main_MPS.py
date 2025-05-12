import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import gc
from network import Net


# Select device: Use MPS if available, otherwise use CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

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

# Calculate Mean Absolute Error and its standard deviation
def calculate_mae_and_std(errors):
    errors = errors.detach()
    mae = torch.mean(torch.abs(errors))
    mae_std = torch.std(torch.abs(errors))
    return mae.item(), mae_std.item()

# Calculate Root Mean Squared Error
def calculate_rmse(errors):
    errors = errors.detach()
    mse = torch.mean(errors ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

# Calculate R-squared value
def calculate_r2(outputs, targets):
    outputs = outputs.detach()
    targets = targets.detach()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    ss_res = torch.sum((targets - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def custom_loss(predict_transport, label_transport, criterion, weight_for_l1loss):
    # Compute individual losses and stack them into a tensor
    losses = torch.stack([criterion(predict_transport[i], label_transport[i]) for i in range(len(weight_for_l1loss))])
    # Multiply losses by weights and compute the weighted average
    return torch.sum(losses * weight_for_l1loss) / len(weight_for_l1loss)


# Training and validation function for hyperparameter search
def train_and_validate_find_hyperparameters(train_loader, val_loader, net, criterion, optimizer, max_epochs, weight_for_l1loss):
    net.train()
    
    # Train for 25% of max_epochs (minimum 100 epochs)
    epochs_to_train = max(100, int(max_epochs * 0.25))
    
    for epoch in range(epochs_to_train):
        for i, data in enumerate(train_loader, 0):
            lambda_data = data['lambda_data'].to(device)
            T_matrix_data = data['T_matrix_data'].to(device)
            anisotropic_tensor_data = data['anisotropic_tensor_data'].to(device)

            # Use lambda_data as input and anisotropic_tensor_data as label
            inputs = lambda_data
            labels = anisotropic_tensor_data
            output = net(inputs).unsqueeze(1)
            input_tensor = T_matrix_data.permute(0, 2, 1)
            predict = output.matmul(input_tensor).reshape(-1, 9)
               
            predict_transport = torch.t(predict)
            label_transport = torch.t(labels)

            weighted_loss_all = custom_loss(predict_transport, label_transport, criterion, weight_for_l1loss)

            optimizer.zero_grad()
            weighted_loss_all.backward()
            optimizer.step()

        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
                lambda_data = data['lambda_data'].to(device)
                T_matrix_data = data['T_matrix_data'].to(device)
                anisotropic_tensor_data = data['anisotropic_tensor_data'].to(device)

                inputs = lambda_data
                labels = anisotropic_tensor_data
                output = net(inputs).unsqueeze(1)
                input_tensor = T_matrix_data.permute(0, 2, 1)

                predict = output.matmul(input_tensor).reshape(-1, 9)
                predict_transport = torch.t(predict)
                label_transport = torch.t(labels)

                # Accumulate validation loss for each batch
                total_val_loss += custom_loss(predict_transport, label_transport, criterion, weight_for_l1loss)

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss


# Training and validation function for K-fold cross-validation
def train_and_validate_k_fold(train_loader, val_loader, net, criterion, optimizer, max_epochs, patience, model_save_path, weight_for_l1loss, fold_num):
    loss_min = float('inf')
    error_of_results = []
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        net.train()
        losses_train = []
        outputs_all = []
        labels_all = []

        for i, data in enumerate(train_loader, 0):
            lambda_data = data['lambda_data'].to(device)
            T_matrix_data = data['T_matrix_data'].to(device)
            anisotropic_tensor_data = data['anisotropic_tensor_data'].to(device)
    
            # Use lambda_data as input and anisotropic_tensor_data as label
            inputs = lambda_data
            labels = anisotropic_tensor_data
            output = net(inputs).unsqueeze(1)
            input_tensor = T_matrix_data.permute(0, 2, 1)

            predict = output.matmul(input_tensor).reshape(-1, 9)
            predict_transport = torch.t(predict)
            label_transport = torch.t(labels)

            weighted_loss_all = custom_loss(predict_transport, label_transport, criterion, weight_for_l1loss)

            optimizer.zero_grad()
            weighted_loss_all.backward()
            optimizer.step()
            losses_train.append(weighted_loss_all.detach())
            outputs_all.append(predict_transport)
            labels_all.append(label_transport)

        outputs_all = torch.cat(outputs_all, dim=1).to(device)
        labels_all = torch.cat(labels_all, dim=1).to(device)

        loss, loss_STD = calculate_mae_and_std(torch.tensor(losses_train))
        rmse = calculate_rmse(outputs_all - labels_all)
        r2 = calculate_r2(outputs_all, labels_all)
        print(f'training: [{epoch+1} / {max_epochs}] Loss: {loss:.4f} Loss_STD: {loss_STD:.4f} RMSE: {rmse:.4f} R²: {r2:.4f}')

        net.eval()
        losses_val = []
        outputs_val_all = []
        labels_val_all = []
        with torch.no_grad():
            for validation_loader in [val_loader]:
                for j, data in enumerate(validation_loader, 0):
                    lambda_data = data['lambda_data'].to(device)
                    T_matrix_data = data['T_matrix_data'].to(device)
                    anisotropic_tensor_data = data['anisotropic_tensor_data'].to(device)
    
                    inputs = lambda_data
                    labels = anisotropic_tensor_data
                    output = net(inputs).unsqueeze(1)
                    input_tensor = T_matrix_data.permute(0, 2, 1)

                    predict = output.matmul(input_tensor).reshape(-1, 9)
                    predict_transport = torch.t(predict)
                    label_transport = torch.t(labels)

                    val_weighted_loss_all = custom_loss(predict_transport, label_transport, criterion, weight_for_l1loss)

                    losses_val.append(val_weighted_loss_all.detach())
                    outputs_val_all.append(predict_transport)
                    labels_val_all.append(label_transport)

        outputs_val_all = torch.cat(outputs_val_all, dim=1).to(device)
        labels_val_all = torch.cat(labels_val_all, dim=1).to(device)

        loss_val, loss_std_val = calculate_mae_and_std(torch.tensor(losses_val))
        rmse_val = calculate_rmse(outputs_val_all - labels_val_all)
        r2_val = calculate_r2(outputs_val_all, labels_val_all)
        print(f'validation: [{epoch+1} / {max_epochs}] Loss: {loss_val:.4f} Loss_STD: {loss_std_val:.4f} RMSE: {rmse_val:.4f} R²: {r2_val:.4f}')

        if loss_min > loss_val:
            torch.save(net.state_dict(), model_save_path)
            print("model update at {}".format(model_save_path))
            loss_min = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        error_of_results.append([epoch, loss, loss_STD, loss_val, loss_std_val, rmse, rmse_val, r2, r2_val])

        if math.isnan(loss_val):
            print('break epoch-forloop! validation loss : {:.4f}'.format(loss_val))
            break

    plot_training_validation_metrics(error_of_results, fold_num)
    return error_of_results


# Visualization function to plot training and validation metrics
def plot_training_validation_metrics(error_of_results, fold_num):
    error_of_results = np.array(error_of_results)
    epochs = error_of_results[:, 0]
    train_loss = error_of_results[:, 1]
    val_loss = error_of_results[:, 3]
    train_rmse = error_of_results[:, 5]
    val_rmse = error_of_results[:, 6]
    train_r2 = error_of_results[:, 7]
    val_r2 = error_of_results[:, 8]

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Fold {fold_num})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}training_validation_loss_fold{fold_num}.png')
    # plt.close()

    # Plot RMSE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_rmse, label='Train RMSE', marker='o')
    plt.plot(epochs, val_rmse, label='Validation RMSE', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'Training and Validation RMSE (Fold {fold_num})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}training_validation_rmse_fold{fold_num}.png')
    # plt.close()

    # Plot R²
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_r2, label='Train R²', marker='o')
    plt.plot(epochs, val_r2, label='Validation R²', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title(f'Training and Validation R² (Fold {fold_num})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}training_validation_r2_fold{fold_num}.png')
    # plt.close()


# Initialize dataset and parameters
train_mat_path = './data/ML_TBNN_training.mat'

# Output directory for results
output_path = "./output/"
os.makedirs(output_path, exist_ok=True)

num_lambda = 7
num_t = 10
n_trials = 50
k_fold = 20
patience = 50
weight_for_l1loss = torch.tensor([1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32, device=device)
train_dataset = turDataset(mat_path=train_mat_path, num_lambda=num_lambda, num_t=num_t)


# Hyperparameter search using Optuna
def objective(trial, num_lambda=num_lambda, num_t=num_t):
    width = trial.suggest_int('width', 50, 200)
    layer = trial.suggest_int('layer', 5, 10)
    max_epochs = trial.suggest_int('max_epochs', 100, 1000)
    batchSize = trial.suggest_int('batchSize', 16, 512)
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    optimizer_class = getattr(optim, optimizer_name)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    val_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)

    net = Net(width, layer, num_lambda, num_t).to(device)
    optimizer = optimizer_class(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_loss = train_and_validate_find_hyperparameters(train_loader, val_loader, net, criterion, optimizer, max_epochs, weight_for_l1loss)
    
    del net, optimizer
    gc.collect()

    return best_loss


# Create and optimize the study
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, num_lambda=num_lambda, num_t=num_t), n_trials=n_trials)

# Get the best hyperparameters
best_params = study.best_params
print("Best params:", best_params)

# Optionally, plot optimization history and parameter importances
# fig1 = plot_optimization_history(study)
# fig1.savefig(f"{output_path}/optuna_optimization_history.png")

# fig2 = plot_param_importances(study)
# fig2.savefig(f"{output_path}/optuna_param_importance.png")


width = best_params['width']
layer = best_params['layer']
max_epochs = best_params['max_epochs']
batchSize = best_params['batchSize']
learning_rate = best_params['lr']
optimizer_name = best_params['optimizer']
optimizer_class = getattr(optim, optimizer_name)

# K-fold cross-validation
kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
fold_results = []

for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batchSize, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batchSize, shuffle=False, num_workers=0)
    
    net = Net(width, layer, num_lambda, num_t).to(device)
    optimizer = optimizer_class(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    fold_model_save_path = f'{output_path}model_fold{fold+1}.pkl'
    fold_result = train_and_validate_k_fold(train_loader, val_loader, net, criterion, optimizer, max_epochs, patience, fold_model_save_path, weight_for_l1loss, fold+1)
    fold_results.append(fold_result)
    
    del net, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Save results for each fold
    error_path = output_path + f"data_of_error_fold{fold+1}.csv"
    fold_results_flat = np.array(fold_result)  # Convert results to numpy array
    fold_results_df = pd.DataFrame(fold_results_flat, columns=["Epoch", "Train_Loss", "Train_Loss_STD", "Val_Loss", "Val_Loss_STD", "Train_RMSE", "Val_RMSE", "Train_R2", "Val_R2"])
    fold_results_df.to_csv(error_path, index=False)

    # Save hyperparameters
    hyper_parameters_path = output_path + "hyper_parameters.csv"
    parameters_results_flat = np.array([width, layer, max_epochs, batchSize, learning_rate, optimizer_name, patience, k_fold, num_lambda, num_t])
    parameters_results_df = pd.DataFrame([parameters_results_flat], columns=["width", "layer", "max_epochs", "batchSize", "learning_rate", "optimizer_name", "patience", "k_fold", "num_lambda", "num_t"])
    parameters_results_df.to_csv(hyper_parameters_path, index=False)

