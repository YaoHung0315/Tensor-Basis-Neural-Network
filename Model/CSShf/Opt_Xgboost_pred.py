import os
import glob
import joblib
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata

# Function to get next version number for saving plot images
def get_next_version(output_dir, prefix, ext):
    pattern = os.path.join(output_dir, f"{prefix}_*{ext}")
    files = glob.glob(pattern)
    if files:
        versions = [int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) for f in files]
        next_version = max(versions) + 1
    else:
        next_version = 1
    return next_version

# Set output directory (make sure it exists)
output_dir = "output_opt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the saved model and scaler
model_path = os.path.join(output_dir, "xgb_model.pkl")
scaler_path = os.path.join(output_dir, "scaler.pkl")
xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load new data from the .mat file (assuming same data names)
data = sio.loadmat('./data/ML_Optimal_method.mat')
eddy_double_LES = data['eddy_double_LES']
eddy_double_TBNN = data['eddy_double_TBNN']
Sij = data['Sij']
Rij = data['Rij']
x = data['x']
y = data['y']
x_point = data['x_point']
y_point = data['y_point']
bij_LES = data['bij_LES']
bij_TBNN = data['bij_TBNN']

# Prepare features and target variables
# Here we use columns 0, 1, 3, 4 from Sij as features
X = np.concatenate((Sij[:, [0, 1, 3, 4]], bij_TBNN[:, [0, 1, 3, 4]]), axis=1)
# X = np.concatenate((Sij[:, [0, 1, 3, 4]], Rij[:, [0, 1, 3, 4]]), axis=1)
# X = Sij[:, [0, 1, 3, 4]]
Y = eddy_double_TBNN.ravel()      # Actual target (for evaluation)
Y_LES = eddy_double_LES.ravel()     # Reference target

# Apply the saved scaler to the features
X_scaled = scaler.transform(X)

# Predict using the loaded XGBoost model
Y_pred = xgb_model.predict(X_scaled)

# Evaluate prediction performance (optional)
rmse = np.sqrt(np.mean((Y - Y_pred)**2))
r2 = r2_score(Y, Y_pred)
print("Loaded XGBoost Model - Training RMSE:", rmse)
print("Loaded XGBoost Model - Training R2:", r2)

# Plot predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(Y_LES, label='Actual eddy_double_LES', color='green')
plt.plot(Y, label='Actual eddy_double_TBNN', color='blue')
plt.plot(Y_pred, label='XGB Prediction', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('XGBoost Prediction vs. Actual')
plt.legend()
plt.grid(True)

# Get the next version number for the plot filename
version = get_next_version(output_dir, "xgb_predictions", ".png")
plot_filename = os.path.join(output_dir, f"xgb_predictions_{version:03d}.png")
plt.savefig(plot_filename, dpi=300)
plt.close()
print("Prediction plot saved as:", plot_filename)

# Interpolate the results onto the spatial grid
points = np.column_stack((x_point.ravel(), y_point.ravel()))
eddy_double_LES_interp = griddata(points, eddy_double_LES.ravel(), (x, y), method='cubic').reshape(x.shape)
eddy_double_TBNN_interp = griddata(points, eddy_double_TBNN.ravel(), (x, y), method='cubic').reshape(x.shape)
Y_pred_interp = griddata(points, Y_pred.ravel(), (x, y), method='cubic').reshape(x.shape)

# Set common color scale limits for the spatial plots
vmin = -0.05
vmax = 0.20

# Plot the interpolated maps in a vertical stack (3 subplots)
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
im0 = axs[0].pcolormesh(x, y, eddy_double_LES_interp, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
axs[0].set_title('eddy double LES')
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlim(np.min(x), np.max(x))
axs[0].set_ylim(np.min(y), np.max(y))

im1 = axs[1].pcolormesh(x, y, eddy_double_TBNN_interp, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
axs[1].set_title('eddy double TBNN')
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlim(np.min(x), np.max(x))
axs[1].set_ylim(np.min(y), np.max(y))

im2 = axs[2].pcolormesh(x, y, Y_pred_interp, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
axs[2].set_title('XGB Y pred')
axs[2].set_aspect('equal', adjustable='box')
axs[2].set_xlim(np.min(x), np.max(x))
axs[2].set_ylim(np.min(y), np.max(y))

fig.subplots_adjust(right=0.85, hspace=0.1)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax)

# Get next version number for the spatial plot
spatial_version = get_next_version(output_dir, "xgb_spatial", ".png")
spatial_plot_filename = os.path.join(output_dir, f"xgb_spatial_{spatial_version:03d}.png")
plt.savefig(spatial_plot_filename, dpi=300)
plt.close()
print("Spatial interpolation plot saved as:", spatial_plot_filename)


# Save Y_pred values as a CSV file (overwrites previous file)
y_pred_csv_filename = os.path.join(output_dir, "Y_pred.csv")
np.savetxt(y_pred_csv_filename, Y_pred, delimiter=",")
print("Y_pred values saved as:", y_pred_csv_filename)