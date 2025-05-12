import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import griddata
import joblib

# Create output folder if not exists
output_dir = "output_opt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data from the .mat file
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
X = np.concatenate((Sij[:, [0, 1, 3, 4]], bij_TBNN[:, [0, 1, 3, 4]]), axis=1)
# X = np.concatenate((Sij[:, [0, 1, 3, 4]], Rij[:, [0, 1, 3, 4]]), axis=1)
# X = Sij[:, [0, 1, 3, 4]]
Y = eddy_double_TBNN.ravel()
Y_LES = eddy_double_LES.ravel()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Grid search for Random Forest hyperparameters using 5-fold cross-validation
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_scaled, Y)
print("Best parameters from GridSearch:", grid_search.best_params_)
print("Best RMSE from GridSearch:", np.sqrt(-grid_search.best_score_))

# Create the final Random Forest model using the best parameters from grid search
best_params = grid_search.best_params_
rf_model = RandomForestRegressor(random_state=42, **best_params)

# Evaluate the model using 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_scaled, Y, cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-np.mean(cv_scores))
print("Cross-validated RMSE:", cv_rmse)
Y_pred_cv = cross_val_predict(rf_model, X_scaled, Y, cv=kf)
print("Cross-validated R2:", r2_score(Y, Y_pred_cv))

# Train the final model on the full dataset
rf_model.fit(X_scaled, Y)
Y_pred = rf_model.predict(X_scaled)
print("Training RMSE:", np.sqrt(mean_squared_error(Y, Y_pred)))
print("Training R2:", r2_score(Y, Y_pred))

# Plot predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(Y_LES, label='Actual eddy_double_LES', color='green')
plt.plot(Y, label='Actual eddy_double_TBNN', color='blue')
plt.plot(Y_pred, label='RF Prediction (Train)', color='red', linestyle='--')
plt.plot(Y_pred_cv, label='RF Prediction (CV)', color='orange', linestyle='-.')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Random Forest Prediction vs. Actual')
plt.legend()
plt.grid(True)
# Save the prediction plot
pred_plot_filename = os.path.join(output_dir, "rf_predictions.png")
plt.savefig(pred_plot_filename, dpi=300)
plt.close()

# Save the trained model and the scaler for future predictions
model_filename = os.path.join(output_dir, "rf_model.pkl")
scaler_filename = os.path.join(output_dir, "scaler.pkl")
joblib.dump(rf_model, model_filename)
joblib.dump(scaler, scaler_filename)
print("Model saved as:", model_filename)
print("Scaler saved as:", scaler_filename)

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
axs[2].set_title('RF Y pred')
axs[2].set_aspect('equal', adjustable='box')
axs[2].set_xlim(np.min(x), np.max(x))
axs[2].set_ylim(np.min(y), np.max(y))

fig.subplots_adjust(right=0.85, hspace=0.1)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax)
# Save the spatial interpolation plot
spatial_plot_filename = os.path.join(output_dir, "rf_spatial.png")
plt.savefig(spatial_plot_filename, dpi=300)
plt.close()
print("Spatial interpolation plot saved as:", spatial_plot_filename)