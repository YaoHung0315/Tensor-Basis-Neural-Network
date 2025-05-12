import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
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

# Create polynomial features
poly = PolynomialFeatures(degree=4, include_bias=True)
X_poly = poly.fit_transform(X_scaled)

# Initialize a Linear Regression model
lin_reg = LinearRegression()

# Evaluate the model using 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lin_reg, X_poly, Y, cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-np.mean(cv_scores))
print("Cross-validated RMSE (Polynomial degree 4):", cv_rmse)
Y_pred_cv = cross_val_predict(lin_reg, X_poly, Y, cv=kf)
print("Cross-validated R2 (Polynomial degree 4):", r2_score(Y, Y_pred_cv))

# Train the final model on the full dataset using polynomial features
lin_reg.fit(X_poly, Y)
Y_pred = lin_reg.predict(X_poly)
print("Training RMSE (Polynomial degree 4):", np.sqrt(mean_squared_error(Y, Y_pred)))
print("Training R2 (Polynomial degree 4):", r2_score(Y, Y_pred))

# Print the regression coefficients and intercept
print("Regression coefficients:", lin_reg.coef_)
print("Regression intercept:", lin_reg.intercept_)

# Plot predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(Y_LES, label='Actual eddy_double_LES', color='green')
plt.plot(Y, label='Actual eddy_double_TBNN', color='blue')
plt.plot(Y_pred, label='Poly Regression Prediction (Train)', color='red', linestyle='--')
plt.plot(Y_pred_cv, label='Poly Regression Prediction (CV)', color='orange', linestyle='-.')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Polynomial Regression (degree 4) Prediction vs. Actual')
plt.legend()
plt.grid(True)
pred_plot_filename = os.path.join(output_dir, "poly_regression_predictions.png")
plt.savefig(pred_plot_filename, dpi=300)
plt.close()

# Save the trained model, scaler, and polynomial transformer for future predictions
model_filename = os.path.join(output_dir, "poly_linear_regression_model.pkl")
scaler_filename = os.path.join(output_dir, "scaler.pkl")
poly_filename = os.path.join(output_dir, "poly_transformer.pkl")
joblib.dump(lin_reg, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(poly, poly_filename)
print("Model saved as:", model_filename)
print("Scaler saved as:", scaler_filename)
print("Polynomial transformer saved as:", poly_filename)

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
axs[2].set_title('Poly Regression Y pred')
axs[2].set_aspect('equal', adjustable='box')
axs[2].set_xlim(np.min(x), np.max(x))
axs[2].set_ylim(np.min(y), np.max(y))

fig.subplots_adjust(right=0.85, hspace=0.1)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax)
spatial_plot_filename = os.path.join(output_dir, "poly_regression_spatial.png")
plt.savefig(spatial_plot_filename, dpi=300)
plt.close()
print("Spatial interpolation plot saved as:", spatial_plot_filename)