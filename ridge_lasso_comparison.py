# ============================================
# IMPORTS
# ============================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# ============================================
# DATA GENERATION
# ============================================
# Set random seed for reproducibility
np.random.seed(42)
n = 100  # Number of samples

# Generate random X values between -3 and 3
X = np.random.uniform(-3, 3, n).reshape(-1, 1)

# True function: cubic with noise 
# y = 1 + 2x + 0.5x² - 0.3x³ + Gaussian noise
y = (1 + 2*X.ravel() + 0.5*X.ravel()**2 - 0.3*X.ravel()**3 
     + np.random.normal(0, 0.5, n))

# ============================================
# TRAIN TEST SPLIT
# ============================================
# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================
# POLYNOMIAL FEATURES
# ============================================
# Create polynomial features up to degree 15 (causes overfitting)
poly = PolynomialFeatures(degree=15, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# ============================================
# STANDARDIZE FEATURES (for better numerical stability)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# ============================================
# HYPERPARAMETERS
# ============================================
# Alpha values to test for Ridge and Lasso
alpha_values_ridge = [0.001, 0.01, 0.1, 1, 10, 100]
alpha_values_lasso = [0.001, 0.01, 0.1, 1, 10]

# ============================================
# RIDGE REGRESSION (L2 Regularization)
# ============================================
print("="*70)
print("RIDGE REGRESSION (L2 Regularization)")
print("="*70)
print(f"{'Alpha':<10} {'Train MSE':<18} {'Test MSE':<18} {'Test R2':<12}")
print("-"*70)

# Store results for plotting
ridge_train_errors = []
ridge_test_errors = []
ridge_r2_scores = []
ridge_best_alpha = None
ridge_best_mse = float('inf')
ridge_best_r2 = 0

for alpha in alpha_values_ridge:
    # Create and train Ridge model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Store results
    ridge_train_errors.append(train_mse)
    ridge_test_errors.append(test_mse)
    ridge_r2_scores.append(r2)
    
    # Track best model
    if test_mse < ridge_best_mse:
        ridge_best_mse = test_mse
        ridge_best_r2 = r2
        ridge_best_alpha = alpha
    
    # Print results
    print(f"{alpha:<10} {train_mse:<18.6f} {test_mse:<18.6f} {r2:<12.4f}")

print("-"*70)
print(f"\nBEST RIDGE MODEL:")
print(f"   Alpha = {ridge_best_alpha}")
print(f"   Test MSE = {ridge_best_mse:.6f}")
print(f"   Test R² = {ridge_best_r2:.4f}")

# ============================================
# LASSO REGRESSION (L1 Regularization)
# ============================================
print("\n" + "="*70)
print("LASSO REGRESSION (L1 Regularization)")
print("="*70)
print(f"{'Alpha':<10} {'Train MSE':<18} {'Test MSE':<18} {'Test R2':<12}")
print("-"*70)

# Store results for plotting
lasso_train_errors = []
lasso_test_errors = []
lasso_r2_scores = []
lasso_best_alpha = None
lasso_best_mse = float('inf')
lasso_best_r2 = 0

for alpha in alpha_values_lasso:
    # Create and train Lasso model with more iterations
    lasso = Lasso(alpha=alpha, max_iter=50000)
    lasso.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = lasso.predict(X_train_scaled)
    y_test_pred = lasso.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Store results
    lasso_train_errors.append(train_mse)
    lasso_test_errors.append(test_mse)
    lasso_r2_scores.append(r2)
    
    # Track best model
    if test_mse < lasso_best_mse:
        lasso_best_mse = test_mse
        lasso_best_r2 = r2
        lasso_best_alpha = alpha
    
    # Print results
    print(f"{alpha:<10} {train_mse:<18.6f} {test_mse:<18.6f} {r2:<12.4f}")

print("-"*70)
print(f"\nBEST LASSO MODEL:")
print(f"   Alpha = {lasso_best_alpha}")
print(f"   Test MSE = {lasso_best_mse:.6f}")
print(f"   Test R² = {lasso_best_r2:.4f}")

# ============================================
# LINEAR REGRESSION (No Regularization)
# ============================================
print("\n" + "="*70)
print("LINEAR REGRESSION (No Regularization - Baseline)")
print("="*70)

# Train standard linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_lr = lr.predict(X_train_scaled)
y_test_pred_lr = lr.predict(X_test_scaled)

# Calculate metrics
lr_train_mse = mean_squared_error(y_train, y_train_pred_lr)
lr_test_mse = mean_squared_error(y_test, y_test_pred_lr)
lr_r2 = r2_score(y_test, y_test_pred_lr)

print(f"Train MSE: {lr_train_mse:.6f}")
print(f"Test MSE: {lr_test_mse:.6f}")
print(f"Test R²: {lr_r2:.4f}")

# ============================================
# VISUALIZATION
# ============================================
# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: MSE vs Alpha (Regularization Strength)
axes[0].plot(alpha_values_ridge, ridge_train_errors, 'o-', 
             label='Ridge Train MSE', color='blue', linewidth=2, markersize=8)
axes[0].plot(alpha_values_ridge, ridge_test_errors, 's-', 
             label='Ridge Test MSE', color='darkblue', linewidth=2, markersize=8)
axes[0].plot(alpha_values_lasso, lasso_train_errors, '^-', 
             label='Lasso Train MSE', color='red', linewidth=2, markersize=8)
axes[0].plot(alpha_values_lasso, lasso_test_errors, 'd-', 
             label='Lasso Test MSE', color='darkred', linewidth=2, markersize=8)

# Add baseline linear regression line
axes[0].axhline(y=lr_test_mse, color='gray', linestyle='--', 
                label=f'Linear Test MSE = {lr_test_mse:.3f}', alpha=0.7)

axes[0].set_xscale('log')  # Log scale for alpha
axes[0].set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
axes[0].set_title('MSE vs Regularization Strength', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: R² Score vs Alpha
axes[1].plot(alpha_values_ridge, ridge_r2_scores, 'o-', 
             label='Ridge R² Score', color='green', linewidth=2, markersize=8)
axes[1].plot(alpha_values_lasso, lasso_r2_scores, 's-', 
             label='Lasso R² Score', color='darkgreen', linewidth=2, markersize=8)

# Add baseline linear regression R² line
axes[1].axhline(y=lr_r2, color='gray', linestyle='--', 
                label=f'Linear R² = {lr_r2:.3f}', alpha=0.7)

axes[1].set_xscale('log')  # Log scale for alpha
axes[1].set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('R² Score (Coefficient of Determination)', fontsize=12, fontweight='bold')
axes[1].set_title('R² Score vs Regularization Strength', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()

# ============================================
# FEATURE ANALYSIS (Lasso Feature Selection)
# ============================================
# Train best Lasso model on all training data
best_lasso = Lasso(alpha=lasso_best_alpha, max_iter=50000)
best_lasso.fit(X_train_scaled, y_train)

# Count how many features Lasso kept (non-zero coefficients)
non_zero_coeffs = np.sum(np.abs(best_lasso.coef_) > 1e-6)
total_features = X_train_scaled.shape[1]

print("\n" + "="*70)
print("LASSO FEATURE SELECTION ANALYSIS")
print("="*70)
print(f"Total features (degree 15 polynomial): {total_features}")
print(f"Features kept by Lasso (non-zero): {non_zero_coeffs}")
print(f"Features removed by Lasso: {total_features - non_zero_coeffs}")
print(f"Feature reduction: {(total_features - non_zero_coeffs)/total_features*100:.1f}%")

# ============================================
# FINAL COMPARISON SUMMARY
# ============================================
print("\n" + "="*70)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*70)
print(f"{'Model':<30} {'Best Alpha':<15} {'Test MSE':<15} {'Test R²':<12}")
print("-"*70)
print(f"{'Linear Regression (No Reg)':<30} {'-':<15} {lr_test_mse:<15.6f} {lr_r2:<12.4f}")
print(f"{'Ridge Regression (L2)':<30} {ridge_best_alpha:<15} {ridge_best_mse:<15.6f} {ridge_best_r2:<12.4f}")
print(f"{'Lasso Regression (L1)':<30} {lasso_best_alpha:<15} {lasso_best_mse:<15.6f} {lasso_best_r2:<12.4f}")
print("="*70)

# ============================================
# CONCLUSION
# ============================================
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if lasso_best_r2 > ridge_best_r2 and lasso_best_r2 > lr_r2:
    print("BEST MODEL: LASSO REGRESSION")
    print(f"   - Achieved {lasso_best_r2*100:.1f}% R² score (best among all models)")
    print(f"   - Reduced features by {int((total_features - non_zero_coeffs)/total_features*100)}%")
    print("   - L1 regularization automatically selects important features")
    print("   - Ideal for high-dimensional data with irrelevant features")
elif ridge_best_r2 > lasso_best_r2 and ridge_best_r2 > lr_r2:
    print("BEST MODEL: RIDGE REGRESSION")
else:
    print("BASELINE: LINEAR REGRESSION (No improvement from regularization)")

print("="*70)