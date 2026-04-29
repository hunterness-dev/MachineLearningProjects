# ============================================
# IMPORT LIBRARIES
# ============================================
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# LOAD AND PREPARE DATA
# ============================================
# Load diabetes dataset
data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1)  # Use BMI feature only (column index 2)
y = data.target  # Disease progression target

print("="*60)
print("DIABETES DATASET - BMI vs DISEASE PROGRESSION")
print("="*60)
print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
print()

# ============================================
# POLYNOMIAL DEGREE SELECTION WITH CROSS-VALIDATION
# ============================================
# Test polynomial degrees from 1 to 10
degrees = range(1, 11)
cv_scores = []      # Mean RMSE for each degree
cv_stds = []        # Standard deviation for each degree

print("CROSS-VALIDATION RESULTS (5-Fold)")
print("-"*60)
print(f"{'Degree':<8} {'Mean RMSE':<15} {'Std Dev':<15} {'95% CI':<20}")
print("-"*60)

for d in degrees:
    # Create pipeline: PolynomialFeatures -> LinearRegression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    
    # Perform 5-fold cross-validation using negative MSE
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    
    # Convert MSE to RMSE
    rmse_scores = np.sqrt(-scores)
    
    # Store results
    cv_scores.append(rmse_scores.mean())
    cv_stds.append(rmse_scores.std())
    
    # Calculate 95% confidence interval
    ci_lower = rmse_scores.mean() - 1.96 * rmse_scores.std() / np.sqrt(5)
    ci_upper = rmse_scores.mean() + 1.96 * rmse_scores.std() / np.sqrt(5)
    
    print(f"{d:<8} {rmse_scores.mean():<15.2f} {rmse_scores.std():<15.2f} "
          f"[{ci_lower:.2f}, {ci_upper:.2f}]")

# ============================================
# FIND BEST DEGREE
# ============================================
best_degree = degrees[np.argmin(cv_scores)]
best_rmse = min(cv_scores)

print("-"*60)
print(f"\nBEST POLYNOMIAL DEGREE: {best_degree}")
print(f"Best RMSE: {best_rmse:.2f}")
print()

# ============================================
# CHECK FOR OVERFITTING
# ============================================
print("="*60)
print("OVERFITTING ANALYSIS")
print("="*60)

# Compare low degree vs high degree
low_degree_rmse = cv_scores[0]  # Degree 1 (linear)
high_degree_rmse = cv_scores[-1]  # Degree 10

if high_degree_rmse > low_degree_rmse * 1.1:
    print(f"High degree (10) performs worse than linear model")
    print(f"   Degree 1 RMSE: {low_degree_rmse:.2f}")
    print(f"   Degree 10 RMSE: {high_degree_rmse:.2f}")
    print(f"   Higher degrees cause OVERFITTING")
else:
    print(f"Higher degrees may be beneficial")
    print(f"   Degree 1 RMSE: {low_degree_rmse:.2f}")
    print(f"   Degree {best_degree} RMSE: {best_rmse:.2f}")

# ============================================
# TRAIN BEST MODEL ON ALL DATA
# ============================================
print("\n" + "="*60)
print("TRAINING BEST MODEL")
print("="*60)

# Create best model pipeline
best_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

# Train on all data
best_model.fit(X, y)

# Get coefficients
coefficients = best_model.named_steps['lr'].coef_
n_features = len(coefficients)

print(f"Model trained with degree {best_degree}")
print(f"Number of features (including polynomial terms): {n_features}")
print(f"First 5 coefficients: {coefficients[:5]}")

# ============================================
# VISUALIZATION
# ============================================
# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: RMSE vs Polynomial Degree (with error bars)
axes[0].errorbar(degrees, cv_scores, yerr=cv_stds, 
                 fmt='o-', capsize=5, capthick=2, 
                 color='blue', markersize=8, linewidth=2,
                 elinewidth=2, markeredgecolor='darkblue')
axes[0].axvline(x=best_degree, color='red', linestyle='--', 
                linewidth=2, label=f'Best Degree = {best_degree}')
axes[0].set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE (Root Mean Square Error)', fontsize=12, fontweight='bold')
axes[0].set_title('Cross-Validation: RMSE vs Degree', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(degrees)

# Plot 2: Scatter plot with best model prediction
X_sorted = np.sort(X, axis=0)
y_pred = best_model.predict(X_sorted)

axes[1].scatter(X, y, alpha=0.5, color='green', label='Actual Data', s=30)
axes[1].plot(X_sorted, y_pred, color='red', linewidth=2, 
             label=f'Polynomial Degree {best_degree}')
axes[1].set_xlabel('BMI', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Disease Progression', fontsize=12, fontweight='bold')
axes[1].set_title(f'Best Model Fit (Degree = {best_degree})', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# COMPARE WITH/WITHOUT REGULARIZATION
# ============================================
print("\n" + "="*60)
print("COMPARISON: Linear vs Ridge vs Lasso (at best degree)")
print("="*60)

# Ridge Regression with cross-validation
ridge_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5))
])

ridge_scores = cross_val_score(ridge_model, X, y, cv=5, 
                                scoring='neg_mean_squared_error')
ridge_rmse = np.sqrt(-ridge_scores).mean()

# Lasso Regression with cross-validation
lasso_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lasso', LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=50000))
])

lasso_scores = cross_val_score(lasso_model, X, y, cv=5,
                                scoring='neg_mean_squared_error')
lasso_rmse = np.sqrt(-lasso_scores).mean()

# Display comparison
print(f"{'Model':<20} {'RMSE':<12} {'Better than Linear?':<20}")
print("-"*60)
print(f"{'Linear (Degree 1)':<20} {cv_scores[0]:<12.2f} {'-':<20}")
print(f"{'Polynomial (Degree '+str(best_degree)+')':<20} {best_rmse:<12.2f} "
      f"{'Yes' if best_rmse < cv_scores[0] else 'No':<20}")
print(f"{'Polynomial + Ridge':<20} {ridge_rmse:<12.2f} "
      f"{'Yes' if ridge_rmse < cv_scores[0] else 'No':<20}")
print(f"{'Polynomial + Lasso':<20} {lasso_rmse:<12.2f} "
      f"{'Yes' if lasso_rmse < cv_scores[0] else 'No':<20}")

# ============================================
# RECOMMENDATION
# ============================================
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if best_degree == 1:
    print("Use simple LINEAR regression (degree 1 is best)")
    print("The relationship between BMI and disease progression is linear")
elif best_degree <= 3:
    print(f"Use degree {best_degree} polynomial regression")
    print("Some curvature detected, but low complexity")
else:
    print(f"Be careful with degree {best_degree}")
    print("Higher risk of overfitting on new data")

print("\nTIP: Always use cross-validation to select polynomial degree")
print("="*60)