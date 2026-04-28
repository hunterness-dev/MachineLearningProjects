import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Load data
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

features = ['age', 'bmi', 'bp']
X = df[features]
y = df['target']

# Model with standardized data (for standardized coefficients)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y_scaled)

# Model with raw data (for raw coefficients)
model_raw = LinearRegression()
model_raw.fit(X, y)

# Calculate real-world interpretation
std_real = X.std()
effect_per_std = model_scaled.coef_ * scaler_y.scale_[0]  # Effect of 1 std change on original target

# Create table
results = pd.DataFrame({
    'Feature': features,
    'Std Coef': model_scaled.coef_,
    'Raw Coef': model_raw.coef_,
    '1 Real Std': std_real.values,
    'Effect of +1 Unit': model_raw.coef_,
    'Effect of +1 Std': effect_per_std
})

print("\n" + "="*80)
print("Diabetes Model - Coefficient Interpretation Table")
print("="*80)
print(results.to_string(index=False))
print("\n" + "="*80)

# Practical example
print("\nExample: If BMI increases by 6 units (which is 1 standard deviation)")
print(f"    => Disease progression increases by {effect_per_std[1]:.1f} units")

# ========== PLOTTING SECTION ==========
# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Actual vs Predicted Plot (Raw Model)
y_pred_raw = model_raw.predict(X)
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(y, y_pred_raw, alpha=0.5, color='steelblue', edgecolors='black', linewidth=0.5)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Disease Progression', fontsize=11)
ax1.set_ylabel('Predicted Disease Progression', fontsize=11)
ax1.set_title('Actual vs Predicted (Raw Model)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residual Plot
residuals = y - y_pred_raw
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(y_pred_raw, residuals, alpha=0.5, color='coral', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Values', fontsize=11)
ax2.set_ylabel('Residuals', fontsize=11)
ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Coefficients Comparison (Standardized vs Raw)
x_pos = np.arange(len(features))
width = 0.35
ax3 = fig.add_subplot(2, 3, 3)
ax3.bar(x_pos - width/2, model_scaled.coef_, width, label='Standardized Coef', color='forestgreen', alpha=0.7)
ax3.bar(x_pos + width/2, model_raw.coef_ / np.abs(model_raw.coef_).max(), width, label='Raw Coef (normalized)', color='goldenrod', alpha=0.7)
ax3.set_xlabel('Features', fontsize=11)
ax3.set_ylabel('Coefficient Value', fontsize=11)
ax3.set_title('Coefficient Comparison', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(features)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Feature Importance (Absolute Standardized Coefficients)
ax4 = fig.add_subplot(2, 3, 4)
importance = np.abs(model_scaled.coef_)
colors = plt.cm.viridis(importance / importance.max())
bars = ax4.barh(features, importance, color=colors)
ax4.set_xlabel('Absolute Standardized Coefficient', fontsize=11)
ax4.set_ylabel('Features', fontsize=11)
ax4.set_title('Feature Importance', fontsize=13, fontweight='bold')
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, importance)):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
             ha='left', va='center', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Effect of +1 Standard Deviation Change
ax5 = fig.add_subplot(2, 3, 5)
colors = ['red' if x < 0 else 'green' for x in effect_per_std]
bars = ax5.barh(features, effect_per_std, color=colors, alpha=0.7)
ax5.set_xlabel('Effect on Disease Progression', fontsize=11)
ax5.set_ylabel('Features', fontsize=11)
ax5.set_title('Effect of 1 Standard Deviation Increase', fontsize=13, fontweight='bold')
ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
# Add value labels
for i, (bar, val) in enumerate(zip(bars, effect_per_std)):
    ax5.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
             ha='left' if val > 0 else 'right', va='center', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# 6. Distribution of Predictions vs Actual (FIXED: changed edgecolors to edgecolor)
ax6 = fig.add_subplot(2, 3, 6)
ax6.hist(y, bins=30, alpha=0.5, label='Actual', color='steelblue', edgecolor='black')
ax6.hist(y_pred_raw, bins=30, alpha=0.5, label='Predicted', color='coral', edgecolor='black')
ax6.set_xlabel('Disease Progression', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Distribution: Actual vs Predicted', fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('Diabetes Progression Model Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Additional plots for deeper analysis
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

# Correlation matrix heatmap
corr_matrix = df[features + ['target']].corr()
im = axes[0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(corr_matrix.columns)))
axes[0].set_yticks(range(len(corr_matrix.columns)))
axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
axes[0].set_yticklabels(corr_matrix.columns)
axes[0].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
# Add colorbar
plt.colorbar(im, ax=axes[0])

# Feature vs Target scatter plots
for idx, feature in enumerate(features):
    axes[1].scatter(df[feature], y, alpha=0.5, color='steelblue', edgecolors='black', linewidth=0.5)
    # Add regression line
    z = np.polyfit(df[feature], y, 1)
    p = np.poly1d(z)
    axes[1].plot(df[feature].sort_values(), p(df[feature].sort_values()), 
                 "r--", alpha=0.8, label=f'Slope: {z[0]:.2f}')
axes[1].set_xlabel('Feature Value', fontsize=11)
axes[1].set_ylabel('Disease Progression', fontsize=11)
axes[1].set_title('Feature vs Target Relationships', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Q-Q plot for residuals
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Advanced Model Diagnostics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Print model performance metrics
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)
print(f"R² Score: {r2_score(y, y_pred_raw):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred_raw):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y, y_pred_raw)):.2f}")

# Print interpretation summary
print("\n" + "="*80)
print("INTERPRETATION SUMMARY")
print("="*80)
print("For raw coefficients (Effect of +1 unit change):")
for i, feature in enumerate(features):
    print(f"  • {feature}: {model_raw.coef_[i]:.2f} units change in disease progression")

print("\nFor standardized coefficients (Effect of +1 std deviation change):")
for i, feature in enumerate(features):
    print(f"  • {feature}: {effect_per_std[i]:.2f} units change in disease progression")