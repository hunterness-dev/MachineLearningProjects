import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             log_loss, brier_score_loss)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("COMPARISON: Random Forest vs XGBoost for Spam Detection")
print("="*70)

# Generate synthetic data with overlap
n_samples = 500

# Spam emails (class 1)
spam_word = np.random.uniform(2, 9, 250)
spam_link = np.random.uniform(0.5, 6, 250)
spam_length = np.random.uniform(120, 400, 250)

# Normal emails (class 0)
normal_word = np.random.uniform(1, 7, 250)
normal_link = np.random.uniform(0, 4, 250)
normal_length = np.random.uniform(80, 450, 250)

X_spam = np.column_stack([spam_word, spam_link, spam_length])
X_normal = np.column_stack([normal_word, normal_link, normal_length])

X = np.vstack([X_spam, X_normal])
y = np.array([1]*250 + [0]*250)

# Add noise for overlap
noise = np.random.normal(0, 1.2, X.shape)
X = X + noise

# Shuffle
indices = np.random.permutation(500)
X = X[indices]
y = y[indices]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")
print(f"Class balance: {sum(y)} Spam, {len(y)-sum(y)} Normal")
print()

# ============================================
# TRAIN MODELS
# ============================================

# Random Forest
print("="*70)
print("1. TRAINING RANDOM FOREST")
print("="*70)
rf_start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_time = time.time() - rf_start
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Training time: {rf_time:.4f} seconds")
print(f"Test Accuracy: {rf_acc:.4f} ({rf_acc:.2%})")

# XGBoost Default
print("\n" + "="*70)
print("2. TRAINING XGBOOST (Default Parameters)")
print("="*70)
xgb_default_start = time.time()
xgb_default = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
xgb_default_time = time.time() - xgb_default_start
xgb_default_pred = xgb_default.predict(X_test)
xgb_default_prob = xgb_default.predict_proba(X_test)[:, 1]
xgb_default_acc = accuracy_score(y_test, xgb_default_pred)
print(f"Training time: {xgb_default_time:.4f} seconds")
print(f"Test Accuracy: {xgb_default_acc:.4f} ({xgb_default_acc:.2%})")

# XGBoost Tuned
print("\n" + "="*70)
print("3. XGBOOST WITH GRID SEARCH")
print("="*70)
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0]
}
print("Searching for best parameters...")
grid_start = time.time()
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
grid_xgb.fit(X_train, y_train)
grid_time = time.time() - grid_start
best_xgb = grid_xgb.best_estimator_
best_xgb_pred = best_xgb.predict(X_test)
best_xgb_prob = best_xgb.predict_proba(X_test)[:, 1]
best_xgb_acc = accuracy_score(y_test, best_xgb_pred)
print(f"Grid search time: {grid_time:.2f} seconds")
print(f"Best parameters: {grid_xgb.best_params_}")
print(f"Test Accuracy: {best_xgb_acc:.4f} ({best_xgb_acc:.2%})")

# ============================================
# COLLECT ALL METRICS
# ============================================
models = {
    'Random Forest': {'pred': rf_pred, 'prob': rf_prob, 'time': rf_time, 'name': 'RF'},
    'XGBoost Default': {'pred': xgb_default_pred, 'prob': xgb_default_prob, 'time': xgb_default_time, 'name': 'XGB-D'},
    'XGBoost Tuned': {'pred': best_xgb_pred, 'prob': best_xgb_prob, 'time': grid_time, 'name': 'XGB-T'}
}

metrics_results = {}
for name, data in models.items():
    metrics_results[name] = {
        'Accuracy': accuracy_score(y_test, data['pred']),
        'Precision': precision_score(y_test, data['pred']),
        'Recall': recall_score(y_test, data['pred']),
        'F1-Score': f1_score(y_test, data['pred']),
        'MCC': matthews_corrcoef(y_test, data['pred']),
        'Log Loss': log_loss(y_test, data['prob']),
        'Brier Score': brier_score_loss(y_test, data['prob'])
    }

# Cross-validation scores
rf_cv = cross_val_score(rf, X, y, cv=5)
xgb_default_cv = cross_val_score(xgb_default, X, y, cv=5)
xgb_best_cv = cross_val_score(best_xgb, X, y, cv=5)

# ROC and AUC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
rf_auc = auc(rf_fpr, rf_tpr)
xgb_default_fpr, xgb_default_tpr, _ = roc_curve(y_test, xgb_default_prob)
xgb_default_auc = auc(xgb_default_fpr, xgb_default_tpr)
xgb_best_fpr, xgb_best_tpr, _ = roc_curve(y_test, best_xgb_prob)
xgb_best_auc = auc(xgb_best_fpr, xgb_best_tpr)

# Precision-Recall curves
rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_prob)
xgb_default_prec, xgb_default_rec, _ = precision_recall_curve(y_test, xgb_default_prob)
xgb_best_prec, xgb_best_rec, _ = precision_recall_curve(y_test, best_xgb_prob)

# ============================================
# CREATE 4x4 GRID OF PLOTS (16 PLOTS)
# ============================================
fig = plt.figure(figsize=(24, 20))

# Color scheme
rf_color = '#2E86AB'
xgb_default_color = '#E63946'
xgb_tuned_color = '#2A9D8F'

# ===== ROW 1: Accuracy and Performance =====
# 1. Accuracy Comparison Bar Chart
ax1 = fig.add_subplot(4, 4, 1)
models_names = list(metrics_results.keys())
accuracies = [metrics_results[m]['Accuracy'] for m in models_names]
colors = [rf_color, xgb_default_color, xgb_tuned_color]
bars = ax1.bar(models_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylim([0, 1])
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('1. Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_xticklabels(models_names, rotation=15, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. F1-Score Comparison
ax2 = fig.add_subplot(4, 4, 2)
f1_scores = [metrics_results[m]['F1-Score'] for m in models_names]
bars = ax2.bar(models_names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylim([0, 1])
ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax2.set_title('2. F1-Score Comparison', fontsize=12, fontweight='bold')
ax2.set_xticklabels(models_names, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
for bar, f1 in zip(bars, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Precision vs Recall Scatter
ax3 = fig.add_subplot(4, 4, 3)
precisions = [metrics_results[m]['Precision'] for m in models_names]
recalls = [metrics_results[m]['Recall'] for m in models_names]
ax3.scatter(precisions, recalls, c=colors, s=100, edgecolors='black', linewidth=1.5)
for i, name in enumerate(models_names):
    ax3.annotate(name, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
ax3.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax3.set_ylabel('Recall', fontsize=11, fontweight='bold')
ax3.set_title('3. Precision vs Recall', fontsize=12, fontweight='bold')
ax3.set_xlim([0.6, 0.8])
ax3.set_ylim([0.6, 0.8])
ax3.grid(True, alpha=0.3)

# 4. Training Time Comparison
ax4 = fig.add_subplot(4, 4, 4)
times = [rf_time, xgb_default_time, grid_time]
time_labels = ['RF', 'XGB-D', 'XGB-T']
time_colors = [rf_color, xgb_default_color, xgb_tuned_color]
bars = ax4.bar(time_labels, times, color=time_colors, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax4.set_title('4. Training Time', fontsize=12, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, axis='y')
for bar, t in zip(bars, times):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{t:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

# ===== ROW 2: Confusion Matrices =====
# 5. Confusion Matrix - Random Forest
ax5 = fig.add_subplot(4, 4, 5)
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax5.set_title('5. RF - Confusion Matrix', fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted', fontsize=10)
ax5.set_ylabel('Actual', fontsize=10)

# 6. Confusion Matrix - XGBoost Default
ax6 = fig.add_subplot(4, 4, 6)
xgb_default_cm = confusion_matrix(y_test, xgb_default_pred)
sns.heatmap(xgb_default_cm, annot=True, fmt='d', cmap='Reds', ax=ax6,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax6.set_title('6. XGB-Default - Confusion Matrix', fontsize=12, fontweight='bold')
ax6.set_xlabel('Predicted', fontsize=10)
ax6.set_ylabel('Actual', fontsize=10)

# 7. Confusion Matrix - XGBoost Tuned
ax7 = fig.add_subplot(4, 4, 7)
xgb_best_cm = confusion_matrix(y_test, best_xgb_pred)
sns.heatmap(xgb_best_cm, annot=True, fmt='d', cmap='Greens', ax=ax7,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax7.set_title('7. XGB-Tuned - Confusion Matrix', fontsize=12, fontweight='bold')
ax7.set_xlabel('Predicted', fontsize=10)
ax7.set_ylabel('Actual', fontsize=10)

# 8. Normalized Confusion Matrix Comparison
ax8 = fig.add_subplot(4, 4, 8)
rf_cm_norm = rf_cm / rf_cm.sum(axis=1)[:, np.newaxis]
xgb_best_cm_norm = xgb_best_cm / xgb_best_cm.sum(axis=1)[:, np.newaxis]
x = np.arange(2)
width = 0.35
ax8.bar(x - width/2, rf_cm_norm[0], width, label='RF - Normal', color=rf_color, alpha=0.7)
ax8.bar(x + width/2, xgb_best_cm_norm[0], width, label='XGB-T - Normal', color=xgb_tuned_color, alpha=0.7)
ax8.set_xticks(x)
ax8.set_xticklabels(['Correct', 'Wrong'])
ax8.set_ylabel('Rate', fontsize=11)
ax8.set_title('8. Normalized Errors', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# ===== ROW 3: ROC, PR, Calibration =====
# 9. ROC Curves
ax9 = fig.add_subplot(4, 4, 9)
ax9.plot(rf_fpr, rf_tpr, color=rf_color, lw=2, label=f'RF (AUC = {rf_auc:.3f})')
ax9.plot(xgb_default_fpr, xgb_default_tpr, color=xgb_default_color, lw=2, label=f'XGB-D (AUC = {xgb_default_auc:.3f})')
ax9.plot(xgb_best_fpr, xgb_best_tpr, color=xgb_tuned_color, lw=2, label=f'XGB-T (AUC = {xgb_best_auc:.3f})')
ax9.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax9.set_xlabel('False Positive Rate', fontsize=11)
ax9.set_ylabel('True Positive Rate', fontsize=11)
ax9.set_title('9. ROC Curves', fontsize=12, fontweight='bold')
ax9.legend(loc='lower right', fontsize=8)
ax9.grid(True, alpha=0.3)

# 10. Precision-Recall Curves
ax10 = fig.add_subplot(4, 4, 10)
ax10.plot(rf_rec, rf_prec, color=rf_color, lw=2, label='RF')
ax10.plot(xgb_default_rec, xgb_default_prec, color=xgb_default_color, lw=2, label='XGB-D')
ax10.plot(xgb_best_rec, xgb_best_prec, color=xgb_tuned_color, lw=2, label='XGB-T')
ax10.set_xlabel('Recall', fontsize=11)
ax10.set_ylabel('Precision', fontsize=11)
ax10.set_title('10. Precision-Recall Curves', fontsize=12, fontweight='bold')
ax10.legend(loc='lower left', fontsize=8)
ax10.grid(True, alpha=0.3)

# 11. Calibration Curves (Reliability Diagrams)
ax11 = fig.add_subplot(4, 4, 11)
from sklearn.calibration import calibration_curve
for name, data, color in zip(models_names, [rf_prob, xgb_default_prob, best_xgb_prob], colors):
    prob_true, prob_pred = calibration_curve(y_test, data, n_bins=5)
    ax11.plot(prob_pred, prob_true, 'o-', label=name, color=color, lw=2, markersize=8)
ax11.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax11.set_xlabel('Mean Predicted Probability', fontsize=11)
ax11.set_ylabel('Fraction of Positives', fontsize=11)
ax11.set_title('11. Calibration Curves', fontsize=12, fontweight='bold')
ax11.legend(loc='lower right', fontsize=7)
ax11.grid(True, alpha=0.3)

# 12. Probability Distribution
ax12 = fig.add_subplot(4, 4, 12)
ax12.hist(rf_prob[y_test==1], alpha=0.5, label='RF - Spam', color=rf_color, bins=10)
ax12.hist(rf_prob[y_test==0], alpha=0.5, label='RF - Normal', color='lightblue', bins=10)
ax12.set_xlabel('Predicted Probability', fontsize=11)
ax12.set_ylabel('Frequency', fontsize=11)
ax12.set_title('12. RF Probability Distribution', fontsize=12, fontweight='bold')
ax12.legend(fontsize=8)
ax12.grid(True, alpha=0.3)

# ===== ROW 4: Cross-validation and Metrics =====
# 13. Cross-Validation Boxplot
ax13 = fig.add_subplot(4, 4, 13)
cv_data = [rf_cv, xgb_default_cv, xgb_best_cv]
bp = ax13.boxplot(cv_data, tick_labels=['RF', 'XGB-D', 'XGB-T'], patch_artist=True)
box_colors = [rf_color, xgb_default_color, xgb_tuned_color]
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(box_colors[i])
    patch.set_alpha(0.7)
ax13.set_ylabel('CV Score', fontsize=11)
ax13.set_title('13. 5-Fold CV Distribution', fontsize=12, fontweight='bold')
ax13.set_ylim([0.5, 0.9])
ax13.grid(True, alpha=0.3, axis='y')

# 14. Feature Importance Comparison
ax14 = fig.add_subplot(4, 4, 14)
features = ['Word Count', 'Link Count', 'Email Length']
rf_imp = rf.feature_importances_
xgb_imp = best_xgb.feature_importances_
x = np.arange(len(features))
width = 0.35
ax14.bar(x - width/2, rf_imp, width, label='RF', color=rf_color, alpha=0.8)
ax14.bar(x + width/2, xgb_imp, width, label='XGB-T', color=xgb_tuned_color, alpha=0.8)
ax14.set_xticks(x)
ax14.set_xticklabels(features, rotation=15)
ax14.set_ylabel('Importance', fontsize=11)
ax14.set_title('14. Feature Importance', fontsize=12, fontweight='bold')
ax14.legend()
ax14.grid(True, alpha=0.3, axis='y')

# 15. Radar Chart for Multiple Metrics
ax15 = fig.add_subplot(4, 4, 15, projection='polar')
metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'CV Score']
rf_radar = [metrics_results['Random Forest']['Accuracy'],
            metrics_results['Random Forest']['Precision'],
            metrics_results['Random Forest']['Recall'],
            metrics_results['Random Forest']['F1-Score'],
            metrics_results['Random Forest']['MCC'],
            rf_cv.mean()]
xgb_default_radar = [metrics_results['XGBoost Default']['Accuracy'],
                     metrics_results['XGBoost Default']['Precision'],
                     metrics_results['XGBoost Default']['Recall'],
                     metrics_results['XGBoost Default']['F1-Score'],
                     metrics_results['XGBoost Default']['MCC'],
                     xgb_default_cv.mean()]
xgb_tuned_radar = [metrics_results['XGBoost Tuned']['Accuracy'],
                   metrics_results['XGBoost Tuned']['Precision'],
                   metrics_results['XGBoost Tuned']['Recall'],
                   metrics_results['XGBoost Tuned']['F1-Score'],
                   metrics_results['XGBoost Tuned']['MCC'],
                   xgb_best_cv.mean()]
angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
rf_radar += rf_radar[:1]
xgb_default_radar += xgb_default_radar[:1]
xgb_tuned_radar += xgb_tuned_radar[:1]
angles += angles[:1]
ax15.plot(angles, rf_radar, 'o-', linewidth=2, color=rf_color, label='RF')
ax15.fill(angles, rf_radar, alpha=0.25, color=rf_color)
ax15.plot(angles, xgb_default_radar, 'o-', linewidth=2, color=xgb_default_color, label='XGB-D')
ax15.fill(angles, xgb_default_radar, alpha=0.25, color=xgb_default_color)
ax15.plot(angles, xgb_tuned_radar, 'o-', linewidth=2, color=xgb_tuned_color, label='XGB-T')
ax15.fill(angles, xgb_tuned_radar, alpha=0.25, color=xgb_tuned_color)
ax15.set_xticks(angles[:-1])
ax15.set_xticklabels(metrics_radar, fontsize=8)
ax15.set_title('15. Radar Chart Comparison', fontsize=12, fontweight='bold', pad=20)
ax15.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)

# 16. Performance Summary Table
ax16 = fig.add_subplot(4, 4, 16)
ax16.axis('off')
table_data = [
    ['Metric', 'RF', 'XGB-D', 'XGB-T'],
    ['Accuracy', f'{metrics_results["Random Forest"]["Accuracy"]:.3f}', 
     f'{metrics_results["XGBoost Default"]["Accuracy"]:.3f}', 
     f'{metrics_results["XGBoost Tuned"]["Accuracy"]:.3f}'],
    ['Precision', f'{metrics_results["Random Forest"]["Precision"]:.3f}', 
     f'{metrics_results["XGBoost Default"]["Precision"]:.3f}', 
     f'{metrics_results["XGBoost Tuned"]["Precision"]:.3f}'],
    ['Recall', f'{metrics_results["Random Forest"]["Recall"]:.3f}', 
     f'{metrics_results["XGBoost Default"]["Recall"]:.3f}', 
     f'{metrics_results["XGBoost Tuned"]["Recall"]:.3f}'],
    ['F1-Score', f'{metrics_results["Random Forest"]["F1-Score"]:.3f}', 
     f'{metrics_results["XGBoost Default"]["F1-Score"]:.3f}', 
     f'{metrics_results["XGBoost Tuned"]["F1-Score"]:.3f}'],
    ['MCC', f'{metrics_results["Random Forest"]["MCC"]:.3f}', 
     f'{metrics_results["XGBoost Default"]["MCC"]:.3f}', 
     f'{metrics_results["XGBoost Tuned"]["MCC"]:.3f}'],
    ['Log Loss', f'{metrics_results["Random Forest"]["Log Loss"]:.3f}', 
     f'{metrics_results["XGBoost Default"]["Log Loss"]:.3f}', 
     f'{metrics_results["XGBoost Tuned"]["Log Loss"]:.3f}'],
    ['AUC', f'{rf_auc:.3f}', f'{xgb_default_auc:.3f}', f'{xgb_best_auc:.3f}'],
    ['CV Mean', f'{rf_cv.mean():.3f}', f'{xgb_default_cv.mean():.3f}', f'{xgb_best_cv.mean():.3f}'],
    ['Time (s)', f'{rf_time:.2f}', f'{xgb_default_time:.2f}', f'{grid_time:.2f}']
]
table = ax16.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)
for i, col in enumerate(['lightblue', rf_color, xgb_default_color, xgb_tuned_color]):
    for j in range(len(table_data)):
        if i == 0:
            table[(j, i)].set_facecolor('#333333')
            table[(j, i)].set_text_props(weight='bold', color='white')
        elif j == 0:
            table[(j, i)].set_facecolor('#dddddd')
            table[(j, i)].set_text_props(weight='bold')
        else:
            table[(j, i)].set_facecolor(col)
ax16.set_title('16. Performance Summary', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Random Forest vs XGBoost: Comprehensive 16-Plot Comparison', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ============================================
# PRINT FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("FINAL CONCLUSION")
print("="*70)

print(f"\n{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Time (s)':<12} {'CV Mean':<12}")
print("-"*70)
print(f"{'Random Forest':<20} {metrics_results['Random Forest']['Accuracy']:<12.4f} "
      f"{metrics_results['Random Forest']['F1-Score']:<12.4f} {rf_time:<12.2f} {rf_cv.mean():<12.4f}")
print(f"{'XGBoost Default':<20} {metrics_results['XGBoost Default']['Accuracy']:<12.4f} "
      f"{metrics_results['XGBoost Default']['F1-Score']:<12.4f} {xgb_default_time:<12.2f} {xgb_default_cv.mean():<12.4f}")
print(f"{'XGBoost Tuned':<20} {metrics_results['XGBoost Tuned']['Accuracy']:<12.4f} "
      f"{metrics_results['XGBoost Tuned']['F1-Score']:<12.4f} {grid_time:<12.2f} {xgb_best_cv.mean():<12.4f}")

print("\n" + "="*70)