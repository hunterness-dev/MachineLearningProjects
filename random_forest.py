import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time

np.random.seed(42)

print("="*70)
print("COMPARISON: Random Forest vs Decision Tree for Spam Detection")
print("COMPLEX DATA WITH HIGH OVERLAP")
print("="*70)

# Generate MORE COMPLEX data with significant overlap
n_samples = 500

# ============================================
# CREATE DATA WITH OVERLAP
# ============================================

# Strategy: Make the distributions overlap significantly
# Spam and Normal will share similar ranges

# Spam emails (class 1) - shifted closer to normal
spam_word = np.random.uniform(2, 9, 250)      # 2-9 words (overlaps with normal)
spam_link = np.random.uniform(0.5, 6, 250)    # 0.5-6 links (overlaps significantly)
spam_length = np.random.uniform(120, 400, 250) # 120-400 characters

# Normal emails (class 0) - shifted closer to spam
normal_word = np.random.uniform(1, 7, 250)     # 1-7 words (high overlap: 2-7)
normal_link = np.random.uniform(0, 4, 250)     # 0-4 links (high overlap: 0.5-4)
normal_length = np.random.uniform(80, 450, 250) # 80-450 characters (high overlap)

# Combine features
X_spam = np.column_stack([spam_word, spam_link, spam_length])
X_normal = np.column_stack([normal_word, normal_link, normal_length])

X = np.vstack([X_spam, X_normal])
y = np.array([1]*250 + [0]*250)

# Add significant noise to create even more overlap
noise = np.random.normal(0, 1.2, X.shape)  # Increased noise
X = X + noise

# Shuffle
indices = np.random.permutation(500)
X = X[indices]
y = y[indices]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
print(f"Class balance: {sum(y)} Spam, {len(y)-sum(y)} Normal")

# ============================================
# VISUALIZE THE OVERLAP
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Word Count distribution
axes[0].hist(X[y==0][:, 0], alpha=0.5, label='Normal', color='blue', bins=20)
axes[0].hist(X[y==1][:, 0], alpha=0.5, label='Spam', color='red', bins=20)
axes[0].set_xlabel('Word Count', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Word Count Distribution (High Overlap)', fontsize=12)
axes[0].legend()

# Plot 2: Link Count distribution
axes[1].hist(X[y==0][:, 1], alpha=0.5, label='Normal', color='blue', bins=20)
axes[1].hist(X[y==1][:, 1], alpha=0.5, label='Spam', color='red', bins=20)
axes[1].set_xlabel('Link Count', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Link Count Distribution (High Overlap)', fontsize=12)
axes[1].legend()

# Plot 3: Email Length distribution
axes[2].hist(X[y==0][:, 2], alpha=0.5, label='Normal', color='blue', bins=20)
axes[2].hist(X[y==1][:, 2], alpha=0.5, label='Spam', color='red', bins=20)
axes[2].set_xlabel('Email Length', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('Email Length Distribution (High Overlap)', fontsize=12)
axes[2].legend()

plt.suptitle('Feature Distributions - High Overlap Between Classes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# TRAIN MODELS
# ============================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:, 1]
dt_prob = dt.predict_proba(X_test)[:, 1]

# ============================================
# METRICS
# ============================================
rf_acc = accuracy_score(y_test, rf_pred)
dt_acc = accuracy_score(y_test, dt_pred)

rf_cm = confusion_matrix(y_test, rf_pred)
dt_cm = confusion_matrix(y_test, dt_pred)

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Metric':<20} {'Random Forest':<18} {'Decision Tree':<18}")
print("-"*70)
print(f"{'Accuracy':<20} {rf_acc:<18.4f} {dt_acc:<18.4f}")
print(f"{'Improvement':<20} {((rf_acc - dt_acc)/dt_acc*100):<18.2f}%")

# Cross-validation scores
rf_cv = cross_val_score(rf, X, y, cv=5)
dt_cv = cross_val_score(dt, X, y, cv=5)

print(f"{'CV Mean (5-fold)':<20} {rf_cv.mean():<18.4f} {dt_cv.mean():<18.4f}")
print(f"{'CV Std':<20} {rf_cv.std():<18.4f} {dt_cv.std():<18.4f}")

# ============================================
# COMPREHENSIVE VISUALIZATION
# ============================================
fig = plt.figure(figsize=(20, 14))

rf_color = '#2E86AB'
dt_color = '#A23B72'

# 1. Accuracy Comparison
ax1 = fig.add_subplot(3, 3, 1)
models = ['Random Forest', 'Decision Tree']
accuracies = [rf_acc, dt_acc]
bars = ax1.bar(models, accuracies, color=[rf_color, dt_color], edgecolor='black')
ax1.set_ylim([0, 1])
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy Comparison (Overlapping Data)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Confusion Matrix - Random Forest
ax2 = fig.add_subplot(3, 3, 2)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax2.set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')

# 3. Confusion Matrix - Decision Tree
ax3 = fig.add_subplot(3, 3, 3)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Oranges', ax=ax3,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax3.set_title('Decision Tree - Confusion Matrix', fontsize=12, fontweight='bold')

# 4. ROC Curves
ax4 = fig.add_subplot(3, 3, 4)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
rf_auc = auc(rf_fpr, rf_tpr)
dt_auc = auc(dt_fpr, dt_tpr)

ax4.plot(rf_fpr, rf_tpr, color=rf_color, lw=2, label=f'RF (AUC = {rf_auc:.3f})')
ax4.plot(dt_fpr, dt_tpr, color=dt_color, lw=2, label=f'DT (AUC = {dt_auc:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax4.set_xlabel('False Positive Rate', fontsize=11)
ax4.set_ylabel('True Positive Rate', fontsize=11)
ax4.set_title('ROC Curves (Overlapping Data)', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 5. Cross-validation boxplot
ax5 = fig.add_subplot(3, 3, 5)
cv_data = [rf_cv, dt_cv]
bp = ax5.boxplot(cv_data, labels=['RF', 'DT'], patch_artist=True)
bp['boxes'][0].set_facecolor(rf_color)
bp['boxes'][1].set_facecolor(dt_color)
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_alpha(0.7)
ax5.set_ylabel('CV Score', fontsize=11)
ax5.set_title('5-Fold CV Distribution', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0.6, 1.0])

# 6. Feature Importance
ax6 = fig.add_subplot(3, 3, 6)
features = ['Word Count', 'Link Count', 'Email Length']
rf_imp = rf.feature_importances_
dt_imp = dt.feature_importances_
x = np.arange(len(features))
width = 0.35
ax6.bar(x - width/2, rf_imp, width, label='RF', color=rf_color, alpha=0.8)
ax6.bar(x + width/2, dt_imp, width, label='DT', color=dt_color, alpha=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels(features)
ax6.set_ylabel('Importance', fontsize=11)
ax6.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# 7. Precision-Recall Curves
ax7 = fig.add_subplot(3, 3, 7)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
dt_precision, dt_recall, _ = precision_recall_curve(y_test, dt_prob)
ax7.plot(rf_recall, rf_precision, color=rf_color, lw=2, label='Random Forest')
ax7.plot(dt_recall, dt_precision, color=dt_color, lw=2, label='Decision Tree')
ax7.set_xlabel('Recall', fontsize=11)
ax7.set_ylabel('Precision', fontsize=11)
ax7.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Depth vs Performance (Decision Tree)
ax8 = fig.add_subplot(3, 3, 8)
depths = range(1, 20)
dt_train_acc = []
dt_test_acc = []
for d in depths:
    dt_temp = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_temp.fit(X_train, y_train)
    dt_train_acc.append(accuracy_score(y_train, dt_temp.predict(X_train)))
    dt_test_acc.append(accuracy_score(y_test, dt_temp.predict(X_test)))
ax8.plot(depths, dt_train_acc, 'o-', label='Train', color=dt_color, linewidth=2)
ax8.plot(depths, dt_test_acc, 's-', label='Test', color='green', linewidth=2)
ax8.axhline(y=rf_acc, color=rf_color, linestyle='--', label=f'RF Baseline ({rf_acc:.3f})')
ax8.set_xlabel('Max Depth', fontsize=11)
ax8.set_ylabel('Accuracy', fontsize=11)
ax8.set_title('Decision Tree: Depth vs Performance', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Prediction Confidence Distribution
ax9 = fig.add_subplot(3, 3, 9)
ax9.hist(rf_prob[y_test==1], alpha=0.5, label='RF - Actual Spam', color=rf_color, bins=15)
ax9.hist(rf_prob[y_test==0], alpha=0.5, label='RF - Actual Normal', color='lightblue', bins=15)
ax9.set_xlabel('Predicted Probability', fontsize=11)
ax9.set_ylabel('Frequency', fontsize=11)
ax9.set_title('RF Prediction Confidence', fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.suptitle('Random Forest vs Decision Tree: High Overlap Data Comparison', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ============================================
# PRINT DETAILED RESULTS
# ============================================
from sklearn.metrics import precision_score, recall_score, f1_score

rf_precision = precision_score(y_test, rf_pred)
dt_precision = precision_score(y_test, dt_pred)
rf_recall = recall_score(y_test, rf_pred)
dt_recall = recall_score(y_test, dt_pred)
rf_f1 = f1_score(y_test, rf_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("\n" + "="*70)
print("DETAILED METRICS COMPARISON (High Overlap Data)")
print("="*70)
print(f"\n{'Metric':<15} {'Random Forest':<20} {'Decision Tree':<20} {'Difference':<15}")
print("-"*70)
print(f"{'Accuracy':<15} {rf_acc:<20.4f} {dt_acc:<20.4f} {rf_acc-dt_acc:+.4f}")
print(f"{'Precision':<15} {rf_precision:<20.4f} {dt_precision:<20.4f} {rf_precision-dt_precision:+.4f}")
print(f"{'Recall':<15} {rf_recall:<20.4f} {dt_recall:<20.4f} {rf_recall-dt_recall:+.4f}")
print(f"{'F1-Score':<15} {rf_f1:<20.4f} {dt_f1:<20.4f} {rf_f1-dt_f1:+.4f}")
print(f"{'AUC-ROC':<15} {rf_auc:<20.4f} {dt_auc:<20.4f} {rf_auc-dt_auc:+.4f}")

# ============================================
# ERROR ANALYSIS
# ============================================
rf_misclassified = np.where(rf_pred != y_test)[0]
dt_misclassified = np.where(dt_pred != y_test)[0]
common_errors = set(rf_misclassified) & set(dt_misclassified)

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)
print(f"\nRandom Forest misclassified: {len(rf_misclassified)}/{len(y_test)} ({len(rf_misclassified)/len(y_test)*100:.1f}%)")
print(f"Decision Tree misclassified: {len(dt_misclassified)}/{len(y_test)} ({len(dt_misclassified)/len(y_test)*100:.1f}%)")
print(f"Both models misclassified same samples: {len(common_errors)}")

# ============================================
# SPEED COMPARISON
# ============================================
rf_time_start = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - rf_time_start

dt_time_start = time.time()
dt.fit(X_train, y_train)
dt_time = time.time() - dt_time_start

print("\n" + "="*70)
print("COMPUTATIONAL COMPLEXITY")
print("="*70)
print(f"\n{'Metric':<20} {'Random Forest':<18} {'Decision Tree':<18}")
print("-"*70)
print(f"{'Training Time (s)':<20} {rf_time:<18.6f} {dt_time:<18.6f}")
print(f"{'Training Speed':<20} {'Slower (ensemble)':<18} {'Faster (single tree)':<18}")

# ============================================
# CONCLUSION
# ============================================
print("\n" + "="*70)
print("CONCLUSION - High Overlap Data")
print("="*70)

print(f"\nData Complexity: HIGH OVERLAP between classes")
print(f"Random Forest Accuracy: {rf_acc:.2%}")
print(f"Decision Tree Accuracy: {dt_acc:.2%}")
print(f"Improvement with RF: {(rf_acc-dt_acc)*100:.1f}%")

if rf_acc - dt_acc > 0.05:
    print("\n✅ Random Forest significantly outperforms Decision Tree on overlapping data")
    print("   - Ensemble method handles noise better")
    print("   - Less prone to overfitting")
    print("   - Better generalization on complex boundaries")
elif rf_acc - dt_acc > 0.02:
    print("\n✅ Random Forest outperforms Decision Tree on overlapping data")
    print("   - Ensemble provides better decision boundaries")
else:
    print("\n⚠️ Both models perform similarly on this dataset")
    print("   - Consider computational cost for your use case")

print("\n" + "="*70)