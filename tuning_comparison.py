import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time

np.random.seed(42)

# ========== 1. Data Generation ==========
n_samples = 500

spam_word = np.random.uniform(2, 9, 250)
spam_link = np.random.uniform(0.5, 6, 250)
spam_length = np.random.uniform(120, 400, 250)

normal_word = np.random.uniform(1, 7, 250)
normal_link = np.random.uniform(0, 4, 250)
normal_length = np.random.uniform(80, 450, 250)

X_spam = np.column_stack([spam_word, spam_link, spam_length])
X_normal = np.column_stack([normal_word, normal_link, normal_length])

X = np.vstack([X_spam, X_normal])
y = np.array([1]*250 + [0]*250)

noise = np.random.normal(0, 1.2, X.shape)
X = X + noise

indices = np.random.permutation(500)
X = X[indices]
y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Spam:", sum(y), "Normal:", len(y)-sum(y))
print("Train size:", len(X_train), "Test size:", len(X_test))

# ========== 2. Base Decision Tree Model ==========
print("\n" + "="*50)
print("Base Decision Tree Model")
print("="*50)

dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)
y_pred_dt = dt_base.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# ========== 3. Base Random Forest Model ==========
print("\n" + "="*50)
print("Base Random Forest Model")
print("="*50)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
y_pred_rf = rf_base.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# ========== 4. Cross-Validation ==========
print("\n" + "="*50)
print("Cross-Validation")
print("="*50)

cv_scores_dt = cross_val_score(dt_base, X_train, y_train, cv=5)
cv_scores_rf = cross_val_score(rf_base, X_train, y_train, cv=5)

print(f"Decision Tree - CV mean: {cv_scores_dt.mean():.4f} (±{cv_scores_dt.std():.4f})")
print(f"Random Forest - CV mean: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std():.4f})")

# ========== 5. GridSearchCV Optimization ==========
print("\n" + "="*50)
print("GridSearchCV Optimization for Random Forest")
print("="*50)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\nBest parameters (GridSearch): {grid_search.best_params_}")
print(f"Best CV score (GridSearch): {grid_search.best_score_:.4f}")
print(f"Search time: {grid_time:.2f} seconds")

best_rf_grid = grid_search.best_estimator_
y_pred_grid = best_rf_grid.predict(X_test)
print(f"Test accuracy (GridSearch): {accuracy_score(y_test, y_pred_grid):.4f}")

# ========== 6. RandomizedSearchCV Optimization ==========
print("\n" + "="*50)
print("RandomizedSearchCV Optimization for Random Forest")
print("="*50)

param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

start_time_random = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time_random

print(f"\nBest parameters (Randomized): {random_search.best_params_}")
print(f"Best CV score (Randomized): {random_search.best_score_:.4f}")
print(f"Search time: {random_time:.2f} seconds")

best_rf_random = random_search.best_estimator_
y_pred_random = best_rf_random.predict(X_test)
print(f"Test accuracy (Randomized): {accuracy_score(y_test, y_pred_random):.4f}")

# ========== 7. ALL PLOTS IN ONE FIGURE ==========
fig = plt.figure(figsize=(22, 18))

# 1. Confusion Matrix - Decision Tree
ax1 = fig.add_subplot(3, 3, 1)
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax1.set_title('Decision Tree - Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# 2. Confusion Matrix - Random Forest
ax2 = fig.add_subplot(3, 3, 2)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax2.set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# 3. Confusion Matrix - GridSearch RF
ax3 = fig.add_subplot(3, 3, 3)
cm_grid = confusion_matrix(y_test, y_pred_grid)
sns.heatmap(cm_grid, annot=True, fmt='d', cmap='Oranges', ax=ax3,
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
ax3.set_title('GridSearch RF - Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# 4. ROC Curves
ax4 = fig.add_subplot(3, 3, 4)
for model, name, color in [(dt_base, 'Decision Tree', 'blue'),
                             (rf_base, 'Random Forest', 'green'),
                             (best_rf_grid, 'GridSearch RF', 'orange'),
                             (best_rf_random, 'Randomized RF', 'purple')]:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax4.set_xlabel('False Positive Rate', fontsize=11)
ax4.set_ylabel('True Positive Rate', fontsize=11)
ax4.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(alpha=0.3)

# 5. Precision-Recall Curves
ax5 = fig.add_subplot(3, 3, 5)
for model, name, color in [(dt_base, 'Decision Tree', 'blue'),
                             (rf_base, 'Random Forest', 'green'),
                             (best_rf_grid, 'GridSearch RF', 'orange'),
                             (best_rf_random, 'Randomized RF', 'purple')]:
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax5.plot(recall, precision, color=color, lw=2, label=name)
ax5.set_xlabel('Recall', fontsize=11)
ax5.set_ylabel('Precision', fontsize=11)
ax5.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
ax5.legend(loc='best')
ax5.grid(alpha=0.3)

# 6. Feature Importance (GridSearch RF)
ax6 = fig.add_subplot(3, 3, 6)
feature_names = ['Word_Count', 'Link_Count', 'Length']
importance = best_rf_grid.feature_importances_
indices = np.argsort(importance)[::-1]
ax6.bar(range(len(importance)), importance[indices], color='steelblue')
ax6.set_xticks(range(len(importance)))
ax6.set_xticklabels([feature_names[i] for i in indices])
ax6.set_xlabel('Features', fontsize=11)
ax6.set_ylabel('Importance', fontsize=11)
ax6.set_title('Feature Importance - GridSearch RF', fontsize=12, fontweight='bold')
for i, imp in enumerate(importance[indices]):
    ax6.text(i, imp + 0.005, f'{imp:.3f}', ha='center')

# 7. CV Comparison Bar Chart
ax7 = fig.add_subplot(3, 3, 7)
models_cv = ['Decision Tree', 'Random Forest', 'GridSearch RF', 'Randomized RF']
cv_scores_list = [cv_scores_dt.mean(), cv_scores_rf.mean(), grid_search.best_score_, random_search.best_score_]
colors_cv = ['blue', 'green', 'orange', 'purple']
bars = ax7.bar(models_cv, cv_scores_list, color=colors_cv, alpha=0.7, edgecolor='black')
ax7.set_ylabel('CV Score', fontsize=11)
ax7.set_title('5-Fold Cross-Validation Comparison', fontsize=12, fontweight='bold')
ax7.set_ylim([0.6, 0.8])
ax7.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, cv_scores_list):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
             f'{score:.3f}', ha='center', fontweight='bold', fontsize=9)

# 8. Test Accuracy Comparison
ax8 = fig.add_subplot(3, 3, 8)
test_accs = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf), 
             accuracy_score(y_test, y_pred_grid), accuracy_score(y_test, y_pred_random)]
bars = ax8.bar(models_cv, test_accs, color=colors_cv, alpha=0.7, edgecolor='black')
ax8.set_ylabel('Test Accuracy', fontsize=11)
ax8.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
ax8.set_ylim([0.6, 0.8])
ax8.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, test_accs):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
             f'{acc:.3f}', ha='center', fontweight='bold', fontsize=9)

# 9. Time Comparison
ax9 = fig.add_subplot(3, 3, 9)
times = [0, 0, grid_time, random_time]
time_colors = ['blue', 'green', 'orange', 'purple']
bars = ax9.bar(models_cv, times, color=time_colors, alpha=0.7, edgecolor='black')
ax9.set_ylabel('Time (seconds)', fontsize=11)
ax9.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
for bar, t in zip(bars, times):
    if t > 0:
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{t:.1f}s', ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Random Forest vs Decision Tree: Comprehensive Comparison with Hyperparameter Tuning', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ========== 8. Final Model Comparison ==========
print("\n" + "="*50)
print("Final Model Comparison")
print("="*50)

print(f"\n{'Model':<20} {'Test Acc':<12} {'CV Mean':<12} {'Time (s)':<12}")
print("-"*56)
print(f"{'Decision Tree':<20} {accuracy_score(y_test, y_pred_dt):<12.4f} {cv_scores_dt.mean():<12.4f} {'-':<12}")
print(f"{'Random Forest':<20} {accuracy_score(y_test, y_pred_rf):<12.4f} {cv_scores_rf.mean():<12.4f} {'-':<12}")
print(f"{'GridSearch RF':<20} {accuracy_score(y_test, y_pred_grid):<12.4f} {grid_search.best_score_:<12.4f} {grid_time:<12.2f}")
print(f"{'Randomized RF':<20} {accuracy_score(y_test, y_pred_random):<12.4f} {random_search.best_score_:<12.4f} {random_time:<12.2f}")

# ========== 9. Prediction for New Samples ==========
print("\n" + "="*50)
print("Prediction for New Samples (Using Best Model - GridSearch RF)")
print("="*50)

new_samples = np.array([
    [5.0, 2.0, 200],
    [2.0, 0.5, 150],
    [8.0, 5.0, 350]
])

predictions = best_rf_grid.predict(new_samples)
probabilities = best_rf_grid.predict_proba(new_samples)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "SPAM" if pred == 1 else "NORMAL"
    print(f"Sample {i+1}: {label} (Spam probability: {prob[1]:.2f})")

# ========== 10. Conclusion ==========
print("\n" + "="*50)
print("CONCLUSION")
print("="*50)

print(f"\nBest Model: GridSearchCV Random Forest")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"Improvement over Decision Tree: {(accuracy_score(y_test, y_pred_grid) - accuracy_score(y_test, y_pred_dt)) * 100:.1f}%")

if grid_time > random_time:
    print(f"\nNote: RandomizedSearchCV was {grid_time/random_time:.1f}x faster than GridSearchCV")
    print(f"     with only {(accuracy_score(y_test, y_pred_random) - accuracy_score(y_test, y_pred_grid)) * 100:.2f}% accuracy difference")
    print("     → Use RandomizedSearchCV for large parameter spaces")
else:
    print(f"\nGridSearchCV was faster in this case")