import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*60)
print("SPAM EMAIL DETECTION USING DECISION TREE")
print("="*60)

# Generate MORE REALISTIC data with overlap between classes
n_samples = 500

# Spam emails (class 1)
spam_word = np.random.uniform(3, 12, 250)      # 3-12 suspicious words
spam_link = np.random.uniform(1, 8, 250)       # 1-8 links
spam_length = np.random.uniform(150, 450, 250) # 150-450 characters

# Normal emails (class 0) - WITH OVERLAP
normal_word = np.random.uniform(0, 5, 250)     # 0-5 words (overlap: 3-5)
normal_link = np.random.uniform(0, 3, 250)     # 0-3 links (overlap: 1-3)
normal_length = np.random.uniform(100, 500, 250) # 100-500 characters

# Combine features
X_spam = np.column_stack([spam_word, spam_link, spam_length])
X_normal = np.column_stack([normal_word, normal_link, normal_length])

X = np.vstack([X_spam, X_normal])
y = np.array([1]*250 + [0]*250)

# Add noise to make it harder
noise = np.random.normal(0, 0.5, X.shape)
X = X + noise

# Shuffle
indices = np.random.permutation(500)
X = X[indices]
y = y[indices]

print(f"Dataset shape: {X.shape}")
print(f"Features: word_count, link_count, email_length")
print(f"Class distribution: {np.sum(y)} Spam, {len(y)-np.sum(y)} Normal")
print()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# Test different depths
depths = range(1, 15)
train_acc = []
test_acc = []

print("TRAINING RESULTS")
print("-"*60)
print(f"{'Depth':<8} {'Train Acc':<12} {'Test Acc':<12} {'Status':<20}")
print("-"*60)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    
    train_accuracy = accuracy_score(y_train, dt.predict(X_train))
    test_accuracy = accuracy_score(y_test, dt.predict(X_test))
    
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    
    # Status determination
    if test_accuracy == max(test_acc):
        status = ">>> BEST SO FAR <<<"
    elif train_accuracy == 1.0 and test_accuracy < 0.95:
        status = "OVERFITTING"
    elif train_accuracy - test_accuracy > 0.05:
        status = "Possible overfitting"
    else:
        status = ""
    
    print(f"{d:<8} {train_accuracy:<12.3f} {test_accuracy:<12.3f} {status}")

# Find best depth
best_depth = depths[np.argmax(test_acc)]
best_test_acc = max(test_acc)
best_train_acc = train_acc[best_depth-1]

print("-"*60)
print(f"\nBEST DEPTH: {best_depth}")
print(f"Train accuracy: {best_train_acc:.3f}")
print(f"Test accuracy: {best_test_acc:.3f}")
print(f"Gap: {best_train_acc - best_test_acc:.3f}")

# Train final model
best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_tree.fit(X_train, y_train)
y_pred = best_tree.predict(X_test)

print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Spam']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize feature importance
feature_importance = best_tree.feature_importances_
features = ['Word Count', 'Link Count', 'Email Length']

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
for name, importance in zip(features, feature_importance):
    print(f"{name}: {importance:.3f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Accuracy vs Depth
axes[0, 0].plot(depths, train_acc, 'bo-', label='Train Accuracy', linewidth=2, markersize=6)
axes[0, 0].plot(depths, test_acc, 'ro-', label='Test Accuracy', linewidth=2, markersize=6)
axes[0, 0].axvline(x=best_depth, color='green', linestyle='--', linewidth=2, label=f'Best Depth = {best_depth}')
axes[0, 0].set_xlabel('Max Depth', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Accuracy vs Tree Depth', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(depths[::2])

# Plot 2: Overfitting gap
gap = np.array(train_acc) - np.array(test_acc)
axes[0, 1].bar(depths, gap, color='orange', alpha=0.7)
axes[0, 1].axhline(y=0.05, color='red', linestyle='--', label='Warning threshold (5%)')
axes[0, 1].set_xlabel('Max Depth', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Train-Test Gap', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Overfitting Detection', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
axes[1, 0].barh(features, feature_importance, color=['blue', 'green', 'red'], alpha=0.7)
axes[1, 0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Confusion Matrix heatmap
cm = confusion_matrix(y_test, y_pred)
im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(['Normal', 'Spam'])
axes[1, 1].set_yticklabels(['Normal', 'Spam'])
axes[1, 1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Actual', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Add text annotations to confusion matrix
for i in range(2):
    for j in range(2):
        text = axes[1, 1].text(j, i, cm[i, j],
                               ha="center", va="center",
                               color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.show()

# Conclusion
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if best_test_acc >= 0.95:
    print(f"Excellent model with {best_test_acc:.1%} accuracy")
elif best_test_acc >= 0.85:
    print(f"Good model with {best_test_acc:.1%} accuracy")
else:
    print(f"Model needs improvement ({best_test_acc:.1%} accuracy)")

if best_train_acc - best_test_acc > 0.1:
    print(f"WARNING: Severe overfitting detected (gap = {best_train_acc - best_test_acc:.1%})")
    print(f"Consider: reducing max_depth, increasing min_samples_split, or getting more data")
elif best_train_acc - best_test_acc > 0.05:
    print(f"CAUTION: Some overfitting detected (gap = {best_train_acc - best_test_acc:.1%})")
else:
    print(f"Good balance between train and test performance")

print(f"\nMost important feature: {features[np.argmax(feature_importance)]}")
print("="*60)