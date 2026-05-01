import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')



np.random.seed(42)

print("="*70)
print("SPAM DETECTION USING NEURAL NETWORK (Keras/TensorFlow)")
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
# BUILD NEURAL NETWORK MODEL
# ============================================
print("="*70)
print("BUILDING NEURAL NETWORK MODEL")
print("="*70)

model = Sequential([
    Dense(32, activation='relu', input_shape=(3,), name='hidden_1'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu', name='hidden_2'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(8, activation='relu', name='hidden_3'),
    Dropout(0.1),
    Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

model.summary()

# ============================================
# CALLBACKS
# ============================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=0
)

# ============================================
# TRAIN THE MODEL
# ============================================
print("\n" + "="*70)
print("TRAINING NEURAL NETWORK")
print("="*70)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================
# EVALUATE THE MODEL
# ============================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
print(f"Test AUC: {test_auc:.4f}")

print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Spam']))

# ============================================
# PREDICT FOR 3 NEW SAMPLES
# ============================================
print("\n" + "="*70)
print("PREDICTION FOR 3 NEW SAMPLES")
print("="*70)

# Create 3 new samples (different from training data)
new_samples = np.array([
    [5.0, 3.0, 250],   # Sample 1: Medium word count, medium links, medium length
    [2.0, 0.5, 150],   # Sample 2: Low word count, few links, short length (likely Normal)
    [8.0, 5.0, 350]    # Sample 3: High word count, many links, long length (likely Spam)
])

# Add some noise to make it realistic
new_samples = new_samples + np.random.normal(0, 0.5, new_samples.shape)

# Predict
new_predictions_proba = model.predict(new_samples)
new_predictions = (new_predictions_proba > 0.5).astype(int)

print("\nPrediction Results:")
print("-"*60)
print(f"{'Sample':<10} {'Word Count':<12} {'Link Count':<12} {'Length':<10} {'Prob(Spam)':<12} {'Prediction':<12}")
print("-"*60)

for i, (sample, prob, pred) in enumerate(zip(new_samples, new_predictions_proba, new_predictions)):
    status = "SPAM" if pred[0] == 1 else "NORMAL"
    prob_percent = prob[0] * 100
    print(f"{i+1:<10} {sample[0]:<12.2f} {sample[1]:<12.2f} {sample[2]:<10.0f} {prob_percent:<11.1f}% {status:<12}")

print("-"*60)

# ============================================
# CREATE 2x3 GRID OF PLOTS (6 PLOTS)
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Training & Validation Accuracy (Row 1, Col 1)
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Accuracy over Epochs', fontsize=13, fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 2. Training & Validation Loss (Row 1, Col 2)
axes[0, 1].plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Loss', fontsize=11)
axes[0, 1].set_title('Loss over Epochs', fontsize=13, fontweight='bold')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# 3. ROC Curve with AUC (Row 1, Col 3) - NEW
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[0, 2].plot(fpr, tpr, color='green', lw=2, label=f'Neural Network (AUC = {roc_auc:.3f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC = 0.500)')
axes[0, 2].fill_between(fpr, tpr, alpha=0.2, color='green')
axes[0, 2].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 2].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 2].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(True, alpha=0.3)

# 4. Confusion Matrix Heatmap (Row 2, Col 1)
axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Normal', 'Spam'])
axes[1, 0].set_yticklabels(['Normal', 'Spam'])
axes[1, 0].set_xlabel('Predicted', fontsize=11)
axes[1, 0].set_ylabel('Actual', fontsize=11)
axes[1, 0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)

# 5. Prediction Probability Distribution (Row 2, Col 2)
axes[1, 1].hist(y_pred_proba[y_test==1], alpha=0.5, label='Actual Spam', color='red', bins=15, edgecolor='black')
axes[1, 1].hist(y_pred_proba[y_test==0], alpha=0.5, label='Actual Normal', color='blue', bins=15, edgecolor='black')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold = 0.5')
axes[1, 1].set_xlabel('Predicted Probability of being Spam', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Prediction Probability Distribution', fontsize=13, fontweight='bold')
axes[1, 1].legend(loc='upper center')
axes[1, 1].grid(True, alpha=0.3)

# 6. New Sample Predictions Bar Chart (Row 2, Col 3)
sample_names = ['Sample 1\n(Medium)', 'Sample 2\n(Low)', 'Sample 3\n(High)']
spam_probs = [new_predictions_proba[i][0] * 100 for i in range(3)]
colors = ['orange' if prob > 50 else 'skyblue' for prob in spam_probs]
bars = axes[1, 2].bar(sample_names, spam_probs, color=colors, edgecolor='black', linewidth=1.5)
axes[1, 2].axhline(y=50, color='red', linestyle='--', linewidth=2, label='Spam Threshold (50%)')
axes[1, 2].set_ylim([0, 100])
axes[1, 2].set_ylabel('Probability of being SPAM (%)', fontsize=11)
axes[1, 2].set_title('New Sample Predictions', fontsize=13, fontweight='bold')
axes[1, 2].legend(loc='upper right')
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, prob in zip(bars, spam_probs):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add prediction text
    status = "SPAM" if prob > 50 else "NORMAL"
    color = 'red' if prob > 50 else 'green'
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 15,
                   status, ha='center', va='bottom', fontweight='bold', 
                   fontsize=11, color=color)

plt.suptitle('Neural Network: Training, Evaluation, and New Predictions', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ============================================
# COMPARE WITH RANDOM FOREST
# ============================================
print("\n" + "="*70)
print("COMPARISON WITH RANDOM FOREST")
print("="*70)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\n{'Model':<20} {'Accuracy':<15} {'F1-Score':<15} {'AUC':<12}")
print("-"*62)
print(f"{'Neural Network':<20} {test_accuracy:<15.4f} {f1_score(y_test, y_pred):<15.4f} {test_auc:<12.4f}")
print(f"{'Random Forest':<20} {rf_acc:<15.4f} {rf_f1:<15.4f} {rf_f1:<12.4f}")

if test_accuracy > rf_acc:
    print(f"\n✓ Neural Network performs {((test_accuracy - rf_acc)/rf_acc)*100:.1f}% better than Random Forest")
elif rf_acc > test_accuracy:
    print(f"\n✓ Random Forest performs {((rf_acc - test_accuracy)/test_accuracy)*100:.1f}% better than Neural Network")
else:
    print(f"\n✓ Both models perform similarly")

# ============================================
# DETAILED PREDICTION TABLE FOR NEW SAMPLES
# ============================================
print("\n" + "="*70)
print("DETAILED ANALYSIS OF NEW SAMPLES")
print("="*70)

for i, (sample, prob, pred) in enumerate(zip(new_samples, new_predictions_proba, new_predictions)):
    print(f"\nSample {i+1}:")
    print(f"  - Word Count: {sample[0]:.2f}")
    print(f"  - Link Count: {sample[1]:.2f}")
    print(f"  - Email Length: {sample[2]:.0f}")
    print(f"  - Probability of being SPAM: {prob[0]*100:.1f}%")
    print(f"  - Classification: {'SPAM' if pred[0] == 1 else 'NORMAL'}")
    
    # Interpretation
    if prob[0] > 0.7:
        print(f"  - Confidence: HIGH (Very likely to be SPAM)")
    elif prob[0] > 0.5:
        print(f"  - Confidence: MODERATE (Likely SPAM, but uncertain)")
    elif prob[0] > 0.3:
        print(f"  - Confidence: MODERATE (Likely NORMAL, but uncertain)")
    else:
        print(f"  - Confidence: HIGH (Very likely to be NORMAL)")

print("\n" + "="*70)