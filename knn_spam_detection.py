import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                             QHBoxLayout, QScrollArea, QLabel, QComboBox, QPushButton)
from PyQt5.QtCore import Qt
import warnings
warnings.filterwarnings('ignore')

# ========== DATA GENERATION ==========
np.random.seed(42)

# 250 spam samples (class 1)
spam_word = np.random.uniform(3, 9, 250)
spam_link = np.random.uniform(2, 7, 250)
spam_length = np.random.uniform(200, 450, 250)

# 250 normal samples (class 0)
normal_word = np.random.uniform(2, 8, 250)
normal_link = np.random.uniform(1, 6, 250)
normal_length = np.random.uniform(150, 400, 250)

# Adding more noise
noise_word = np.random.normal(0, 0.8, 250)
noise_link = np.random.normal(0, 0.8, 250)

spam_word = spam_word + noise_word
normal_word = normal_word + noise_word

# Combine data
X_spam = np.column_stack([spam_word, spam_link, spam_length])
X_normal = np.column_stack([normal_word, normal_link, normal_length])

X = np.vstack([X_spam, X_normal])
y = np.array([1]*250 + [0]*250)

# Shuffle the data
indices = np.random.permutation(500)
X = X[indices]
y = y[indices]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Spam count:", sum(y))
print("Normal count:", len(y) - sum(y))

# ========== TRAIN TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== KNN WITH DIFFERENT K VALUES ==========
k_values = [1, 3, 5, 7, 10, 15, 20, 25, 30]
train_accuracies = []
test_accuracies = []

print("\n" + "="*60)
print("KNN CLASSIFIER PERFORMANCE")
print("="*60)
print(f"{'k':<8} {'Train Accuracy':<18} {'Test Accuracy':<18}")
print("-"*60)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f"{k:<8} {train_acc:<18.4f} {test_acc:<18.4f}")

# ========== FIND BEST K ==========
best_k = k_values[test_accuracies.index(max(test_accuracies))]
best_accuracy = max(test_accuracies)

print("-"*60)
print(f"\nBEST K = {best_k} with Test Accuracy = {best_accuracy:.4f}")
print("="*60)

# ========== TRAIN BEST MODEL ==========
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX:")
print(cm)
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Spam']))

# ========== PREDICT NEW EMAIL ==========
new_email = [[8, 5, 300]]
prediction = best_model.predict(new_email)

print("\n" + "="*60)
print("NEW EMAIL PREDICTION")
print("="*60)
print(f"Features: [word_count=8, link_count=5, email_length=300]")

if prediction[0] == 1:
    print("Result: SPAM Email")
else:
    print("Result: NORMAL Email")
print("="*60)

# ========== SCROLLABLE PLOT GUI ==========
class ScrollableKNNPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KNN Classifier Analysis - Spam Detection")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QHBoxLayout()
        control_panel.addWidget(QLabel("Select K Value:"))
        self.k_combo = QComboBox()
        for k in k_values:
            self.k_combo.addItem(f"k = {k} (Accuracy: {test_accuracies[k_values.index(k)]:.3f})")
        self.k_combo.currentIndexChanged.connect(self.update_plot)
        control_panel.addWidget(self.k_combo)
        control_panel.addStretch()
        
        self.show_comparison_btn = QPushButton("Show K Comparison")
        self.show_comparison_btn.clicked.connect(self.show_comparison_plot)
        control_panel.addWidget(self.show_comparison_btn)
        
        self.show_scroll_btn = QPushButton("Show Scrollable View")
        self.show_scroll_btn.clicked.connect(self.show_scrollable_plots)
        control_panel.addWidget(self.show_scroll_btn)
        
        main_layout.addLayout(control_panel)
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        
        main_layout.addWidget(self.scroll_area)
        
        self.current_figure = None
        self.update_plot()
    
    def create_knn_plot(self, k):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Train the model for this specific k
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Plot 1: Decision Boundary (using first 2 features)
        ax1 = axes[0, 0]
        h = 0.5
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape[0])])
        Z = Z.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        scatter = ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                              edgecolors='black', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Word Count', fontsize=11)
        ax1.set_ylabel('Link Count', fontsize=11)
        ax1.set_title(f'Decision Boundary (k={k})\nAccuracy: {accuracy:.3f}', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs K
        ax2 = axes[0, 1]
        ax2.plot(k_values, train_accuracies, 'bo-', linewidth=2, markersize=8, label='Train Accuracy')
        ax2.plot(k_values, test_accuracies, 'ro-', linewidth=2, markersize=8, label='Test Accuracy')
        ax2.axvline(x=k, color='g', linestyle='--', linewidth=2, alpha=0.7, label=f'k={k}')
        ax2.set_xlabel('K Value', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Accuracy vs K Value', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix
        ax3 = axes[1, 0]
        cm = confusion_matrix(y_test, y_pred)
        im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Normal', 'Spam'])
        ax3.set_yticklabels(['Normal', 'Spam'])
        ax3.set_xlabel('Predicted', fontsize=11)
        ax3.set_ylabel('Actual', fontsize=11)
        ax3.set_title(f'Confusion Matrix (k={k})', fontsize=11, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, cm[i, j], ha='center', va='center', fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Feature Distribution
        ax4 = axes[1, 1]
        spam_idx = y == 1
        normal_idx = y == 0
        
        ax4.scatter(X[spam_idx, 0], X[spam_idx, 1], alpha=0.5, label='Spam', color='red', s=30)
        ax4.scatter(X[normal_idx, 0], X[normal_idx, 1], alpha=0.5, label='Normal', color='blue', s=30)
        
        # Highlight new email
        ax4.scatter([8], [5], color='green', s=200, marker='*', 
                   edgecolors='black', linewidth=2, label='New Email', zorder=5)
        ax4.set_xlabel('Word Count', fontsize=11)
        ax4.set_ylabel('Link Count', fontsize=11)
        ax4.set_title('Feature Distribution (Word vs Link)', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'KNN Classifier Analysis - k = {k}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def update_plot(self):
        idx = self.k_combo.currentIndex()
        if idx < 0:
            return
        k = k_values[idx]
        
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        fig = self.create_knn_plot(k)
        canvas = FigureCanvas(fig)
        self.scroll_layout.addWidget(canvas)
        
        toolbar = NavigationToolbar(canvas, self)
        self.scroll_layout.addWidget(toolbar)
        
        self.current_figure = fig
    
    def show_comparison_plot(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_values, train_accuracies, 'bo-', linewidth=2, markersize=8, label='Train Accuracy', markerfacecolor='blue')
        ax.plot(k_values, test_accuracies, 'ro-', linewidth=2, markersize=8, label='Test Accuracy', markerfacecolor='red')
        ax.axvline(x=best_k, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'Best k = {best_k}')
        ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('KNN Performance: Training vs Testing Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        # Add value labels
        for i, (k, test_acc) in enumerate(zip(k_values, test_accuracies)):
            ax.annotate(f'{test_acc:.3f}', (k, test_acc), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        self.scroll_layout.addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self)
        self.scroll_layout.addWidget(toolbar)
        self.current_figure = fig
    
    def show_scrollable_plots(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        for k in k_values:
            fig, ax = plt.subplots(figsize=(7, 6))
            
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            h = 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape[0])])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                      edgecolors='black', linewidth=0.5, alpha=0.7)
            ax.scatter([8], [5], color='green', s=150, marker='*', 
                      edgecolors='black', linewidth=2, label='New Email', zorder=5)
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Link Count')
            ax.set_title(f'k = {k} | Accuracy = {accuracy:.4f}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            canvas = FigureCanvas(fig)
            self.scroll_layout.addWidget(canvas)
            toolbar = NavigationToolbar(canvas, self)
            self.scroll_layout.addWidget(toolbar)

# ========== RUN THE GUI ==========
app = QApplication(sys.argv)
window = ScrollableKNNPlot()
window.show()
sys.exit(app.exec_())