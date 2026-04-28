import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                             QHBoxLayout, QScrollArea, QLabel, QComboBox, QPushButton)
from PyQt5.QtCore import Qt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ========== GENERATE DATA ==========
np.random.seed(42)
x = np.random.uniform(-2, 2, 200)
y = 1.5 + 0.8 * x + 1.2 * x**2 - 0.5 * x**3 + np.random.normal(0, 0.3, 200)

df = pd.DataFrame({'x': x, 'y': y})
X = df[['x']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== TRAIN MODELS AND STORE SCORES ==========
models = {}
poly_features = {}
train_scores = {}
test_scores = {}

for degree in range(1, 9):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    models[degree] = model
    poly_features[degree] = poly
    train_scores[degree] = {'R²': train_r2, 'MSE': train_mse, 'RMSE': train_rmse}
    test_scores[degree] = {'R²': test_r2, 'MSE': test_mse, 'RMSE': test_rmse}

# ========== PRINT RESULTS ==========
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY (Degree 1 to 8)")
print("="*80)
print(f"{'Degree':<8} {'Train R²':<12} {'Test R²':<12} {'Difference':<12} {'Status':<15}")
print("-"*80)

best_degree = 1
best_test_r2 = -1

for degree in range(1, 9):
    train_r2 = train_scores[degree]['R²']
    test_r2 = test_scores[degree]['R²']
    diff = abs(train_r2 - test_r2)
    
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_degree = degree
    
    if degree == 3:
        status = "BEST (True Function)"
    elif diff < 0.03:
        status = "Excellent"
    elif diff < 0.08:
        status = "Good"
    elif diff < 0.15:
        status = "Overfitting"
    else:
        status = "Severe Overfitting"
    
    print(f"{degree:<8} {train_r2:<12.4f} {test_r2:<12.4f} {diff:<12.4f} {status:<15}")

print("="*80)
print(f"\nBEST MODEL: Degree {best_degree}")
print(f"Test R²: {best_test_r2:.4f}")
print("="*80)

# ========== SCROLLABLE PLOT WINDOW ==========
class ScrollablePlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polynomial Regression - Degrees 1 to 8")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        control_panel = QHBoxLayout()
        control_panel.addWidget(QLabel("Select Model Degree:"))
        self.degree_combo = QComboBox()
        for d in range(1, 9):
            self.degree_combo.addItem(f"Degree {d}")
        self.degree_combo.currentIndexChanged.connect(self.update_plot)
        control_panel.addWidget(self.degree_combo)
        control_panel.addStretch()
        
        self.show_all_btn = QPushButton("Show All Models (Grid)")
        self.show_all_btn.clicked.connect(self.show_all_plots)
        control_panel.addWidget(self.show_all_btn)
        
        self.show_scroll_btn = QPushButton("Show Scrollable View")
        self.show_scroll_btn.clicked.connect(self.show_scrollable_plots)
        control_panel.addWidget(self.show_scroll_btn)
        
        main_layout.addLayout(control_panel)
        
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
    
    def create_plot_for_degree(self, degree):
        fig, ax = plt.subplots(figsize=(9, 7))
        
        X_smooth = pd.DataFrame(np.linspace(-2, 2, 200), columns=['x'])
        y_true = 1.5 + 0.8*X_smooth.values.ravel() + 1.2*X_smooth.values.ravel()**2 - 0.5*X_smooth.values.ravel()**3
        
        poly = poly_features[degree]
        model = models[degree]
        X_smooth_poly = poly.transform(X_smooth)
        y_pred_smooth = model.predict(X_smooth_poly)
        
        ax.scatter(X_train, y_train, alpha=0.3, label='Train Data', s=20, color='blue')
        ax.scatter(X_test, y_test, alpha=0.6, label='Test Data', s=30, color='red')
        ax.plot(X_smooth, y_pred_smooth, 'g-', linewidth=2, label=f'Degree {degree} Model')
        ax.plot(X_smooth, y_true, 'k--', linewidth=2, label='True Function', alpha=0.7)
        
        r2 = test_scores[degree]['R²']
        mse = test_scores[degree]['MSE']
        rmse = test_scores[degree]['RMSE']
        
        ax.set_title(f'Polynomial Regression - Degree {degree}\nTest R² = {r2:.4f} | MSE = {mse:.4f} | RMSE = {rmse:.4f}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2, 6)
        
        plt.tight_layout()
        return fig
    
    def update_plot(self):
        degree = self.degree_combo.currentIndex() + 1
        
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        fig = self.create_plot_for_degree(degree)
        canvas = FigureCanvas(fig)
        self.scroll_layout.addWidget(canvas)
        
        toolbar = NavigationToolbar(canvas, self)
        self.scroll_layout.addWidget(toolbar)
        
        self.current_figure = fig
    
    def show_all_plots(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        X_smooth = pd.DataFrame(np.linspace(-2, 2, 200), columns=['x'])
        y_true = 1.5 + 0.8*X_smooth.values.ravel() + 1.2*X_smooth.values.ravel()**2 - 0.5*X_smooth.values.ravel()**3
        
        for idx, degree in enumerate(range(1, 9)):
            poly = poly_features[degree]
            model = models[degree]
            X_smooth_poly = poly.transform(X_smooth)
            y_pred_smooth = model.predict(X_smooth_poly)
            
            ax = axes[idx]
            ax.scatter(X_train, y_train, alpha=0.3, s=15, color='blue')
            ax.scatter(X_test, y_test, alpha=0.6, s=25, color='red')
            ax.plot(X_smooth, y_pred_smooth, 'g-', linewidth=2)
            ax.plot(X_smooth, y_true, 'k--', linewidth=1.5, alpha=0.5)
            
            r2 = test_scores[degree]['R²']
            ax.set_title(f'Degree {degree} (R²={r2:.3f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2, 6)
        
        plt.suptitle('Polynomial Regression Models Comparison (Degree 1-8)', fontsize=14, fontweight='bold')
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
        
        X_smooth = pd.DataFrame(np.linspace(-2, 2, 200), columns=['x'])
        y_true = 1.5 + 0.8*X_smooth.values.ravel() + 1.2*X_smooth.values.ravel()**2 - 0.5*X_smooth.values.ravel()**3
        
        for degree in range(1, 9):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            poly = poly_features[degree]
            model = models[degree]
            X_smooth_poly = poly.transform(X_smooth)
            y_pred_smooth = model.predict(X_smooth_poly)
            
            ax.scatter(X_train, y_train, alpha=0.3, s=15, color='blue', label='Train')
            ax.scatter(X_test, y_test, alpha=0.6, s=25, color='red', label='Test')
            ax.plot(X_smooth, y_pred_smooth, 'g-', linewidth=2, label='Model')
            ax.plot(X_smooth, y_true, 'k--', linewidth=1.5, alpha=0.5, label='True')
            
            r2 = test_scores[degree]['R²']
            mse = test_scores[degree]['MSE']
            
            ax.set_title(f'Degree {degree}\nR²={r2:.4f}, MSE={mse:.4f}', fontsize=11, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2, 6)
            
            plt.tight_layout()
            
            canvas = FigureCanvas(fig)
            self.scroll_layout.addWidget(canvas)
            toolbar = NavigationToolbar(canvas, self)
            self.scroll_layout.addWidget(toolbar)

# ========== RUN APPLICATION ==========
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScrollablePlotWindow()
    window.show()
    sys.exit(app.exec_())