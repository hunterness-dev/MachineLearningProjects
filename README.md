# Machine Learning Projects 🚀

## 📁 Projects

### 1. Diabetes Regression Analysis 📊
Predict diabetes progression using BMI, blood pressure, and age with linear regression.
- **Features:** Coefficient interpretation (standardized vs raw), comprehensive visualization
- **Plots:** Actual vs predicted, residuals, feature importance, Q-Q plot, correlation matrix
- **Libraries:** scikit-learn, pandas, matplotlib, seaborn, scipy

### 2. Polynomial Regression GUI 🎨
Interactive desktop application for visualizing polynomial regression and overfitting.
- **Features:** Choose degree 1-8, scroll through models, compare all at once
- **Key insight:** Visualize bias-variance tradeoff in real-time
- **Libraries:** scikit-learn, PyQt5, matplotlib

### 3. KNN Spam Detection with GUI 🎯

Desktop application for spam email detection using K-Nearest Neighbors.

- **Features:** Interactive K selection (1-30), decision boundary visualization
- **Metrics:** Confusion matrix, accuracy vs K plot, real-time prediction
- **Libraries:** scikit-learn (KNN), PyQt5, matplotlib

### 4. Ridge vs Lasso Comparison 📉

Compare L1 (Lasso) and L2 (Ridge) regularization for polynomial regression.

- **Purpose:** Prevent overfitting in high-degree polynomial models (degree 15)
- **Key insight:** Lasso automatically selects important features (reduced 15 → 5 features)
- **Results:** Lasso (R²=0.959) slightly better than Ridge (R²=0.956)
- **Libraries:** scikit-learn (Ridge, Lasso, PolynomialFeatures), matplotlib

## 🛠️ Installation

```bash
pip install -r requirements.txt