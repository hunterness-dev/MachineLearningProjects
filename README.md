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

### 4. Polynomial Degree Selection with Cross-Validation 🔍
Select optimal polynomial degree using cross-validation to prevent overfitting.
- **Dataset:** Diabetes (BMI vs Disease Progression)
- **Key insight:** Higher degrees (8-10) cause severe overfitting (RMSE increases from 62 to 76)
- **Methods:** 5-Fold CV, 95% confidence intervals, PolynomialFeatures
- **Visualization:** RMSE vs Degree with error bars, best model fit plot
- **Result:** Degree 1 (linear) is best for BMI-Diabetes relationship

### 5. Ridge vs Lasso Comparison 📉
Compare L1 (Lasso) and L2 (Ridge) regularization for polynomial regression.
- **Purpose:** Prevent overfitting in high-degree polynomial models (degree 15)
- **Key insight:** Lasso automatically selects important features (reduced 15 → 5 features)
- **Results:** Lasso (R²=0.959) slightly better than Ridge (R²=0.956)
- **Libraries:** scikit-learn (Ridge, Lasso, PolynomialFeatures), matplotlib

## 📊 Results Summary

| Project | Best Model | Key Metric | Insight |
|---------|-----------|------------|---------|
| Diabetes Regression | Linear Regression | R² = 0.96 | BMI, BP, age explain 96% variance |
| Polynomial GUI | Degree 3 (true function) | RMSE = 0.30 | Visualize overfitting at degree 8+ |
| KNN Spam | K=5 | Accuracy = 100% | Simple data, perfect separation |
| Polynomial CV Selection | Degree 1 (Linear) | RMSE = 62.46 | Higher degrees cause overfitting |
| Ridge vs Lasso | Lasso (α=0.01) | R² = 0.959 | Lasso reduced 15→5 features |

## 🛠️ Installation

```bash
pip install -r requirements.txt