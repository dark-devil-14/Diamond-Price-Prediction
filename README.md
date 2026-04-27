# 💎 Diamond Price Prediction

Predict diamond prices using machine learning based on carat, cut, color, clarity, and physical dimensions.

## 📊 Dataset Overview
- **Source:** Kaggle Diamond Dataset
- **Size:** 53,940 entries
- **Target Variable:** `price` (USD)
- **Features:** 
    - **Primary:** Carat, x, y, z (dimensions)
    - **Categorical:** Cut, Color, Clarity
    - **Proportional:** Depth, Table

## 🛠️ Data Preprocessing & Engineering
- **Data Cleaning:** Identified and removed 20 rows with zero values in dimensions (x, y, z) which are physically impossible.
- **Feature Engineering:** Converted categorical features into numerical values using **Ordinal Encoding** to preserve the quality hierarchy:
    - **Cut:** Fair (1) to Ideal (5)
    - **Color:** J (1) to D (7)
    - **Clarity:** I1 (1) to IF (8)

## 🔍 Key Insights from EDA
- **The Power of Carat:** Carat weight has a **0.92 correlation** with price, making it the strongest predictor.
- **Multicollinearity:** Dimensions (x, y, z) are near-perfectly correlated with Carat (up to 0.98), suggesting potential redundancy in linear models.
- **The Quality Paradox:** Lower-grade diamonds (e.g., SI2 clarity or J color) sometimes show higher median prices because they are often larger in size, proving that Carat weight frequently outweighs individual quality metrics.

## 🤖 Model Performance Comparison
| Model | Test R² | Test RMSE | Test MAE |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.9764** | **$615.19** | **$340.23** |
| XGBoost | 0.9761 | $618.81 | $340.12 |
| Linear Regression | 0.9100 | $1,201.39 | $790.37 |
| AdaBoost | 0.8839 | $1,364.33 | $1,122.37 |

## ⚙️ Advanced Model Optimization
To find the optimal balance between performance and simplicity, hyperparameter tuning was performed using **GridSearchCV** and **RandomizedSearchCV** for regularized linear models:

### 1. Regularization Results (Tuned)
| Model | Best Alpha | Best L1 Ratio | Test R² | Test RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **ElasticNet** | 0.01 | 0.5 | **0.9125** | **$1,184.72** |
| Lasso | 10.00 | 1.0 (L1) | 0.9113 | $1,192.32 |
| Ridge | 10.00 | 0.0 (L2) | 0.9105 | $1,198.08 |

### 2. Automatic Feature Selection (Lasso)
Using **Lasso Regression (L1 Penalty)** with an optimal alpha of 10, the model successfully identified and removed redundant features:
- ✅ **Features Kept:** Carat, Cut, Color, Clarity, Depth, Table, Z.
- ❌ **Features Removed:** **X** and **Y** dimensions (removed due to near-perfect correlation with Carat).
- **Insight:** This proves that the model can simplify itself by identifying multicollinearity without losing significant accuracy.

## 🧪 Advanced Analysis: Bias-Variance Tradeoff
Performed a comparative analysis using three levels of complexity:
1.  **Simple Model (Underfit):** Used only 'carat'. High bias, but stable.
2.  **Medium Model (Balanced):** All features included. Best generalization.
3.  **Complex Model (Overfit):** Polynomial Features (degree 3). Achieved high training R² (0.977) but failed catastrophically on test data (R²: -10,415), demonstrating the dangers of over-parameterization.

## 🚀 Technologies
- **Python 3.x**
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
- `Data/`: Contains the raw Kaggle dataset.
- `notebooks/`: Detailed Jupyter notebook with full analysis and training steps.
- `images/`: Visualizations of correlation heatmaps, model comparisons, and learning curves.
