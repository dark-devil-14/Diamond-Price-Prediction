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
- **Initial Data Exploration:** The dataset was initially explored to understand its structure, including its shape, data types, and statistical summary.
- **Data Cleaning:** Identified and removed 20 rows with zero values in dimensions (x, y, z) which are physically impossible.
- **Feature Engineering:** Converted categorical features into numerical values using **Ordinal Encoding** to preserve the quality hierarchy:
    - **Cut:** Fair (1) to Ideal (5)
    - **Color:** J (1) to D (7)
    - **Clarity:** I1 (1) to IF (8)
- **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets to prepare for model training and evaluation.

## 🔍 Key Insights from EDA
- **The Power of Carat:** Carat weight has a **0.92 correlation** with price, making it the strongest predictor.
- **Multicollinearity:** Dimensions (x, y, z) are near-perfectly correlated with Carat (up to 0.98), suggesting potential redundancy in linear models.
- **The Quality Paradox:** Lower-grade diamonds (e.g., SI2 clarity or J color) sometimes show higher median prices because they are often larger in size, proving that Carat weight frequently outweighs individual quality metrics.

## 🤖 Model Training and Performance
Four different regression models were trained and evaluated to find the best performer. The following models were used with their respective parameters:

1.  **Linear Regression**: A standard linear regression model with default parameters.
2.  **AdaBoost Regressor**:
    *   `n_estimators`: 100
    *   `random_state`: 42
3.  **Gradient Boosting Regressor**:
    *   `n_estimators`: 100
    *   `learning_rate`: 0.1
    *   `max_depth`: 3
    *   `random_state`: 42
4.  **XGBoost Regressor**:
    *   `n_estimators`: 100
    *   `learning_rate`: 0.1
    *   `max_depth`: 3
    *   `random_state`: 42

### Model Performance Comparison
| Model | Test R² | Test RMSE | Test MAE |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.9764** | **$615.19** | **$340.23** |
| XGBoost | 0.9761 | $618.81 | $340.12 |
| Linear Regression | 0.9100 | $1,201.39 | $790.37 |
| AdaBoost | 0.8839 | $1,364.33 | $1,122.37 |

## ⚙️ Advanced Analysis

### Bias-Variance Tradeoff
A comparative analysis was performed using three levels of model complexity to understand the bias-variance tradeoff:
1.  **Simple Model (Underfit):** Used only the 'carat' feature. This model had high bias but was stable.
2.  **Medium Model (Balanced):** Included all features. This model provided the best generalization.
3.  **Complex Model (Overfit):** Used Polynomial Features (degree 3). It achieved a high training R² (0.977) but failed catastrophically on the test data (R²: -10,415), demonstrating the dangers of over-parameterization.

### Cross-Validation
Cross-validation was used to get a more reliable estimate of model performance. The Gradient Boosting model, for instance, achieved a mean R² of 0.976 with a low standard deviation across 5 folds, confirming its robust performance.

### Feature Importance
The feature coefficients from the linear models were analyzed to interpret the model. As expected, `carat` had the largest positive impact on the price.

## � Real-World Problems Solved

   1. **Fair Market Valuation (Price Transparency):**
      For a typical consumer, the "4 Cs" (Carat, Cut, Color, Clarity) are difficult to balance manually. This model acts as an objective advisor, helping buyers ensure they aren't
  overpaying and sellers ensure they are pricing competitively based on 50,000+ historical data points.

   2. **Inventory Appraisal for Jewelers:**
      Jewelers and diamond traders can use this tool to rapidly value large inventories. By inputting physical dimensions ($x, y, z$) and quality metrics, they can automate what was
  previously a time-consuming manual expert appraisal process.

   3. **Identification of "Value Diamonds":**
      By analyzing feature importance (identifying that Carat and Dimensions are the primary drivers while Depth/Table are secondary), the project helps investors find diamonds that may
  have lower quality "on paper" but offer better physical presence and value for money.

   4. **Insurance and Risk Assessment:**
      Insurance companies can use these models to establish replacement values for jewelry. Because the model accounts for non-linear price jumps (e.g., the massive price increase as a
  stone crosses the 1.0ct threshold), it provides more accurate coverage estimates than simple linear math.

## �🚀 Technologies
- **Python 3.x**
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
- `Data/`: Contains the raw Kaggle dataset.
- `notebooks/`: Detailed Jupyter notebook with full analysis and training steps.
- `images/`: Visualizations of correlation heatmaps, model comparisons, and learning curves.
