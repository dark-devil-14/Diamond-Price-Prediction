# 💎 Diamond Price Prediction

Predict diamond prices using machine learning based on carat, cut, color, clarity, and physical dimensions.

## 📊 Dataset
- **Source:** Kaggle Diamond Dataset
- **Size:** 54,000 diamonds
- **Features:** Carat, Cut, Color, Clarity, Depth, Table, X, Y, Z
- **Target:** Price (USD)

## 🤖 Models Trained
| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Linear Regression | ~0.88 | ~$1,200 |
| AdaBoost | ~0.95 | ~$800 |
| **Gradient Boosting** | **0.976** | **$615** ✅ |
| XGBoost | ~0.978 | ~$600 |

## 🎯 Best Model
**Gradient Boosting Regressor** achieved 97.6% accuracy with lowest RMSE.

## 🛠️ Technologies
- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Diamond_price_prediction.ipynb
