# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💎 Diamond Price Predictor")
st.markdown("Predict diamond prices using machine learning based on carat, cut, color, clarity, and physical dimensions.")

import os

# Get absolute path to the directory containing app.py (deploy/)
current_dir = os.path.dirname(os.path.abspath(__file__))

st.sidebar.header("Select Prediction Model")
model_choice = st.sidebar.selectbox(
    "AI Model",
    ["XGBoost (Optuna Tuned)", "Gradient Boosting", "AdaBoost", "Linear Regression"]
)

model_info = {
    "XGBoost (Optuna Tuned)": {
        "filename": "xgboost_model.pkl",
        "description": "🏆 **Best Accuracy** (R² ~ 98.1%)\n\nUses gradient-boosted decision trees tuned with Optuna. Captures non-linear relationships extremely well.",
    },
    "Gradient Boosting": {
        "filename": "gradient_boosting_model.pkl",
        "description": "📈 **High Accuracy** (R² ~ 97.4%)\n\nStandard gradient boosting ensemble. Very strong performance, slightly less optimized than the tuned XGBoost model.",
    },
    "AdaBoost": {
        "filename": "adaboost_model.pkl",
        "description": "⚠️ **Moderate Accuracy** (R² ~ 88.2%)\n\nAdaptive boosting ensemble. Tends to oversimplify predictions and can behave erratically on extreme outliers.",
    },
    "Linear Regression": {
        "filename": "linear_regression_model.pkl",
        "description": "📉 **Lowest Accuracy** (R² ~ 92.0%)\n\nFast and simple, but because diamond prices scale exponentially with carat size, a straight line model has high errors.",
    }
}

selected_info = model_info[model_choice]
st.sidebar.info(selected_info["description"])

# Resolve paths dynamically relative to this file
model_path = os.path.join(current_dir, f"../models/{selected_info['filename']}")
preprocessor_path = os.path.join(current_dir, "../models/preprocessor.pkl")

# Load the trained model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
# Initialize session state for diamond features if not already present
default_specs = {
    "carat": 1.0,
    "cut": "Ideal",
    "color": "E",
    "clarity": "VS2",
    "depth": 61.5,
    "table": 57.0,
    "x": 6.4,
    "y": 6.4,
    "z": 3.9
}

for key, val in default_specs.items():
    if key not in st.session_state:
        st.session_state[key] = val


def check_physical_validity(carat, depth, table, x, y, z):
    warnings = []
    
    # 1. Zero or negative dimensions
    if x <= 0 or y <= 0 or z <= 0:
        warnings.append("❌ **Zero Dimensions**: A diamond cannot have a length (X), width (Y), or depth (Z) of 0 mm.")
        return warnings, True  # Critical error, block prediction
        
    # 2. Round Cut Symmetry (x and y should be close)
    xy_diff_pct = abs(x - y) / max(x, y) * 100
    if xy_diff_pct > 10:
        warnings.append(f"⚠️ **Asymmetry Warning**: Length (X={x}mm) and Width (Y={y}mm) differ by {xy_diff_pct:.1f}%. Round brilliant cuts should be nearly symmetrical (within 10%).")
        
    # 3. Depth Percentage Mismatch
    # depth = z / mean(x, y) * 100
    calculated_depth = (z / ((x + y) / 2)) * 100 if (x + y) > 0 else 0
    depth_diff = abs(depth - calculated_depth)
    if depth_diff > 3.0:
        warnings.append(f"⚠️ **Depth Contradiction**: Slider is set to **{depth}%** depth, but physical dimensions (X, Y, Z) calculate to **{calculated_depth:.1f}%** depth. This mathematical mismatch confuses the AI model.")
        
    # 4. Density/Volume consistency
    box_volume = x * y * z
    if carat > 0:
        ratio = box_volume / carat
        # Real diamonds have bounding box volume / carat ratio roughly between 60 and 140
        if ratio < 50:
            warnings.append(f"⚠️ **Density Mismatch (Too Heavy)**: A {carat}-carat diamond is physically too heavy for these tiny dimensions. A real diamond of this size would weigh less.")
        elif ratio > 170:
            warnings.append(f"⚠️ **Density Mismatch (Too Light)**: A {carat}-carat diamond is physically too light for these large dimensions. A real diamond of this size would weigh much more.")
            
    return warnings, False

st.sidebar.header("Enter Diamond Specifications")

# Add a reset/refresh button to set values to a standard valid diamond
if st.sidebar.button("⚙️ Reset to Valid Specs", use_container_width=True):
    for key, val in default_specs.items():
        st.session_state[key] = val
    st.rerun()

carat = st.sidebar.slider("Carat (Weight)", 0.1, 5.0, key="carat", step=0.01)
cut = st.sidebar.selectbox("Cut (Quality)", ["Fair", "Good", "Very Good", "Premium", "Ideal"], key="cut")
color = st.sidebar.selectbox("Color (Grade)", ["J", "I", "H", "G", "F", "E", "D"], key="color")
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], key="clarity")
depth = st.sidebar.slider("Depth (%)", 40.0, 80.0, key="depth", step=0.1)
table = st.sidebar.slider("Table (%)", 40.0, 100.0, key="table", step=0.1)
x = st.sidebar.slider("Length (X) in mm", 0.0, 15.0, key="x", step=0.01)
y = st.sidebar.slider("Width (Y) in mm", 0.0, 15.0, key="y", step=0.01)
z = st.sidebar.slider("Depth (Z) in mm", 0.0, 10.0, key="z", step=0.01)

# Create a dataframe from the input values
input_df = pd.DataFrame({
    "carat": [carat],
    "cut": [cut],
    "color": [color],
    "clarity": [clarity],
    "depth": [depth],
    "table": [table],
    "x": [x],
    "y": [y],
    "z": [z]
})

# Run validation checks
warnings_list, is_critical = check_physical_validity(carat, depth, table, x, y, z)

if is_critical:
    st.error(warnings_list[0])
else:
    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)
    # Predict the price using the trained XGBoost model
    predicted_price = model.predict(processed_data)
    pred_val = predicted_price[0]
    
    # Display Prediction
    if pred_val > 0:
        st.success(f"### Predicted Diamond Price: ${pred_val:,.2f}")
        
        # Format Google and Blue Nile search links dynamically with all specs (including numerical dimensions)
        search_query = f"{carat} carat {cut} cut {color} color {clarity} clarity {depth}% depth {table}% table {x}x{y}x{z} mm, natural diamond price"
        google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        blue_nile_url = f"https://www.bluenile.com/diamond-search?carats={carat}&cuts={cut.upper()}&colors={color}&clarities={clarity}&depthMin={depth}&depthMax={depth}&tableMin={table}&tableMax={table}"
        
        # Display Live Market Reference
        st.markdown("### 🌐 Live Market References")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Estimated Retail Price Ranges:**")
            st.markdown(f"- **Natural Diamond (Retail):** \${pred_val * 0.90:,.2f} - \${pred_val * 1.15:,.2f}")
            st.markdown(f"- **Lab-Grown Diamond:** \${pred_val * 0.15:,.2f} - \${pred_val * 0.25:,.2f} *(~80% cheaper)*")
            
        with col2:
            st.markdown("**Compare Live Pricing:**")
            st.link_button("🔍 Search on Google", google_url, use_container_width=True)
            st.link_button("💎 Compare on Blue Nile", blue_nile_url, use_container_width=True)
            
    else:
        st.error("Could not predict the price. The model output is negative or invalid.")

    # Display Physical Integrity Warnings if any exist
    if warnings_list:
        with st.expander("⚠️ Physical & Mathematical Inconsistencies Detected", expanded=True):
            st.markdown(
                "Standard AI regression models look at numeric inputs independently and do not "
                "understand physics or geometry. Below are contradictions detected in your inputs that "
                "could make the prediction less reliable:"
            )
            for warn in warnings_list:
                st.markdown(f"- {warn}")

    # Display Geometry Visual Guide on the main page
    st.markdown("---")
    st.markdown("### 📐 Geometry Visual Guide")
    
    html_code = """
    <div style="text-align: center; font-family: sans-serif; background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #e0e0e0; margin-top: 10px;">
        <svg viewBox="0 0 120 80" width="100%" height="180px">
            <!-- Table -->
            <line x1="40" y1="12" x2="80" y2="12" stroke="#1f77b4" stroke-width="2"/>
            <text x="60" y="8" font-size="6" text-anchor="middle" fill="#1f77b4" font-weight="bold">Table %</text>
            
            <!-- Crown -->
            <line x1="40" y1="12" x2="20" y2="30" stroke="#555" stroke-width="1.5"/>
            <line x1="80" y1="12" x2="100" y2="30" stroke="#555" stroke-width="1.5"/>
            <line x1="20" y1="30" x2="100" y2="30" stroke="#555" stroke-width="2"/>
            <text x="60" y="27" font-size="6" text-anchor="middle" fill="#333">Width (Y) / Length (X)</text>
            
            <!-- Pavilion -->
            <line x1="20" y1="30" x2="60" y2="75" stroke="#555" stroke-width="1.5"/>
            <line x1="100" y1="30" x2="60" y2="75" stroke="#555" stroke-width="1.5"/>
            
            <!-- Center line representing Depth Z -->
            <line x1="60" y1="12" x2="60" y2="75" stroke="#ff7f0e" stroke-width="1.5" stroke-dasharray="2,2"/>
            <text x="64" y="50" font-size="6" fill="#ff7f0e" font-weight="bold">Depth (Z)</text>
            
            <!-- Facet details -->
            <line x1="40" y1="12" x2="45" y2="30" stroke="#ddd" stroke-width="1"/>
            <line x1="80" y1="12" x2="75" y2="30" stroke="#ddd" stroke-width="1"/>
            <line x1="45" y1="30" x2="60" y2="75" stroke="#ddd" stroke-width="1"/>
            <line x1="75" y1="30" x2="60" y2="75" stroke="#ddd" stroke-width="1"/>
        </svg>
        <p style="font-size: 11px; color: #555; text-align: left; margin: 8px 0 0 0; line-height: 1.3;">
            💡 <b>Table %</b> is the width of the top flat facet relative to the diamond's overall width.<br>
            💡 <b>Depth %</b> is the ratio of depth (Z) to the average of X and Y diameter.
        </p>
    </div>
    """
    components.html(html_code, height=280)

    