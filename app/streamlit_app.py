import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Page Config
st.set_page_config(page_title="Elite House Predictor", page_icon="🏠", layout="wide")

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = "app/static/style.css"
if os.path.exists(css_path):
    local_css(css_path)
else:
    # Fallback to inline if file not found
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        .prediction-card { padding: 20px; background-color: #1e1e1e; border-radius: 15px; text-align: center; }
        .prediction-value { font-size: 3rem; color: #4CAF50; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)


st.title("🏠 Elite Residential Valuation Engine")
st.markdown("---")

# Sidebar for Inputs
st.sidebar.header("Property Characteristics")

def user_input_features():
    # Primary features for prediction
    overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.sidebar.number_input("Living Area (sqft)", 500, 10000, 1500)
    total_bsmt_sf = st.sidebar.number_input("Total Basement SF", 0, 5000, 1000)
    year_built = st.sidebar.slider("Year Built", 1872, 2010, 1990)
    garage_cars = st.sidebar.selectbox("Garage Cars", [0, 1, 2, 3, 4, 5], 2)
    full_bath = st.sidebar.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], 2)
    neighborhood = st.sidebar.selectbox("Neighborhood", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'NridgHt', 'Gilbert', 'Sawyer', 'NWAmes', 'SawyerW', 'Mitchel', 'BrkSide', 'Crawfor', 'IDOTRR', 'Timber', 'NoRidge', 'StoneBr', 'SWISU', 'ClearCr', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blmngtn', 'Greens', 'GrnHill', 'Landmrk'])

    data = {
        'Overall Qual': overall_qual,
        'Gr Liv Area': gr_liv_area,
        'Total Bsmt SF': total_bsmt_sf,
        'Year Built': year_built,
        'Garage Cars': garage_cars,
        'Full Bath': full_bath,
        'Neighborhood': neighborhood,
        # Defaulting other features for the demo to median values
        'MS SubClass': 20,
        'MS Zoning': 'RL',
        'Lot Area': 10000,
        'Street': 'Pave',
        'Lot Shape': 'Reg',
        'Land Contour': 'Lvl',
        'Utilities': 'AllPub',
        'Lot Config': 'Inside',
        'Land Slope': 'Gtl',
        'Condition 1': 'Norm',
        'Condition 2': 'Norm',
        'Bldg Type': '1Fam',
        'House Style': '1Story',
        'Overall Cond': 5,
        'Year Remod/Add': year_built,
        'Roof Style': 'Gable',
        'Roof Matl': 'CompShg',
        'Exterior 1st': 'VinylSd',
        'Exterior 2nd': 'VinylSd',
        'Mas Vnr Type': 'None',
        'Mas Vnr Area': 0.0,
        'Exter Qual': 'TA',
        'Exter Cond': 'TA',
        'Foundation': 'PConc',
        'Bsmt Qual': 'TA',
        'Bsmt Cond': 'TA',
        'Bsmt Exposure': 'No',
        'BsmtFin Type 1': 'Unf',
        'BsmtFin SF 1': 0.0,
        'BsmtFin Type 2': 'Unf',
        'BsmtFin SF 2': 0.0,
        'Bsmt Unf SF': total_bsmt_sf,
        'Heating': 'GasA',
        'Heating QC': 'Ex',
        'Central Air': 'Y',
        'Electrical': 'SBrkr',
        '1st Flr SF': gr_liv_area,
        '2nd Flr SF': 0,
        'Low Qual Fin SF': 0,
        'Bsmt Full Bath': 0.0,
        'Bsmt Half Bath': 0.0,
        'Half Bath': 0,
        'Bedroom AbvGr': 3,
        'Kitchen AbvGr': 1,
        'Kitchen Qual': 'TA',
        'TotRms AbvGrd': 6,
        'Functional': 'Typ',
        'Fireplaces': 0,
        'Garage Type': 'Attchd',
        'Garage Yr Blt': year_built,
        'Garage Finish': 'Unf',
        'Garage Area': 400.0,
        'Garage Qual': 'TA',
        'Garage Cond': 'TA',
        'Paved Drive': 'Y',
        'Wood Deck SF': 0,
        'Open Porch SF': 0,
        'Enclosed Porch': 0,
        '3Ssn Porch': 0,
        'Screen Porch': 0,
        'Pool Area': 0,
        'Misc Val': 0,
        'Mo Sold': 6,
        'Yr Sold': 2010,
        'Sale Type': 'WD ',
        'Sale Condition': 'Normal'
    }
    return pd.DataFrame([data])

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Parameters Summary")
    input_df = user_input_features()
    st.dataframe(input_df.T.head(10).rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("Valuation Result")
    
    # Intelligence Engine (Now exclusively Standard XGBoost)
    st.info("Intelligence Engine: Standard (XGBoost) Active")
    
    # Load Model locally if API is not running
    MODEL_PATH = "models/advanced_xgb.joblib"
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        log_pred = model.predict(input_df)
        prediction = np.expm1(log_pred)[0]
        
        st.markdown(f"""
            <div class="prediction-card">
                <p style="color: #888;">ESTIMATED MARKET VALUE</p>
                <p class="prediction-value">${prediction:,.2f}</p>
                <p style="color: #4CAF50;">Confidence: High (XGBoost Engine)</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model artifact not found. Please train XGBoost model first.")

st.markdown("---")
st.info("💡 Tip: Increasing Overall Quality and Living Area has the highest impact on valuation.")
