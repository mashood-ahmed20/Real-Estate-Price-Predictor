import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Pakistan House Price Predictor", page_icon="🏠", layout="centered")

# ── Train model on load ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('Cleaned_data_for_model.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df[df['price'] > 10000]
    df = df[(df['bedrooms'] > 0) & (df['baths'] > 0)]
    df = df[(df['price'] <= 4.5e7) & (df['Area_in_Marla'] <= 100)]
    df = df[df['purpose'] == 'For Sale']

    le_city = LabelEncoder()
    le_type = LabelEncoder()
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['type_encoded'] = le_type.fit_transform(df['property_type'])
    df['total_rooms'] = df['bedrooms'] + df['baths']

    features = ['baths', 'bedrooms', 'Area_in_Marla', 'city_encoded', 'type_encoded', 'total_rooms']
    X = df[features]
    y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    return model, scaler, le_city, le_type

model, scaler, le_city, le_type = load_model()

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2196F3;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 30px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover { background-color: #1976D2; }
    .result-box {
        background: linear-gradient(135deg, #2196F3, #21CBF3);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box h2 { margin: 0; font-size: 36px; }
    .result-box p  { margin: 5px 0 0; font-size: 16px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 🏠 Pakistan House Price Predictor")
st.markdown("Enter property details below to get an estimated price.")
st.markdown("---")

# Input form
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("🏙️ City", ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad'])
    bedrooms = st.slider("🛏️ Bedrooms", 1, 6, 3)
    area = st.number_input("📐 Area (Marla)", min_value=1.0, max_value=100.0, value=7.0, step=0.5)

with col2:
    property_type = st.selectbox("🏗️ Property Type", ['House', 'Flat', 'Upper Portion', 'Lower Portion', 'Farm House', 'Penthouse', 'Room'])
    baths = st.slider("🚿 Bathrooms", 1, 7, 2)

st.markdown("---")

# Predict
if st.button("🔍 Predict Price"):
    city_enc  = le_city.transform([city])[0]
    type_enc  = le_type.transform([property_type])[0]
    total_rooms = bedrooms + baths

    inp = [[baths, bedrooms, area, city_enc, type_enc, total_rooms]]
    inp_scaled = scaler.transform(inp)
    price = model.predict(inp_scaled)[0]

    price_m = price / 1_000_000
    if price >= 1_000_000:
        price_str = f"PKR {price_m:.2f} Million"
    else:
        price_str = f"PKR {price:,.0f}"

    st.markdown(f"""
    <div class="result-box">
        <p>Estimated Property Price</p>
        <h2>{price_str}</h2>
        <p>{city} · {property_type} · {bedrooms} bed · {baths} bath · {area} Marla</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("### 📊 Quick Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price per Marla", f"PKR {price/area:,.0f}")
    c2.metric("Price per Room", f"PKR {price/(bedrooms+baths):,.0f}")
    c3.metric("Area", f"{area} Marla")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:13px;'>Based on Zameen.com Pakistan real estate data · ML Model: Random Forest</p>", unsafe_allow_html=True)
