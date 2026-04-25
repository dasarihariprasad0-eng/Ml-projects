import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Page Config ──────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Predictor")
st.markdown("Trained on a **manually created real-world pandas dataset** of Indian cities (2024–25)")
st.divider()

# ── MANUALLY CREATED PANDAS DATASET ──────────
@st.cache_data
def load_dataset():
    data = {
        "City": [
            "Mumbai","Mumbai","Mumbai","Mumbai","Mumbai","Mumbai","Mumbai","Mumbai","Mumbai","Mumbai",
            "Delhi","Delhi","Delhi","Delhi","Delhi","Delhi","Delhi","Delhi","Delhi","Delhi",
            "Bangalore","Bangalore","Bangalore","Bangalore","Bangalore","Bangalore","Bangalore","Bangalore","Bangalore","Bangalore",
            "Hyderabad","Hyderabad","Hyderabad","Hyderabad","Hyderabad","Hyderabad","Hyderabad","Hyderabad","Hyderabad","Hyderabad",
            "Pune","Pune","Pune","Pune","Pune","Pune","Pune","Pune","Pune","Pune",
            "Chennai","Chennai","Chennai","Chennai","Chennai","Chennai","Chennai","Chennai","Chennai","Chennai",
            "Kolkata","Kolkata","Kolkata","Kolkata","Kolkata","Kolkata","Kolkata","Kolkata","Kolkata","Kolkata",
        ],
        "Area_sqft": [
            450,650,900,1100,1400,1800,2200,2800,3500,4500,
            500,700,950,1200,1500,1900,2300,2900,3600,4800,
            600,800,1000,1300,1600,2000,2500,3000,3800,5000,
            550,750,1050,1250,1550,1850,2400,2950,3700,4700,
            500,720,980,1180,1480,1780,2280,2780,3480,4480,
            520,740,960,1160,1460,1760,2260,2760,3460,4460,
            480,680,930,1130,1430,1730,2230,2730,3430,4430,
        ],
        "Bedrooms": [
            1,1,2,2,3,3,4,4,5,5,
            1,1,2,2,3,3,4,4,5,5,
            1,2,2,3,3,3,4,4,5,5,
            1,1,2,2,3,3,4,4,5,5,
            1,1,2,2,3,3,4,4,5,5,
            1,1,2,2,3,3,4,4,5,5,
            1,1,2,2,3,3,4,4,5,5,
        ],
        "Bathrooms": [
            1,1,2,2,2,3,3,4,4,5,
            1,1,2,2,2,3,3,4,4,5,
            1,1,2,2,2,3,3,4,4,5,
            1,1,2,2,2,3,3,4,4,5,
            1,1,2,2,2,3,3,4,4,5,
            1,1,2,2,2,3,3,4,4,5,
            1,1,2,2,2,3,3,4,4,5,
        ],
        "Age_years": [
            2,5,1,8,3,10,2,6,1,4,
            3,6,2,9,4,11,3,7,2,5,
            1,4,3,7,2,9,1,5,3,2,
            4,7,1,8,5,12,2,6,4,3,
            2,5,3,6,1,8,4,7,2,5,
            3,6,2,7,4,10,3,8,1,4,
            5,8,3,9,6,12,4,9,3,6,
        ],
        "Distance_from_center_km": [
            2,5,8,12,15,20,25,30,35,40,
            3,6,9,13,16,21,26,31,36,41,
            4,7,10,14,17,22,27,32,37,42,
            2,5,8,11,14,19,24,29,34,39,
            3,6,9,12,15,20,25,30,35,40,
            4,7,10,13,16,21,26,31,36,41,
            3,6,9,12,15,20,25,30,35,40,
        ],
        "School_rating": [
            9,8,9,7,8,7,8,6,7,8,
            8,7,8,6,7,6,7,5,6,7,
            9,8,9,8,9,7,8,7,8,9,
            8,7,8,7,8,6,7,6,7,8,
            7,6,7,6,7,5,6,5,6,7,
            8,7,8,7,8,6,7,6,7,8,
            7,6,7,6,7,5,6,5,6,7,
        ],
        "Parking": [
            0,1,1,1,2,2,2,3,3,3,
            0,1,1,1,2,2,2,3,3,3,
            1,1,1,2,2,2,2,3,3,3,
            0,1,1,1,2,2,2,3,3,3,
            0,1,1,1,2,2,2,3,3,3,
            0,1,1,1,2,2,2,3,3,3,
            0,1,1,1,2,2,2,3,3,3,
        ],
        # Price in ₹ Lakhs — manually set based on 2024-25 market
        "Price_Lakhs": [
            55,80,145,190,280,370,480,620,780,950,     # Mumbai (highest)
            45,70,120,165,245,320,420,550,700,870,     # Delhi
            40,65,110,155,230,300,390,510,650,820,     # Bangalore
            30,50,90,125,185,250,330,440,570,720,      # Hyderabad
            25,42,78,108,162,215,285,380,495,630,      # Pune
            28,46,85,115,172,228,298,395,510,648,      # Chennai
            20,35,65,92,140,188,248,332,432,548,       # Kolkata (lowest)
        ]
    }
    df = pd.DataFrame(data)
    return df

df = load_dataset()

# ── Encode City & Train ───────────────────────
@st.cache_resource
def train_model(df):
    city_map = {"Mumbai":6,"Delhi":5,"Bangalore":4,
                "Hyderabad":3,"Pune":2,"Chennai":2,"Kolkata":1}
    df["City_code"] = df["City"].map(city_map)

    features = ["Area_sqft","Bedrooms","Bathrooms","Age_years",
                "Distance_from_center_km","School_rating","Parking","City_code"]
    X = df[features].values
    y = df["Price_Lakhs"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    return model, scaler, city_map, mse, r2, rmse

model, scaler, city_map, mse, r2, rmse = train_model(df)

# ── Metrics ───────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("📉 MSE",      f"{mse:.2f}")
c2.metric("📈 R² Score", f"{r2:.4f}")
c3.metric("📏 RMSE",     f"₹ {rmse:.2f} Lakhs")

st.divider()

# ── Predict ───────────────────────────────────
st.subheader("🔮 Predict Your House Price")

col1, col2, col3 = st.columns(3)
with col1:
    city      = st.selectbox("🏙️ City", list(city_map.keys()))
    area      = st.slider("📐 Area (sq ft)",        300, 5000, 1200, step=50)
    bedrooms  = st.slider("🛏️ Bedrooms",             1,   5,    2)
with col2:
    bathrooms = st.slider("🚿 Bathrooms",             1,   5,    2)
    age       = st.slider("📅 Property Age (years)",  0,   40,   5)
    distance  = st.slider("📍 Distance from Center (km)", 1.0, 50.0, 10.0, step=0.5)
with col3:
    school    = st.slider("🎓 School Rating (1-10)", 1.0, 10.0,  7.0, step=0.5)
    parking   = st.slider("🚗 Parking Spaces",        0,   3,    1)

if st.button("💡 Predict Price", use_container_width=True, type="primary"):
    city_code  = city_map[city]
    user_input = np.array([[area, bedrooms, bathrooms, age,
                            distance, school, parking, city_code]])
    user_scaled = scaler.transform(user_input)
    pred = max(model.predict(user_scaled)[0], 0)
    low  = max(pred - rmse, 0)
    high = pred + rmse

    st.success(f"### 🏠 Estimated Price in {city}: ₹ {pred:.1f} Lakhs")
    st.info(f"📊 Price Range: ₹ {low:.1f}L – ₹ {high:.1f}L")

st.divider()

# ── City Avg Price Bar Chart ──────────────────
st.subheader("🏙️ Average Price by City (₹ Lakhs)")
avg = df.groupby("City")["Price_Lakhs"].mean().sort_values(ascending=False)
st.bar_chart(avg, height=320)

st.divider()

# ── Raw Dataset ───────────────────────────────
st.subheader("🗂️ Our Custom Dataset")
st.dataframe(df, use_container_width=True, hide_index=True)