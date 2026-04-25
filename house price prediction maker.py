import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title("🏠 House Price Predictor")

# ── Dataset ───────────────────────────────────
df = pd.DataFrame({
    "City":       ["Mumbai","Mumbai","Delhi","Delhi","Bangalore","Bangalore","Hyderabad","Hyderabad","Pune","Pune"],
    "Area":       ["Bandra","Andheri","CP","Dwarka","Koramangala","Whitefield","Banjara Hills","Gachibowli","Koregaon Park","Hinjewadi"],
    "Sqft":       [650, 1200, 700, 1500, 800, 1600, 750, 1550, 720, 1480],
    "Bedrooms":   [1, 2, 1, 3, 2, 3, 1, 3, 1, 3],
    "Age":        [5, 3, 6, 2, 4, 1, 7, 2, 5, 1],
    "Price_Lakh": [80, 190, 70, 245, 65, 230, 50, 185, 42, 162]
})

# ── Train ─────────────────────────────────────
le_city = LabelEncoder()
le_area = LabelEncoder()
df["City_code"] = le_city.fit_transform(df["City"])
df["Area_code"] = le_area.fit_transform(df["Area"])

X = df[["Sqft","Bedrooms","Age","City_code","Area_code"]]
y = df["Price_Lakh"]
model = LinearRegression().fit(X, y)

# ── UI ────────────────────────────────────────
city     = st.selectbox("🏙️ City", df["City"].unique())
area     = st.selectbox("📌 Area", df[df["City"] == city]["Area"].unique())
sqft     = st.slider("📐 Sqft",     300, 5000, 1000, step=50)
bedrooms = st.slider("🛏️ Bedrooms",  1,  5,    2)
age      = st.slider("📅 Age",       0,  40,   5)

if st.button("Predict 💡", use_container_width=True):
    city_code = le_city.transform([city])[0]
    area_code = le_area.transform([area])[0]
    pred = model.predict([[sqft, bedrooms, age, city_code, area_code]])[0]
    st.success(f"### Estimated Price: ₹ {pred:.1f} Lakhs")

st.divider()
st.subheader("📊 Dataset")
st.dataframe(df[["City","Area","Sqft","Bedrooms","Age","Price_Lakh"]], hide_index=True)
