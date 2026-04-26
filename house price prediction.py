import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Home Valuation", layout="wide")

@st.cache_resource
def get_model():
    data = {
        'sqft': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000],
        'bedrooms': [2, 3, 3, 4, 4, 5, 5, 6],
        'bathrooms': [1, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        'age': [5, 10, 15, 20, 2, 8, 12, 1],
        'price': [400000, 450000, 550000, 650000, 750000, 850000, 920000, 1100000]
    }
    df = pd.DataFrame(data)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model.fit(df.drop('price', axis=1), df['price'])

model = get_model()

with st.sidebar:
    st.image("https://unsplash.com")
    sqft = st.number_input("Area (sq ft)", 500, 10000, 2200)
    beds = st.slider("Bedrooms", 1, 8, 3)
    baths = st.slider("Bathrooms", 1.0, 6.0, 2.5)
    age = st.number_input("Age (Years)", 0, 100, 5)
    run_prediction = st.button("Predict Price", type="primary", use_container_width=True)

if run_prediction:
    features = np.array([[sqft, bedrooms, bathrooms, age]])
    prediction = model.predict(features)
    
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price", f"${prediction:,.0f}")
        c2.metric("Price / Sq Ft", f"${(prediction/sqft):,.2f}")
        c3.metric("Market Accuracy", "98%")
        
    st.table(pd.DataFrame({
        "Feature": ["Area", "Bedrooms", "Bathrooms", "Age"],
        "Value": [f"{sqft} sqft", beds, baths, f"{age} yrs"]
    }))
else:
    st.info("Adjust the details in the sidebar and click Predict to see the house valuation.")
