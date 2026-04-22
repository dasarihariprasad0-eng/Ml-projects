import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Train model
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=32)

scale = StandardScaler()
x_train_scale = scale.fit_transform(x_train)
x_test_scale = scale.transform(x_test)

model = SVC(kernel="linear")
model.fit(x_train_scale, y_train)

# App UI
st.title("🎗️ Breast Cancer Prediction App")
st.write("Fill in the details below to check if a tumor is **Malignant** or **Benign**")

st.sidebar.header("Enter Tumor Details")

mean_radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.0)
mean_texture = st.sidebar.slider("Mean Texture", 9.0, 40.0, 19.0)
mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 200.0, 92.0)
mean_area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 654.0)
mean_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.16, 0.10)
mean_compactness = st.sidebar.slider("Mean Compactness", 0.02, 0.35, 0.10)
mean_concavity = st.sidebar.slider("Mean Concavity", 0.0, 0.43, 0.09)
mean_concave_points = st.sidebar.slider("Mean Concave Points", 0.0, 0.20, 0.05)
mean_symmetry = st.sidebar.slider("Mean Symmetry", 0.10, 0.30, 0.18)
mean_fractal = st.sidebar.slider("Mean Fractal Dimension", 0.05, 0.10, 0.06)

# Use only first 10 features
input_data = np.array([[mean_radius, mean_texture, mean_perimeter,
                         mean_area, mean_smoothness, mean_compactness,
                         mean_concavity, mean_concave_points,
                         mean_symmetry, mean_fractal,
                         0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0]])

input_scaled = scale.transform(input_data)

st.subheader("Your Input Values:")
col1, col2 = st.columns(2)
col1.metric("Mean Radius", mean_radius)
col1.metric("Mean Texture", mean_texture)
col1.metric("Mean Perimeter", mean_perimeter)
col1.metric("Mean Area", mean_area)
col1.metric("Mean Smoothness", mean_smoothness)
col2.metric("Mean Compactness", mean_compactness)
col2.metric("Mean Concavity", mean_concavity)
col2.metric("Mean Concave Points", mean_concave_points)
col2.metric("Mean Symmetry", mean_symmetry)
col2.metric("Mean Fractal Dimension", mean_fractal)

if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.success("✅ Result: BENIGN (Not Cancer)")
        st.balloons()
    else:
        st.error("🚨 Result: MALIGNANT (Cancer Detected)")
    
    st.warning("⚠️ This app is for educational purposes only. Always consult a doctor!")




