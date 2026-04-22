
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

@st.cache_resource
def train_model():
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=32)
    scale = StandardScaler()
    x_train_scale = scale.fit_transform(x_train)
    x_test_scale = scale.transform(x_test)
    model = SVC(kernel="linear", probability=True)
    model.fit(x_train_scale, y_train)
    accuracy = model.score(x_test_scale, y_test)
    y_pred = model.predict(x_test_scale)
    return model, scale, accuracy, y_test, y_pred, cancer

model, scale, accuracy, y_test, y_pred, cancer = train_model()

# Title
st.title("🎗️ Breast Cancer Prediction & Analysis")
st.info(f"🤖 Model Accuracy: {accuracy*100:.2f}%")

# Tabs
tab1, tab2 = st.tabs(["🔍 Predict", "📊 Analysis"])

# Tab 1 - Prediction
with tab1:
    st.subheader("Enter Tumor Measurements")
    st.write("Adjust the values and click Predict!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Mean Values**")
        f1 = st.number_input("Mean Radius", 0.0, 30.0, 14.0)
        f2 = st.number_input("Mean Texture", 0.0, 40.0, 19.0)
        f3 = st.number_input("Mean Perimeter", 0.0, 200.0, 92.0)
        f4 = st.number_input("Mean Area", 0.0, 2500.0, 654.0)
        f5 = st.number_input("Mean Smoothness", 0.0, 0.2, 0.096)
        f6 = st.number_input("Mean Compactness", 0.0, 0.4, 0.104)
        f7 = st.number_input("Mean Concavity", 0.0, 0.5, 0.088)
        f8 = st.number_input("Mean Concave Points", 0.0, 0.2, 0.048)
        f9 = st.number_input("Mean Symmetry", 0.0, 0.4, 0.181)
        f10 = st.number_input("Mean Fractal Dim", 0.0, 0.1, 0.062)

    with col2:
        st.write("**SE Values**")
        f11 = st.number_input("SE Radius", 0.0, 3.0, 0.405)
        f12 = st.number_input("SE Texture", 0.0, 5.0, 1.216)
        f13 = st.number_input("SE Perimeter", 0.0, 22.0, 2.866)
        f14 = st.number_input("SE Area", 0.0, 542.0, 40.33)
        f15 = st.number_input("SE Smoothness", 0.0, 0.03, 0.007)
        f16 = st.number_input("SE Compactness", 0.0, 0.14, 0.025)
        f17 = st.number_input("SE Concavity", 0.0, 0.4, 0.031)
        f18 = st.number_input("SE Concave Points", 0.0, 0.05, 0.011)
        f19 = st.number_input("SE Symmetry", 0.0, 0.08, 0.020)
        f20 = st.number_input("SE Fractal Dim", 0.0, 0.03, 0.003)

    with col3:
        st.write("**Worst Values**")
        f21 = st.number_input("Worst Radius", 0.0, 40.0, 16.26)
        f22 = st.number_input("Worst Texture", 0.0, 50.0, 25.67)
        f23 = st.number_input("Worst Perimeter", 0.0, 260.0, 107.2)
        f24 = st.number_input("Worst Area", 0.0, 4254.0, 880.5)
        f25 = st.number_input("Worst Smoothness", 0.0, 0.3, 0.132)
        f26 = st.number_input("Worst Compactness", 0.0, 1.1, 0.254)
        f27 = st.number_input("Worst Concavity", 0.0, 1.3, 0.272)
        f28 = st.number_input("Worst Concave Points", 0.0, 0.3, 0.114)
        f29 = st.number_input("Worst Symmetry", 0.0, 0.7, 0.290)
        f30 = st.number_input("Worst Fractal Dim", 0.0, 0.21, 0.083)

    if st.button("🔍 Predict Now"):
        input_data = np.array([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
                                f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,
                                f21,f22,f23,f24,f25,f26,f27,f28,f29,f30]])
        input_scaled = scale.transform(input_data)
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0]

        st.divider()
        st.subheader("🧪 Result:")
        if prediction[0] == 1:
            st.success("✅ BENIGN — Tumor is NOT cancerous!")
            st.balloons()
        else:
            st.error("🚨 MALIGNANT — Tumor appears cancerous!")
        st.progress(int(max(prob) * 100))
        st.write(f"Confidence: **{max(prob)*100:.1f}%**")
        st.warning("⚠️ Always consult a qualified doctor!")

# Tab 2 - Analysis
with tab2:
    st.subheader("📊 Dataset Overview")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total Samples", len(cancer.data))
    col2.metric("✅ Benign", int(sum(cancer.target)))
    col3.metric("🚨 Malignant", int(len(cancer.target) - sum(cancer.target)))

    st.divider()

    # Pie chart
    st.subheader("🥧 Benign vs Malignant")
    benign = int(sum(cancer.target))
    malignant = int(len(cancer.target) - sum(cancer.target))
    fig1, ax1 = plt.subplots()
    ax1.pie(
        [benign, malignant],
        labels=["Benign", "Malignant"],
        autopct="%1.1f%%",
        colors=["#21c354", "#ff4b4b"],
        startangle=90
    )
    ax1.axis("equal")
    st.pyplot(fig1)

    st.divider()

    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Malignant", "Benign"])
    ax2.set_yticklabels(["Malignant", "Benign"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="black", fontsize=16)
    plt.colorbar(im)
    st.pyplot(fig2)

    st.divider()

    # Feature importance
    st.subheader("📏 Top 10 Important Features")
    feature_names = cancer.feature_names
    coef = model.coef_[0]
    top10_idx = np.argsort(np.abs(coef))[-10:]
    fig3, ax3 = plt.subplots()
    ax3.barh(
        [feature_names[i] for i in top10_idx],
        [coef[i] for i in top10_idx],
        color=["red" if c < 0 else "green" for c in [coef[i] for i in top10_idx]]
    )
    ax3.set_title("Top 10 Features")
    ax3.set_xlabel("Coefficient Value")
    st.pyplot(fig3)

    st.divider()

    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=["Malignant", "Benign"]
    )
    st.text(report)

st.caption("⚠️ For educational purposes only. Always consult a doctor!")










