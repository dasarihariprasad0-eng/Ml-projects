import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    cv = CountVectorizer()
    x = cv.fit_transform(df["Message"])
    y = df["Category"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return model, cv, accuracy

model, cv, accuracy = train_model()

st.title("📧 Spam Message Detector")
st.write("Type any message below to check if it is Spam or Not!")
st.info(f"🤖 Model Accuracy: {accuracy*100:.2f}%")

st.divider()

message = st.text_area(
    "✉️ Enter your message here:",
    placeholder="e.g. Congratulations! You won a free prize...",
    height=150
)

if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        new = cv.transform([message])
        prediction = model.predict(new)
        if prediction[0] == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM!")

st.divider()
st.subheader("💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.error("🚨 Spam:")
    st.code("Congratulations! You won $1000!")
    st.code("Free entry win cash click now")
with col2:
    st.success("✅ Not Spam:")
    st.code("Hey are you coming to meeting?")
    st.code("Your order has been shipped!")

st.caption("⚠️ For educational purposes only.")








