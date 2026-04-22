
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load and train model
@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
st.write(df.columns.tolist())
    df.columns = ["Category", "Message"]
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

# App UI
st.title("📧 Spam Message Detector")
st.write("Type any message below and instantly find out if it is **Spam or Not!**")

# Show accuracy
st.info(f"🤖 Model Accuracy: {accuracy*100:.2f}%")

st.divider()

# Input
message = st.text_area(
    "✉️ Enter your message here:",
    placeholder="e.g. Congratulations! You have won a free prize...",
    height=150
)

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_btn = st.button("🔍 Check Message", use_container_width=True)

st.divider()

if predict_btn:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        new = cv.transform([message])
        prediction = model.predict(new)
        probability = model.predict_proba(new)[0]

        st.subheader("📊 Result:")

        if prediction[0] == "spam":
            st.error("🚨 This message is **SPAM!**")
            st.progress(int(max(probability) * 100))
            st.write(f"Spam confidence: **{max(probability)*100:.1f}%**")
        else:
            st.success("✅ This message is **NOT SPAM!**")
            st.progress(int(max(probability) * 100))
            st.write(f"Ham confidence: **{max(probability)*100:.1f}%**")

st.divider()

# Example messages
st.subheader("💡 Try these examples:")
col1, col2 = st.columns(2)

with col1:
    st.error("🚨 Spam Examples:")
    st.code("Congratulations! You won $1000 prize!")
    st.code("Free entry to win cash reward click now")
    st.code("Urgent! Your account will be suspended")

with col2:
    st.success("✅ Not Spam Examples:")
    st.code("Hey, are you coming to the meeting?")
    st.code("Can you pick up milk on the way home?")
    st.code("Your order has been shipped successfully")

st.caption("⚠️ This app is for educational purposes only.")





