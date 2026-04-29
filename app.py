import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FORGE AI | Elite Fitness",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #0A0A0A; }
    .stApp { background-color: #0A0A0A; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    .kicker { color: #39FF14; font-weight: 800; letter-spacing: 2px; font-size: 0.8rem; margin-bottom: 0px; }
    .title-text { font-size: 3rem; font-weight: 900; line-height: 1.1; margin-bottom: 20px; }
    .card { background: #1A1A1A; padding: 20px; border-radius: 15px; border: 1px solid #333; margin-bottom: 15px; }
    .metric-val { color: #39FF14; font-size: 1.5rem; font-weight: 800; }
    .metric-label { font-size: 0.7rem; text-transform: uppercase; color: #888; }
    .stButton>button { 
        background: linear-gradient(90deg, #39FF14, #00FF41) !important; 
        color: black !important; font-weight: bold !important; 
        border-radius: 30px !important; border: none !important;
        width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0px 0px 15px #39FF14; }
</style>
""", unsafe_allow_html=True)

# --- API INITIALIZATION ---
# Using the key provided by the user
API_KEY = "AIzaSyAt-InpijWwHe2Q0acdsGqp-dDaQkN62qE"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- HELPERS ---
def clean_json_response(text):
    """Extracts JSON content from AI markdown response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<p class="kicker">SYSTEM STATUS: ONLINE</p>', unsafe_allow_html=True)
    st.title("⚡ FORGE")
    nav = st.radio("Navigation", ["Dashboard", "Physique Scan", "Nutrition & Workouts", "Progress"], label_visibility="collapsed")
    st.divider()
    st.caption("Powered by Gemini 1.5 Flash Free Tier")

# --- DASHBOARD ---
if nav == "Dashboard":
    st.markdown('<p class="kicker">COMMAND CENTER</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="title-text">OPTIMIZE YOUR<br>BIOLOGY.</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><p class="metric-label">Status</p><p class="metric-val">CALIBRATING</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><p class="metric-label">Next Phase</p><p class="metric-val">BODY SCAN</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><p class="metric-label">API Mode</p><p class="metric-val">FREE TIER</p></div>', unsafe_allow_html=True)

# --- PHYSIQUE SCAN ---
elif nav == "Physique Scan":
    st.markdown('<p class="kicker">VISION ANALYSIS</p>', unsafe_allow_html=True)
    st.title("Body Composition Scan")
    
    uploaded_file = st.file_uploader("Upload a full-body photo", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Subject Analysis In Progress", width=400)
        
        if st.button("EXECUTE ANALYSIS"):
            with st.spinner("Processing bio-markers..."):
                prompt = """
                Analyze this physique image as an elite sports scientist. 
                Return a JSON object with:
                {
                    "body_fat_estimate": "percentage string",
                    "posture_assessment": "short string",
                    "muscle_strengths": ["list of 3 items"],
                    "focus_areas": ["list of 3 items"],
                    "build_type": "string"
                }
                """
                response = model.generate_content([prompt, img])
                data = clean_json_response(response.text)
                
                if data:
                    st.success("Analysis Complete")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f'<div class="card"><h3>{data["body_fat_estimate"]}</h3><p>Estimated Body Fat</p></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card"><h3>{data["build_type"]}</h3><p>Build Type</p></div>', unsafe_allow_html=True)
                    with c2:
                        st.write("**Posture:** " + data["posture_assessment"])
                        st.write("**Strengths:**")
                        for s in data["muscle_strengths"]: st.write(f"- {s}")
                        st.write("**Focus Areas:**")
                        for f in data["focus_areas"]: st.write(f"- {f}")
                else:
                    st.write(response.text)

# --- NUTRITION & WORKOUTS ---
elif nav == "Nutrition & Workouts":
    st.markdown('<p class="kicker">FUEL & FORCE</p>', unsafe_allow_html=True)
    st.title("Protocol Generation")
    
    with st.form("protocol_form"):
        goal = st.selectbox("Primary Objective", ["Hypertrophy", "Fat Loss", "General Health"])
        weight = st.number_input("Weight (kg)", 40, 200, 75)
        diet = st.selectbox("Dietary Preference", ["Omnivore", "Vegetarian", "Vegan", "Keto"])
        submitted = st.form_submit_button("FORGE PROTOCOL")
        
        if submitted:
            with st.spinner("Engineering protocol..."):
                prompt = f"Create a daily 2500kcal {diet} meal plan and a 5-day workout split for {goal} at {weight}kg bodyweight. Focus on immunity-boosting foods. Use markdown tables."
                response = model.generate_content(prompt)
                st.markdown(response.text)

# --- PROGRESS ---
elif nav == "Progress":
    st.title("Evolution Tracking")
    st.info("Log your daily metrics to see your progress trends.")
    st.line_chart({"Weight": [80, 79.5, 79.2, 78.8, 78.5, 78.2]})




