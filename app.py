import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

st.set_page_config(page_title="In Shape AI", page_icon="💪", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600&display=swap');
html, body, [class*="css"] { background-color: #000000 !important; color: #e0ffe0 !important; font-family: 'Rajdhani', sans-serif !important; }
.stApp { background: radial-gradient(ellipse at top, #001a00 0%, #000000 60%) !important; }
.logo-title { font-family: 'Orbitron', monospace; font-size: 42px; font-weight: 900; color: #00ff66; text-shadow: 0 0 20px #00ff6680; letter-spacing: 6px; text-align: center; }
.logo-title span { color: #ffffff; }
.tagline { text-align: center; color: #4caf50; letter-spacing: 4px; font-size: 13px; text-transform: uppercase; margin-bottom: 30px; }
.section-header { font-family: 'Orbitron', monospace; font-size: 13px; color: #00ff66; letter-spacing: 3px; text-transform: uppercase; border-bottom: 1px solid #1a4a1a; padding-bottom: 8px; margin-top: 20px; margin-bottom: 15px; }
.bmi-box { background: #050f05; border: 1px solid #1a4a1a; border-radius: 12px; padding: 16px; text-align: center; margin: 10px 0; }
.bmi-val { font-family: 'Orbitron', monospace; font-size: 32px; color: #00ff66; }
.bmi-label { color: #4caf50; font-size: 14px; letter-spacing: 2px; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #00aa44 0%, #007733 100%) !important; color: white !important; font-family: 'Orbitron', monospace !important; font-size: 15px !important; letter-spacing: 3px !important; border: none !important; border-radius: 12px !important; padding: 14px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: 1px solid #1f4f1f !important; border-radius: 20px !important; color: #4caf50 !important; }
.stTabs [aria-selected="true"] { background: #00ff6615 !important; border-color: #00ff66 !important; color: #00ff66 !important; }
hr { border-color: #1a4a1a !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="logo-title">IN<span>SHAPE</span></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Personalized Fitness & Diet Intelligence</div>', unsafe_allow_html=True)
st.markdown("---")

def calc_bmi(height, weight, unit):
    if not height or not weight: return None
    if unit == "Metric (cm / kg)":
        hm = height / 100
        wkg = weight
    else:
        hm = height * 0.0254
        wkg = weight * 0.453592
    return round(wkg / (hm ** 2), 1)

def bmi_label(b):
    if b < 18.5: return "⚠️ Underweight"
    if b < 25: return "✅ Normal"
    if b < 30: return "⚠️ Overweight"
    return "🔴 Obese"

def build_prompt(age, gender, height, weight, unit, activity, goal, diet_pref, health_notes, num_photos):
    h_unit = "cm" if "Metric" in unit else "inches"
    w_unit = "kg" if "Metric" in unit else "lbs"
    b = calc_bmi(height, weight, unit)
    bl = bmi_label(b) if b else "N/A"
    return f"""You are an expert certified personal trainer and sports nutritionist.
Analyze the following person and provide a comprehensive personalized fitness and diet plan.

PROFILE:
- Age: {age} years
- Gender: {gender}
- Height: {height} {h_unit}
- Weight: {weight} {w_unit}
- BMI: {b or 'N/A'} ({bl})
- Activity Level: {activity}
- Goal: {goal}
- Diet Preference: {diet_pref}
- Health Notes: {health_notes or 'None'}
{"- Body photos provided (" + str(num_photos) + " angles) please assess body composition visually." if num_photos > 0 else ""}

Respond in this exact format:

## BODY ANALYSIS
## DAILY CALORIE & MACROS TARGET
## 7-DAY MEAL PLAN
## WEEKLY WORKOUT PLAN
## SUPPLEMENT RECOMMENDATIONS
## LIFESTYLE TIPS
## PROGRESS MILESTONES

Be specific, practical and motivating."""

def get_section(text, marker, next_marker=None):
    lines = text.split("\n")
    capturing = False
    out = []
    for line in lines:
        if marker.upper() in line.upper():
            capturing = True
            out.append(line)
            continue
        if capturing and next_marker and next_marker.upper() in line.upper():
            break
        if capturing:
            out.append(line)
    return "\n".join(out) if out else text

st.markdown('<div class="section-header">🔑 Google Gemini API Key</div>', unsafe_allow_html=True)
api_key = st.text_input("Enter your api key", type="password", placeholder="AIzaSy...")
st.caption("Get your free key at aistudio.google.com")

st.markdown('<div class="section-header">📋 Personal Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with col3:
    unit = st.selectbox("Units", ["Metric (cm / kg)", "Imperial (in / lbs)"])

col4, col5 = st.columns(2)
with col4:
    height = st.number_input(f"Height ({'cm' if 'Metric' in unit else 'inches'})", min_value=50, max_value=300, value=175 if "Metric" in unit else 69, step=1)
with col5:
    weight = st.number_input(f"Weight ({'kg' if 'Metric' in unit else 'lbs'})", min_value=20, max_value=500, value=70 if "Metric" in unit else 154, step=1)

b = calc_bmi(height, weight, unit)
if b:
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown(f'<div class="bmi-box"><div class="bmi-val">{b}</div><div class="bmi-label">BMI</div></div>', unsafe_allow_html=True)
    with col_b2:
        st.markdown(f'<div class="bmi-box"><div class="bmi-val" style="font-size:20px">{bmi_label(b)}</div><div class="bmi-label">Category</div></div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    activity = st.selectbox("Activity Level", ["Sedentary (desk job)", "Lightly Active (1-2x/week)", "Moderate (3-5x/week)", "Very Active (6-7x/week)", "Athlete (2x/day)"])
with col7:
    goal = st.selectbox("Primary Goal", ["Lose Weight / Fat Loss", "Build Muscle / Bulk", "Body Recomposition", "Maintain & Tone", "Improve Endurance", "Increase Strength"])

col8, col9 = st.columns(2)
with col8:
    diet_pref = st.selectbox("Diet Preference", ["No Restriction", "Vegetarian", "Vegan", "Keto", "Paleo", "Gluten-Free", "Halal"])
with col9:
    health_notes = st.text_input("Health Notes (optional)", placeholder="e.g. knee injury, diabetes")

st.markdown('<div class="section-header">📸 Body Photos (Optional)</div>', unsafe_allow_html=True)
st.caption("Upload up to 4 photos for better AI body analysis.")
uploaded_files = st.file_uploader("Upload photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, label_visibility="collapsed")

pil_images = []
if uploaded_files:
    uploaded_files = uploaded_files[:4]
    angle_labels = ["Front", "Side", "Back", "Angled"]
    cols = st.columns(len(uploaded_files))
    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        with col:
            img = Image.open(f).convert("RGB")
            pil_images.append(img)
            st.image(img, caption=angle_labels[i], use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("⚡ GENERATE MY PLAN")

if generate:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key above.")
    else:
        with st.spinner("🔄 Analyzing your profile and building your plan..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = build_prompt(age, gender, height, weight, unit, activity, goal, diet_pref, health_notes, len(pil_images))
                content = []
                for img in pil_images:
                    content.append(img)
                content.append(prompt)
                response = model.generate_content(content)
                result = response.text
                st.session_state["result"] = result
                st.session_state["profile"] = {"age": age, "height": height, "weight": weight, "unit": unit, "bmi": b}
                st.success("✅ Your plan is ready!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if "result" in st.session_state:
    result = st.session_state["result"]
    profile = st.session_state.get("profile", {})
    st.markdown("---")
    st.markdown('<div class="section-header">⚡ Your Personalized Plan</div>', unsafe_allow_html=True)
    u = profile.get("unit", "")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Age", f"{profile.get('age')} yrs")
    with c2: st.metric("Height", f"{profile.get('height')}{'cm' if 'Metric' in u else 'in'}")
    with c3: st.metric("Weight", f"{profile.get('weight')}{'kg' if 'Metric' in u else 'lb'}")
    with c4: st.metric("BMI", profile.get('bmi', 'N/A'))

    tabs = st.tabs(["🧍 Overview", "🥗 Nutrition", "🍽️ Meal Plan", "🏋️ Workout", "💊 Supplements", "🌙 Lifestyle", "📈 Progress"])
    sections = [("BODY ANALYSIS", "DAILY CALORIE"), ("DAILY CALORIE", "7-DAY MEAL"), ("7-DAY MEAL", "WEEKLY WORKOUT"), ("WEEKLY WORKOUT", "SUPPLEMENT"), ("SUPPLEMENT", "LIFESTYLE"), ("LIFESTYLE", "PROGRESS"), ("PROGRESS", None)]
    for tab, (marker, next_marker) in zip(tabs, sections):
        with tab:
            st.markdown(get_section(result, marker, next_marker))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Start Over"):
        del st.session_state["result"]
        st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center; color:#1a4a1a; font-size:11px; letter-spacing:2px;'>IN SHAPE · POWERED BY GEMINI · ALWAYS CONSULT A PHYSICIAN</p>", unsafe_allow_html=True)