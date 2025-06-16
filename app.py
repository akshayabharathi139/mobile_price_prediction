import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model/price_predictor.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI Design
st.set_page_config(page_title="MobiWorth", layout="wide")
st.title("MobiWorth")
st.markdown("---")

# Custom Styles
st.markdown(
    """
    <style>
        body {background-color: #f7f9fc;}
        .stButton>button {background-color: #ff4b4b; color: white; font-size: 18px; border-radius: 8px; padding: 10px 24px;}
        .stTextInput>div>div>input {font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Input Features (Row-wise layout)
cols = st.columns(5)
with cols[0]:
    battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=5000, step=100)
    clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, step=0.1)
    fc = st.number_input("Front Camera (MP)", min_value=0, max_value=20, step=1)
    int_memory = st.number_input("Internal Memory (GB)", min_value=1, max_value=256, step=1)

with cols[1]:
    mobile_wt = st.number_input("Mobile Weight (g)", min_value=80, max_value=300, step=5)
    n_cores = st.number_input("Number of Cores", min_value=1, max_value=12, step=1)
    pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=50, step=1)
    ram = st.number_input("RAM (MB)", min_value=256, max_value=8192, step=256)

with cols[2]:
    sc_h = st.number_input("Screen Height (cm)", min_value=5, max_value=20, step=1)
    sc_w = st.number_input("Screen Width (cm)", min_value=3, max_value=12, step=1)
    talk_time = st.number_input("Talk Time (hrs)", min_value=2, max_value=24, step=1)
    px_height = st.number_input("Pixel Height", min_value=200, max_value=2500, step=100)

with cols[3]:
    px_width = st.number_input("Pixel Width", min_value=200, max_value=2500, step=100)
    m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, step=0.1)
    touch_screen = st.selectbox("Touch Screen", [0, 1])
    wifi = st.selectbox("WiFi", [0, 1])

with cols[4]:
    blue = st.selectbox("Bluetooth", [0, 1])
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    four_g = st.selectbox("4G", [0, 1])
    three_g = st.selectbox("3G", [0, 1])

# Submit Button
if st.button("🔍 Predict Price"):
    input_data = np.array([
        battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep,
        mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w,
        talk_time, three_g, touch_screen, wifi
    ]).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]
    price_dict = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
    
    st.success(f"💰 Predicted Price Range: **{price_dict[prediction]}**")

