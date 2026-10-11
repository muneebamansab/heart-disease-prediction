
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #e11d48, #be123c 60%, #881337);
    padding: 2rem 1.5rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 30px rgba(190, 18, 60, 0.35);
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.1rem; color: white; padding: 0; }
.hero p { margin: 0.4rem 0 0; color: white; opacity: 0.92; }
.heart { font-size: 3rem; display: inline-block; animation: beat 1.2s infinite; }
@keyframes beat {
    0%, 100% { transform: scale(1); }
    20% { transform: scale(1.2); }
    40% { transform: scale(1); }
    60% { transform: scale(1.15); }
}

div.stButton > button {
    background: linear-gradient(135deg, #e11d48, #be123c);
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    font-size: 1.05rem;
    width: 100%;
    box-shadow: 0 4px 14px rgba(190, 18, 60, 0.35);
}
div.stButton > button p { color: white; }
div.stButton > button:hover { filter: brightness(1.12); border: none; }

.result {
    padding: 1.6rem;
    border-radius: 16px;
    text-align: center;
    margin-top: 1.2rem;
}
.result h2 { margin: 0; color: inherit; padding: 0; }
.result p { margin: 0.5rem 0 0; color: inherit; }
.bad  { background: #fee2e2; border: 2px solid #ef4444; color: #991b1b; }
.good { background: #dcfce7; border: 2px solid #22c55e; color: #166534; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return pickle.load(open("svc_trained_model.pkl", "rb"))


model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ❤️ About")
    st.write(
        "This system predicts whether a patient has heart disease using "
        "a Support Vector Classifier trained on the UCI Cleveland dataset."
    )
    st.markdown("### Input attributes")
    st.markdown(
        "- **thal**: thalassemia test result\n"
        "- **ca**: major vessels colored (0 to 3)\n"
        "- **oldpeak**: ST depression after exercise\n"
        "- **thalach**: maximum heart rate achieved"
    )
    st.caption("Machine Learning Fundamentals | Assignment 2")

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="heart">❤️</div>
    <h1>Heart Disease Prediction System</h1>
    <p>Enter the patient's test results and get an instant prediction</p>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
with st.container(border=True):
    st.subheader("🩺 Patient Details")
    col1, col2 = st.columns(2)

    with col1:
        thal_label = st.selectbox(
            "Thalassemia (thal)",
            ["Normal", "Fixed defect", "Reversible defect"],
        )
        ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3])

    with col2:
        oldpeak = st.number_input(
            "ST depression induced by exercise (oldpeak)",
            min_value=0.0, max_value=7.0, value=1.0, step=0.1,
        )
        thalach = st.number_input(
            "Maximum heart rate achieved (thalach)",
            min_value=60, max_value=220, value=150, step=1,
        )

thal_map = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}
thal = thal_map[thal_label]

st.write("")
predict_clicked = st.button("🔍 Predict")

# ---------- Prediction ----------
if predict_clicked:
    # same attribute order as training: thal, ca, oldpeak, thalach
    data = pd.DataFrame({
        "thal": [float(thal)],
        "ca": [float(ca)],
        "oldpeak": [float(oldpeak)],
        "thalach": [float(thalach)],
    })

    result = model.predict(data)[0]

    if result == 1:
        st.markdown("""
        <div class="result bad">
            <h2>⚠️ HEART DISEASE</h2>
            <p>The model predicts a presence of heart disease.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result good">
            <h2>✅ NO HEART DISEASE</h2>
            <p>The model predicts no heart disease.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Thal", thal_label)
    m2.metric("Vessels (ca)", ca)
    m3.metric("Oldpeak", oldpeak)
    m4.metric("Max HR", thalach)

st.caption("⚕️ Educational project only. Not a substitute for professional medical advice.")