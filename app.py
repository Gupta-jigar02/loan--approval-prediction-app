import streamlit as st
import pickle
import numpy as np

# ---------------- Page config ----------------
st.set_page_config(page_title="Loan Approval Prediction using Machine Learning", page_icon="🏦", layout="wide")

# ---------------- Dark colorful theme + side images + glass-card styling ----------------
def add_style():
    st.markdown(
        """
        <style>
        /* Dark premium colorful gradient background with animated glow */
        .stApp {
            background: radial-gradient(circle at 20% 20%, #3a1c71 0%, transparent 45%),
                        radial-gradient(circle at 80% 15%, #d76d77 0%, transparent 40%),
                        radial-gradient(circle at 50% 90%, #00c6ff33 0%, transparent 50%),
                        linear-gradient(135deg, #0b0c1e, #1a1440, #250f3e);
            background-attachment: fixed;
        }

        /* frosted glass card behind content - premium gold-rimmed */
        .block-container {
            background: linear-gradient(160deg, rgba(30,25,55,0.8), rgba(15,12,35,0.85));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 2.5rem 3rem 3rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0 0 1px rgba(255,215,120,0.25),
                        0 20px 60px rgba(0,0,0,0.6),
                        0 0 80px rgba(229, 46, 113, 0.15);
            border: 1px solid rgba(255,215,120,0.3);
        }

        h1 {
            background: linear-gradient(90deg, #ffd700, #ff8a00, #e52e71, #a855f7, #00c6ff, #ffd700);
            background-size: 300% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: 900;
            font-size: 68px;
            margin-bottom: 0.2rem;
            letter-spacing: 2px;
            text-shadow: 0 0 50px rgba(229, 46, 113, 0.4);
        }

        .subheading {
            text-align: center;
            color: #ffd700;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: 1px;
            text-shadow: 0 0 15px rgba(255,215,0,0.35);
        }

        .subheading::before, .subheading::after {
            content: "✦";
            color: #ffd700;
            margin: 0 12px;
        }

        .tagline {
            text-align: center;
            color: #d8c8ff;
            font-size: 16px;
            font-weight: 400;
            margin-bottom: 2rem;
            letter-spacing: 0.5px;
            opacity: 0.9;
        }

        /* field label styling */
        .field-label {
            font-weight: 700;
            color: #ffd700;
            font-size: 15px;
            padding-top: 8px;
            text-shadow: 0 0 10px rgba(255,215,0,0.25);
        }

        /* dropdowns / selects */
        div[data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.95) !important;
            border-radius: 10px !important;
            border: 1px solid #a855f7 !important;
            box-shadow: 0 2px 10px rgba(168,85,247,0.25);
        }

        /* number inputs */
        div[data-testid="stNumberInput"] input {
            background-color: rgba(255,255,255,0.95) !important;
            border-radius: 10px !important;
            border: 1px solid #a855f7 !important;
            box-shadow: 0 2px 10px rgba(168,85,247,0.25);
        }

        /* predict button - premium colorful gradient */
        div.stButton > button {
            background: linear-gradient(90deg, #ffd700, #ff8a00, #e52e71, #a855f7);
            background-size: 250% auto;
            color: #1a0b2e;
            font-weight: 800;
            font-size: 18px;
            border: none;
            border-radius: 10px;
            padding: 14px 0;
            width: 100%;
            transition: 0.3s;
            box-shadow: 0 6px 25px rgba(229, 46, 113, 0.5), 0 0 0 1px rgba(255,215,0,0.4);
            letter-spacing: 0.5px;
        }

        div.stButton > button:hover {
            background-position: right center;
            color: #ffffff;
            transform: scale(1.015);
            box-shadow: 0 8px 30px rgba(168, 85, 247, 0.6), 0 0 0 1px rgba(255,215,0,0.6);
        }

        .stSuccess {
            background: linear-gradient(90deg, rgba(30, 200, 120, 0.3), rgba(0, 198, 255, 0.15)) !important;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            color: #d0ffe6 !important;
            border: 1px solid rgba(30, 200, 120, 0.4);
        }

        .stError {
            background: linear-gradient(90deg, rgba(220, 50, 50, 0.3), rgba(229, 46, 113, 0.15)) !important;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            color: #ffd6d6 !important;
            border: 1px solid rgba(220, 50, 50, 0.4);
        }

        /* form row spacing for a cleaner premium layout */
        div[data-testid="column"] {
            padding-top: 6px;
            padding-bottom: 6px;
        }

        div[data-testid="stHorizontalBlock"] {
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 6px 14px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,215,120,0.12);
            transition: 0.2s;
        }
        div[data-testid="stHorizontalBlock"]:hover {
            border: 1px solid rgba(255,215,120,0.35);
            background: rgba(255,255,255,0.05);
        }

        .section-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,215,0,0.6), transparent);
            margin: 0 0 1.8rem 0;
        }
        .side-img {
            position: fixed;
            top: 30px;
            width: 140px;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6), 0 0 0 2px rgba(255,215,0,0.3);
            border: 2px solid rgba(255,215,120,0.35);
            z-index: 999;
        }
        .side-img-left { left: 30px; }
        .side-img-right { right: 30px; }

        /* top and bottom banner images */
        .banner-img {
            width: 100%;
            max-height: 130px;
            object-fit: cover;
            border-radius: 14px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.15);
        }

        /* footer credit */
        .footer-credit {
            text-align: center;
            color: #b8b8e0;
            font-size: 14px;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.15);
        }
        .footer-credit b {
            background: linear-gradient(90deg, #ff8a00, #e52e71, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        @media (max-width: 900px) {
            .side-img { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_style()

# Left / right fixed decorative images (banking / finance themed)
st.markdown(
    """
    <img class="side-img side-img-left" src="https://picsum.photos/seed/loanleft/300/500">
    <img class="side-img side-img-right" src="https://picsum.photos/seed/loanright/300/500">
    """,
    unsafe_allow_html=True,
)

# ---------------- Top banner image + heading ----------------
st.markdown(
    """
    <img class="banner-img" src="https://picsum.photos/seed/loanbanner/1200/300">
    """,
    unsafe_allow_html=True,
)

st.title("🏦 Loan Approval Prediction using Machine Learning")
st.markdown("<div class='subheading'>Loan Approval Prediction using Machine Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Fill in your details below to check loan approval chances</div>", unsafe_allow_html=True)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ---------------- Load model ----------------
model = pickle.load(open("loan_model.pkl", "rb"))

# ---------------- Helper for label + input pairs ----------------
def field_row(label1, widget1_fn, label2=None, widget2_fn=None):
    c1, c2, c3, c4 = st.columns([1, 1.3, 1, 1.3])
    with c1:
        st.markdown(f"<div class='field-label'>{label1}</div>", unsafe_allow_html=True)
    with c2:
        val1 = widget1_fn(c2)
    val2 = None
    if label2 and widget2_fn:
        with c3:
            st.markdown(f"<div class='field-label'>{label2}</div>", unsafe_allow_html=True)
        with c4:
            val2 = widget2_fn(c4)
    return val1, val2

# Row 1: Gender / Married
Gender, Married = field_row(
    "👤 Gender:", lambda c: c.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed"),
    "💍 Married:", lambda c: c.selectbox("Married", ["Yes", "No"], label_visibility="collapsed"),
)

# Row 2: Dependents / Education
Dependents, Education = field_row(
    "👨‍👩‍👧 Dependents:", lambda c: c.selectbox("Dependents", ["0", "1", "2", "3+"], label_visibility="collapsed"),
    "🎓 Education:", lambda c: c.selectbox("Education", ["Graduate", "Not Graduate"], label_visibility="collapsed"),
)

# Row 3: Self Employed / Applicant Income
Self_Employed, ApplicantIncome = field_row(
    "💼 Self Employed:", lambda c: c.selectbox("Self Employed", ["Yes", "No"], label_visibility="collapsed"),
    "💰 Applicant Income:", lambda c: c.number_input("Applicant Income", value=5000, label_visibility="collapsed"),
)

# Row 4: Coapplicant Income / Loan Amount
CoapplicantIncome, LoanAmount = field_row(
    "🤝 Coapplicant Income:", lambda c: c.number_input("Coapplicant Income", value=2000, label_visibility="collapsed"),
    "🏠 Loan Amount:", lambda c: c.number_input("Loan Amount", value=200, label_visibility="collapsed"),
)

# Row 5: Loan Amount Term / Credit History
Loan_Amount_Term, Credit_History_label = field_row(
    "📅 Loan Amount Term:", lambda c: c.number_input("Loan Amount Term", value=360, label_visibility="collapsed"),
    "📊 Credit History:", lambda c: c.selectbox("Credit History", ["Good", "Bad"], label_visibility="collapsed"),
)

# Row 6: Property Area (single field)
c1, c2, c3, c4 = st.columns([1, 1.3, 1, 1.3])
with c1:
    st.markdown("<div class='field-label'>📍 Property Area:</div>", unsafe_allow_html=True)
with c2:
    Property_Area = c2.selectbox("Property Area", ["Urban", "Rural", "Semiurban"], label_visibility="collapsed")

Loan_ID = 10001  # not shown in this layout, kept for model input order

st.write("")

# ---------------- Encoding ----------------
gender = 1 if Gender == "Male" else 0
married = 1 if Married == "Yes" else 0
dependents = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}[Dependents]
education = 0 if Education == "Graduate" else 1
self_employed = 1 if Self_Employed == "Yes" else 0
credit_history = 1 if Credit_History_label == "Good" else 0
property_area = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[Property_Area]

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_data = np.array([[
        Loan_ID,                  # 1
        gender,                   # 2
        married,                  # 3
        dependents,               # 4
        education,                # 5
        self_employed,            # 6
        ApplicantIncome,          # 7
        CoapplicantIncome,        # 8
        LoanAmount,               # 9
        Loan_Amount_Term,         # 10
        credit_history,           # 11
        property_area             # 12
    ]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(
        "Approval Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )

# ---------------- Bottom banner image ----------------
st.markdown(
    """
    <img class="banner-img" src="https://picsum.photos/seed/loanbottom/1200/300">
    """,
    unsafe_allow_html=True,
)

# ---------------- Footer credit ----------------
st.markdown(
    "<div class='footer-credit'>Developed by <b>Jigar Kumar Gupta</b> using Streamlit, Machine Learning &amp; Data Science</div>",
    unsafe_allow_html=True,
)