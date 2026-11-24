
# ============================================================
# PUNE REAL ESTATE PRICE PREDICTION & INVESTMENT ADVISOR
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
import re

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate AI Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Pune_House_Data.csv"
MODEL_PATH = BASE_DIR / "models" / "real_estate_model.pkl"

FEATURES = [
    "site_location",
    "area_type",
    "total_sqft",
    "BHK",
    "bath",
    "balcony"
]

# Demo credentials only. Do not use these for a production application.
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

# ============================================================
# 3. SESSION STATE
# ============================================================

defaults = {
    "is_logged_in": False,
    "theme": "Default",
    "prediction_history": [],
    "last_prediction": None,
    "confirm_logout": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# 4. THEME AND CSS
# ============================================================

def apply_theme(theme):

    if theme == "Dark":
        page_bg = "#0B1220"
        card_bg = "#151F32"
        text_color = "#F1F5F9"
        muted_color = "#AAB8CC"
        input_bg = "#1E293B"
        border_color = "#334155"
        popup_bg = "#172338"

    elif theme == "Light":
        page_bg = "#F4F7FB"
        card_bg = "#FFFFFF"
        text_color = "#172033"
        muted_color = "#64748B"
        input_bg = "#FFFFFF"
        border_color = "#DCE3EC"
        popup_bg = "#FFFFFF"

    else:
        page_bg = "#F0FDF8"
        card_bg = "#FFFFFF"
        text_color = "#16332B"
        muted_color = "#64748B"
        input_bg = "#FFFFFF"
        border_color = "#D1E7DD"
        popup_bg = "#FFFFFF"

    st.markdown(
        f"""
        <style>
        /* ================= THEME VARIABLES ================= */

        :root {{
            --page-bg: {page_bg};
            --card-bg: {card_bg};
            --text-color: {text_color};
            --muted-color: {muted_color};
            --input-bg: {input_bg};
            --border-color: {border_color};
            --popup-bg: {popup_bg};
        }}

        /* ================= MAIN APP ================= */

        .stApp {{
            background-color: var(--page-bg) !important;
            color: var(--text-color) !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: var(--page-bg) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: var(--page-bg) !important;
        }}

        [data-testid="stSidebar"] {{
            background-color: var(--card-bg) !important;
            border-right: 1px solid var(--border-color);
        }}

        [data-testid="stSidebar"] > div {{
            background-color: var(--card-bg) !important;
        }}

        /* ================= TEXT ================= */

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp p,
        .stApp label,
        .stApp li,
        .stApp strong,
        .stApp [data-testid="stMarkdownContainer"] {{
            color: var(--text-color);
        }}

        .stApp [data-testid="stCaptionContainer"],
        .stApp .stCaption {{
            color: var(--muted-color) !important;
        }}

        .main-title {{
            font-size: 2.25rem;
            font-weight: 800;
            color: #059669 !important;
            margin-bottom: 5px;
        }}

        .subtitle {{
            color: var(--muted-color) !important;
            font-size: 1rem;
            margin-bottom: 22px;
        }}

        /* ================= CARDS ================= */

        .custom-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
        }}

        [data-testid="stMetric"] {{
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color);
            padding: 16px;
            border-radius: 12px;
        }}

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {{
            color: var(--muted-color) !important;
        }}

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {{
            color: var(--text-color) !important;
        }}

        /* ================= TEXT INPUTS ================= */

        .stApp input,
        .stApp textarea {{
            background-color: var(--input-bg) !important;
            color: var(--text-color) !important;
            -webkit-text-fill-color: var(--text-color) !important;
            border-color: var(--border-color) !important;
            caret-color: var(--text-color) !important;
        }}

        .stApp input::placeholder,
        .stApp textarea::placeholder {{
            color: var(--muted-color) !important;
            -webkit-text-fill-color: var(--muted-color) !important;
            opacity: 1 !important;
        }}

        [data-testid="stTextInput"] > div,
        [data-testid="stTextArea"] > div,
        [data-testid="stNumberInput"] > div {{
            background-color: var(--input-bg) !important;
            border-color: var(--border-color) !important;
        }}

        [data-testid="stNumberInput"] button {{
            background-color: var(--input-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}

        [data-testid="stNumberInput"] button svg {{
            fill: var(--text-color) !important;
            color: var(--text-color) !important;
        }}

        /* ================= SELECTBOX ================= */

        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
            background-color: var(--input-bg) !important;
            border-color: var(--border-color) !important;
        }}

        [data-testid="stSelectbox"] [data-baseweb="select"] *,
        [data-testid="stMultiSelect"] [data-baseweb="select"] * {{
            color: var(--text-color) !important;
        }}

        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg {{
            fill: var(--text-color) !important;
            color: var(--text-color) !important;
        }}

        /* ================= DROPDOWN POPUP ================= */

        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [data-baseweb="listbox"] {{
            background-color: var(--popup-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}

        [data-baseweb="popover"] li,
        [data-baseweb="menu"] li,
        [data-baseweb="listbox"] li,
        [role="option"] {{
            background-color: var(--popup-bg) !important;
            color: var(--text-color) !important;
        }}

        [role="option"] *,
        [data-baseweb="menu"] *,
        [data-baseweb="listbox"] * {{
            color: var(--text-color) !important;
        }}

        [role="option"]:hover,
        [role="option"][aria-selected="true"] {{
            background-color: var(--border-color) !important;
        }}

        /* ================= RADIO AND CHECKBOX ================= */

        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label {{
            color: var(--text-color) !important;
        }}

        [data-testid="stRadio"] label *,
        [data-testid="stCheckbox"] label *,
        [data-testid="stToggle"] label * {{
            color: var(--text-color) !important;
        }}

        /* ================= DATAFRAME ================= */

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ================= BUTTONS ================= */

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {{
            background-color: #059669 !important;
            color: #FFFFFF !important;
            border: 1px solid #047857 !important;
            border-radius: 9px !important;
            font-weight: 600 !important;
            min-height: 42px;
        }}

        .stButton > button *,
        .stFormSubmitButton > button *,
        .stDownloadButton > button * {{
            color: #FFFFFF !important;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover {{
            background-color: #047857 !important;
            border-color: #065F46 !important;
        }}

        /* RED LOGOUT AND YES LOGOUT BUTTONS */

        [data-testid="stSidebar"] .st-key-logout_button button,
        [data-testid="stSidebar"] .st-key-confirm_logout_yes button {{
            background-color: #DC2626 !important;
            color: #FFFFFF !important;
            border: 1px solid #B91C1C !important;
            border-radius: 9px !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebar"] .st-key-logout_button button *,
        [data-testid="stSidebar"] .st-key-confirm_logout_yes button * {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        [data-testid="stSidebar"] .st-key-logout_button button:hover,
        [data-testid="stSidebar"] .st-key-confirm_logout_yes button:hover {{
            background-color: #B91C1C !important;
            border-color: #991B1B !important;
        }}

        /* ================= ALERTS ================= */

        [data-testid="stAlert"] {{
            border-radius: 10px;
        }}

        /* ================= DIVIDERS AND FOOTER ================= */

        hr {{
            border-color: var(--border-color) !important;
        }}

        .footer {{
            text-align: center;
            color: var(--muted-color) !important;
            font-size: 0.85rem;
            padding: 25px 0 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_theme(st.session_state.theme)

# ============================================================
# 5. LOAD DATASET AND MODEL
# ============================================================

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


df = None
model = None
data_error = None
model_error = None

if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as e:
        data_error = str(e)
else:
    data_error = f"Dataset not found: {DATA_PATH}"

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as e:
        model_error = str(e)
else:
    model_error = f"Model not found: {MODEL_PATH}"

# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def clean_sqft(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    try:
        if "-" in value:
            parts = value.split("-")
            numbers = [
                float(part.strip())
                for part in parts
                if part.strip().replace(".", "", 1).isdigit()
            ]

            if numbers:
                return sum(numbers) / len(numbers)

        match = re.search(r"\d+\.?\d*", value)

        if match:
            return float(match.group())

    except (ValueError, TypeError):
        return np.nan

    return np.nan


def get_model_prediction(property_data):
    if model is None:
        raise RuntimeError(
            "Model could not be loaded. Check models/real_estate_model.pkl."
        )

    input_df = pd.DataFrame([property_data], columns=FEATURES)
    prediction = float(np.asarray(model.predict(input_df)).ravel()[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return max(0.0, prediction)


def get_location_options():
    if df is not None and "site_location" in df.columns:
        values = sorted(
            df["site_location"].dropna().astype(str).unique().tolist()
        )

        if values:
            return values

    return ["Other"]


def get_area_type_options():
    if df is not None and "area_type" in df.columns:
        values = sorted(
            df["area_type"].dropna().astype(str).unique().tolist()
        )

        if values:
            return values

    return [
        "Super built-up Area",
        "Built-up Area",
        "Plot Area",
        "Carpet Area"
    ]


def make_property_data(location, area_type, sqft, bhk, bath, balcony):
    return {
        "site_location": location,
        "area_type": area_type,
        "total_sqft": float(sqft),
        "BHK": int(bhk),
        "bath": int(bath),
        "balcony": int(balcony)
    }


def format_price(value):
    return f"₹ {value:,.2f} lakh"


def style_chart(fig, height=420):
    dark = st.session_state.theme == "Dark"

    font_color = "#F1F5F9" if dark else "#263445"
    grid_color = "#334155" if dark else "#E8EEF4"

    fig.update_layout(
        height=height,
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="#151F32" if dark else "#FFFFFF",
        plot_bgcolor="#151F32" if dark else "#FFFFFF",
        font=dict(color=font_color, size=12),
        title_font=dict(size=18, color=font_color),
        margin=dict(l=20, r=20, t=65, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False
    )

    return fig

# ============================================================
# 7. LOGIN PAGE
# ============================================================

if not st.session_state.is_logged_in:

    st.markdown(
        """
        <div style="text-align:center; padding-top:55px;">
            <div style="font-size:65px;">🏡</div>
            <h1 style="color:#059669;">Real Estate AI Advisor</h1>
            <p style="font-size:17px;">
                Pune Real Estate Price Prediction & Investment Advisor
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, login_col, _ = st.columns([1, 1.2, 1])

    with login_col:
        st.markdown("### 🔐 Login to Continue")

        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )

            login_clicked = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if login_clicked:
            if (
                username == DEMO_USERNAME
                and password == DEMO_PASSWORD
            ):
                st.session_state.is_logged_in = True
                st.session_state.confirm_logout = False
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.info(
            "College demo login: username `admin`, password `admin123`."
        )

    st.markdown(
        '<div class="footer">Real Estate AI Advisor | Academic Project Demo</div>',
        unsafe_allow_html=True
    )

    st.stop()

# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🏡 Real Estate AI")
    st.caption("Smart Property Decision System")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Price Prediction",
            "Investment Advisor",
            "Dataset Insights",
            "Property Comparison",
            "Prediction History"
        ],
        key="navigation"
    )

    st.divider()
    st.markdown("### 🎨 Appearance")

    selected_theme = st.selectbox(
        "Choose theme",
        ["Default", "Light", "Dark"],
        index=["Default", "Light", "Dark"].index(
            st.session_state.theme
        ),
        key="theme_selector"
    )

    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Account")

    if st.button(
        "🧹 Clear Prediction History",
        key="clear_history",
        use_container_width=True
    ):
        st.session_state.prediction_history = []
        st.session_state.last_prediction = None
        st.success("Prediction history cleared.")

    if st.button(
        "🚪 Logout",
        key="logout_button",
        use_container_width=True
    ):
        st.session_state.confirm_logout = True

    if st.session_state.confirm_logout:
        st.warning("Are you sure you want to log out?")

        if st.button(
            "Yes, Logout",
            key="confirm_logout_yes",
            use_container_width=True
        ):
            st.session_state.is_logged_in = False
            st.session_state.prediction_history = []
            st.session_state.last_prediction = None
            st.session_state.confirm_logout = False
            st.rerun()

        if st.button(
            "Cancel",
            key="confirm_logout_cancel",
            use_container_width=True
        ):
            st.session_state.confirm_logout = False
            st.rerun()

# ============================================================
# 9. APP HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏡 Real Estate AI Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict property prices, evaluate investment options, and explore data.'
    '</div>',
    unsafe_allow_html=True
)

if data_error:
    st.warning(f"Dataset issue: {data_error}")

if model_error:
    st.warning(f"Model issue: {model_error}")

# ============================================================
# 10. HOME PAGE
# ============================================================

if page == "Home":

    st.subheader("Welcome to Your Smart Property Assistant")

    st.write(
        "This application uses a trained machine-learning model to estimate "
        "property prices from location, area type, size, bedrooms, bathrooms, "
        "and balconies."
    )

    total_records = len(df) if df is not None else 0

    total_locations = (
        df["site_location"].nunique()
        if df is not None and "site_location" in df.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", f"{total_records:,}")
    col2.metric("Locations in Dataset", f"{total_locations:,}")
    col3.metric(
        "Predictions Made",
        len(st.session_state.prediction_history)
    )

    st.markdown("### ✨ What You Can Do")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            <div class="custom-card">
                <h3>💰 Price Prediction</h3>
                <p>Estimate a property's price using its features.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="custom-card">
                <h3>📊 Dataset Insights</h3>
                <p>Explore price distributions and property locations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="custom-card">
                <h3>🕘 Prediction History</h3>
                <p>Review and download predictions from this session.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col2:
        st.markdown(
            """
            <div class="custom-card">
                <h3>📈 Investment Advisor</h3>
                <p>Compare estimated prices against your budget.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="custom-card">
                <h3>⚖️ Property Comparison</h3>
                <p>Compare estimates for different properties.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="custom-card">
                <h3>🎨 Custom Appearance</h3>
                <p>Choose Default, Light, or Dark mode.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Predictions are estimates, not guaranteed market prices. Verify "
        "the dataset's geographic coverage and price units before relying "
        "on results."
    )

# ============================================================
# 11. PRICE PREDICTION PAGE
# ============================================================

elif page == "Price Prediction":

    st.subheader("💰 Property Price Prediction")
    st.write("Enter the property details to estimate its price.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            location = st.selectbox(
                "Property Location",
                get_location_options()
            )

            area_type = st.selectbox(
                "Area Type",
                get_area_type_options()
            )

            total_sqft = st.number_input(
                "Total Area (sq. ft.)",
                min_value=100.0,
                max_value=100000.0,
                value=1000.0,
                step=50.0
            )

        with col2:
            bhk = st.selectbox(
                "Bedrooms (BHK)",
                [1, 2, 3, 4, 5, 6, 7, 8]
            )

            bath = st.selectbox(
                "Bathrooms",
                [1, 2, 3, 4, 5, 6, 7, 8]
            )

            balcony = st.selectbox(
                "Balconies",
                [0, 1, 2, 3, 4]
            )

        predict_clicked = st.form_submit_button(
            "🔍 Predict Property Price",
            use_container_width=True
        )

    if predict_clicked:
        try:
            property_data = make_property_data(
                location,
                area_type,
                total_sqft,
                bhk,
                bath,
                balcony
            )

            predicted_price = get_model_prediction(property_data)
            estimated_per_sqft = predicted_price * 100000 / total_sqft

            result = {
                "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Location": location,
                "Area Type": area_type,
                "Area (sq. ft.)": total_sqft,
                "BHK": bhk,
                "Bathrooms": bath,
                "Balconies": balcony,
                "Predicted Price (Lakh)": round(predicted_price, 2),
                "Estimated Price per sq. ft. (₹)": round(
                    estimated_per_sqft, 2
                )
            }

            st.session_state.last_prediction = result
            st.session_state.prediction_history.append(result)

        except Exception as e:
            st.error(f"Could not generate prediction: {e}")

    if st.session_state.last_prediction is not None:
        result = st.session_state.last_prediction

        st.divider()
        st.markdown("### 📋 Prediction Result")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Estimated Property Price",
            format_price(result["Predicted Price (Lakh)"])
        )

        metric2.metric(
            "Estimated Price per sq. ft.",
            f"₹ {result['Estimated Price per sq. ft. (₹)']:,.2f}"
        )

        metric3.metric("Bedrooms", f"{result['BHK']} BHK")

        st.success("Prediction generated successfully.")

        st.markdown("#### Property Details")

        st.dataframe(
            pd.DataFrame([result]),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction",
            data=pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
            file_name="property_prediction.csv",
            mime="text/csv"
        )

        st.caption(
            "Confirm that your model's target variable is measured in lakh "
            "before interpreting the displayed prices."
        )

# ============================================================
# 12. INVESTMENT ADVISOR PAGE
# ============================================================

elif page == "Investment Advisor":

    st.subheader("📈 Investment Advisor")

    st.write(
        "Estimate a property's price and compare it with your available budget."
    )

    with st.form("investment_form"):

        col1, col2 = st.columns(2)

        with col1:
            inv_location = st.selectbox(
                "Property Location",
                get_location_options(),
                key="inv_location"
            )

            inv_area_type = st.selectbox(
                "Area Type",
                get_area_type_options(),
                key="inv_area_type"
            )

            inv_sqft = st.number_input(
                "Property Area (sq. ft.)",
                min_value=100.0,
                max_value=100000.0,
                value=1000.0,
                step=50.0,
                key="inv_sqft"
            )

        with col2:
            inv_bhk = st.selectbox(
                "Bedrooms (BHK)",
                [1, 2, 3, 4, 5, 6, 7, 8],
                key="inv_bhk"
            )

            inv_bath = st.selectbox(
                "Bathrooms",
                [1, 2, 3, 4, 5, 6, 7, 8],
                key="inv_bath"
            )

            inv_balcony = st.selectbox(
                "Balconies",
                [0, 1, 2, 3, 4],
                key="inv_balcony"
            )

        budget = st.number_input(
            "Your Budget (₹ lakh)",
            min_value=1.0,
            max_value=100000.0,
            value=50.0,
            step=5.0
        )

        investment_clicked = st.form_submit_button(
            "📊 Analyze Investment",
            use_container_width=True
        )

    if investment_clicked:
        try:
            property_data = make_property_data(
                inv_location,
                inv_area_type,
                inv_sqft,
                inv_bhk,
                inv_bath,
                inv_balcony
            )

            estimated_price = get_model_prediction(property_data)
            difference = budget - estimated_price

            st.divider()
            st.markdown("### Investment Analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Estimated Price",
                format_price(estimated_price)
            )

            c2.metric(
                "Your Budget",
                format_price(budget)
            )

            c3.metric(
                "Budget Difference",
                format_price(abs(difference)),
                delta=(
                    "Budget remaining"
                    if difference >= 0
                    else "Amount over budget"
                ),
                delta_color="normal" if difference >= 0 else "inverse"
            )

            if estimated_price <= budget:
                st.success(
                    "The estimated price is within your stated budget. "
                    "Consider further due diligence."
                )
            else:
                st.warning(
                    "The estimated price is above your stated budget. "
                    "Consider another property or budget."
                )

            st.info(
                "This is a budget comparison, not a guarantee of investment "
                "returns. Consider taxes, registration, maintenance, loan "
                "interest, rental demand, and legal verification."
            )

        except Exception as e:
            st.error(f"Investment analysis failed: {e}")

# ============================================================
# 13. DATASET INSIGHTS PAGE
# ============================================================

elif page == "Dataset Insights":

    st.subheader("📊 Dataset Insights")

    if df is None:
        st.error(
            "Dataset unavailable. Place Pune_House_Data.csv inside the data folder."
        )

    else:
        st.markdown("### Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

        with st.expander("View dataset sample"):
            st.dataframe(df.head(100), use_container_width=True)

        with st.expander("View missing values"):
            missing = df.isna().sum().reset_index()
            missing.columns = ["Column", "Missing Values"]
            missing = missing[missing["Missing Values"] > 0]

            if missing.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(
                    missing.sort_values(
                        "Missing Values",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()
        st.markdown("### Price Distribution")

        if "price" in df.columns:
            prices = pd.to_numeric(
                df["price"],
                errors="coerce"
            ).dropna()

            prices = prices[prices >= 0]

            if not prices.empty:
                fig = px.histogram(
                    pd.DataFrame({"Price": prices}),
                    x="Price",
                    nbins=35,
                    title="Distribution of Property Prices",
                    color_discrete_sequence=["#10B981"]
                )

                fig.update_traces(
                    marker_line_color="#047857",
                    marker_line_width=0.5
                )

                fig.update_xaxes(title="Dataset Price Value")
                fig.update_yaxes(title="Number of Properties")

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

                st.caption(
                    "The chart uses prices as stored in the dataset. "
                    "Confirm the units before interpreting them as rupees or lakh."
                )

                stat1, stat2, stat3, stat4 = st.columns(4)

                stat1.metric("Average", f"{prices.mean():,.2f}")
                stat2.metric("Median", f"{prices.median():,.2f}")
                stat3.metric("Minimum", f"{prices.min():,.2f}")
                stat4.metric("Maximum", f"{prices.max():,.2f}")

            else:
                st.warning("No valid numeric price values are available.")

        else:
            st.warning("The dataset does not contain a 'price' column.")

        st.divider()
        st.markdown("### Properties by Location")

        if "site_location" in df.columns:
            top_locations = (
                df["site_location"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(15)
                .sort_values()
                .reset_index()
            )

            top_locations.columns = ["Location", "Property Count"]

            if not top_locations.empty:
                fig = px.bar(
                    top_locations,
                    x="Property Count",
                    y="Location",
                    orientation="h",
                    title="Top 15 Locations by Number of Records",
                    color="Property Count",
                    color_continuous_scale="Greens"
                )

                st.plotly_chart(
                    style_chart(fig, height=500),
                    use_container_width=True
                )

        else:
            st.warning(
                "The dataset does not contain a 'site_location' column."
            )

        st.divider()
        st.markdown("### Area Type Distribution")

        if "area_type" in df.columns:
            area_counts = (
                df["area_type"]
                .dropna()
                .astype(str)
                .value_counts()
                .reset_index()
            )

            area_counts.columns = ["Area Type", "Count"]

            if not area_counts.empty:
                fig = px.pie(
                    area_counts,
                    names="Area Type",
                    values="Count",
                    hole=0.45,
                    title="Properties by Area Type",
                    color_discrete_sequence=[
                        "#10B981",
                        "#6366F1",
                        "#F59E0B",
                        "#06B6D4",
                        "#EC4899"
                    ]
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

# ============================================================
# 14. PROPERTY COMPARISON PAGE
# ============================================================

elif page == "Property Comparison":

    st.subheader("⚖️ Property Comparison")

    st.write(
        "Enter details for two or more properties to compare estimated prices."
    )

    with st.form("comparison_form"):

        number_of_properties = st.selectbox(
            "Number of Properties",
            [2, 3, 4, 5],
            index=1
        )

        property_inputs = []

        for i in range(number_of_properties):
            st.markdown(f"#### 🏠 Property {i + 1}")

            col1, col2, col3 = st.columns(3)

            with col1:
                loc = st.selectbox(
                    f"Location — Property {i + 1}",
                    get_location_options(),
                    key=f"comp_location_{i}"
                )

                area = st.selectbox(
                    f"Area Type — Property {i + 1}",
                    get_area_type_options(),
                    key=f"comp_area_{i}"
                )

            with col2:
                sqft = st.number_input(
                    f"Area in sq. ft. — Property {i + 1}",
                    min_value=100.0,
                    max_value=100000.0,
                    value=1000.0 + i * 200,
                    step=50.0,
                    key=f"comp_sqft_{i}"
                )

                bhk_value = st.selectbox(
                    f"BHK — Property {i + 1}",
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    index=min(i + 1, 7),
                    key=f"comp_bhk_{i}"
                )

            with col3:
                bath_value = st.selectbox(
                    f"Bathrooms — Property {i + 1}",
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    index=min(i + 1, 7),
                    key=f"comp_bath_{i}"
                )

                balcony_value = st.selectbox(
                    f"Balconies — Property {i + 1}",
                    [0, 1, 2, 3, 4],
                    index=1,
                    key=f"comp_balcony_{i}"
                )

            property_inputs.append({
                "Location": loc,
                "Area Type": area,
                "Area (sq. ft.)": sqft,
                "BHK": bhk_value,
                "Bathrooms": bath_value,
                "Balconies": balcony_value
            })

        compare_clicked = st.form_submit_button(
            "⚖️ Compare Properties",
            use_container_width=True
        )

    if compare_clicked:
        try:
            comparison_results = []

            for i, item in enumerate(property_inputs, start=1):
                model_input = make_property_data(
                    item["Location"],
                    item["Area Type"],
                    item["Area (sq. ft.)"],
                    item["BHK"],
                    item["Bathrooms"],
                    item["Balconies"]
                )

                estimated_price = get_model_prediction(model_input)

                comparison_results.append({
                    "Property": f"Property {i}",
                    "Location": item["Location"],
                    "Area Type": item["Area Type"],
                    "Area (sq. ft.)": item["Area (sq. ft.)"],
                    "BHK": item["BHK"],
                    "Bathrooms": item["Bathrooms"],
                    "Balconies": item["Balconies"],
                    "Estimated Price (Lakh)": round(estimated_price, 2)
                })

            comparison_df = pd.DataFrame(comparison_results)

            st.divider()
            st.markdown("### Comparison Results")

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                comparison_df,
                x="Property",
                y="Estimated Price (Lakh)",
                color="Property",
                text="Estimated Price (Lakh)",
                title="Estimated Property Price Comparison",
                color_discrete_sequence=[
                    "#10B981",
                    "#6366F1",
                    "#F59E0B",
                    "#06B6D4",
                    "#EC4899"
                ]
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig.update_yaxes(title="Estimated Price (Lakh)")
            fig.update_xaxes(title="Property")

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

            cheapest = comparison_df.loc[
                comparison_df["Estimated Price (Lakh)"].idxmin()
            ]

            st.success(
                f"Lowest estimated price: {cheapest['Property']} "
                f"at {format_price(cheapest['Estimated Price (Lakh)'])}."
            )

            st.download_button(
                "⬇️ Download Comparison",
                data=comparison_df.to_csv(index=False).encode("utf-8"),
                file_name="property_comparison.csv",
                mime="text/csv"
            )

            st.caption(
                "The lowest estimated price is not necessarily the best "
                "investment. Also compare location, quality, legal status, "
                "amenities, and future demand."
            )

        except Exception as e:
            st.error(f"Property comparison failed: {e}")

# ============================================================
# 15. PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":

    st.subheader("🕘 Prediction History")

    history = st.session_state.prediction_history

    if not history:
        st.info(
            "No predictions have been recorded in this session. "
            "Go to Price Prediction to begin."
        )

    else:
        history_df = pd.DataFrame(history)

        st.metric(
            "Total Recorded Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if "Predicted Price (Lakh)" in history_df.columns:
            prices = pd.to_numeric(
                history_df["Predicted Price (Lakh)"],
                errors="coerce"
            ).dropna()

            if not prices.empty:
                st.markdown("### Recorded Price Estimates")

                chart_df = history_df.copy()
                chart_df["Prediction Number"] = range(
                    1,
                    len(chart_df) + 1
                )

                fig = px.line(
                    chart_df,
                    x="Prediction Number",
                    y="Predicted Price (Lakh)",
                    markers=True,
                    title="Predicted Prices in This Session",
                    color_discrete_sequence=["#10B981"]
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>Real Estate AI Advisor</strong><br>
        Machine Learning | Property Price Prediction | Investment Support<br>
        Academic project demonstration. Verify dataset geography, price units,
        and model performance before using predictions for financial decisions.
    </div>
    """,
    unsafe_allow_html=True
)