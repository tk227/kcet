import streamlit as st
import pandas as pd
import requests
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# =====================
# CONFIG & STYLE
# =====================
st.set_page_config(page_title="KCET Predictor Pro", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    .main-title {text-align:center; color:#0073e6; font-size:40px; font-weight:700;}
    .sub-title {text-align:center; color:#555; font-size:20px;}
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# USER AUTH
# =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {"admin": {"password": "admin123", "cities": []}}

def signup_user(username, password):
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = {"password": password, "cities": []}
    return True

def login_user(username, password):
    user = st.session_state.users.get(username)
    if user and user["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = "🏠 Dashboard"
        return True
    return False

def logout_user():
    st.session_state.logged_in = False
    st.session_state.page = "🏠 Dashboard"

# =====================
# LOGIN / SIGNUP
# =====================
if not st.session_state.logged_in:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h2 class='main-title'>🎓 KCET College Predictor Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>AI + Cutoff Based Seat Prediction System</p>", unsafe_allow_html=True)
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])
        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login_user(username, password):
                    st.success(f"Welcome {username} 👋")
                    st.rerun()
                else:
                    st.error("Invalid username or password!")
        with tab2:
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if signup_user(new_user, new_pass):
                    st.success("Signup successful! Please log in.")
                else:
                    st.warning("Username already exists.")
    st.stop()

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/tk227/kcet/refs/heads/main/CET_Database_Final2020.csv"
    content = requests.get(url).content
    return pd.read_csv(io.StringIO(content.decode("utf-8")))

df = load_data()

# =====================
# SIDEBAR NAV
# =====================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"

menu = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Dashboard", "📊 Cutoff Based", "🤖 ML Based", "👤 Profile", "ℹ️ About"],
    index=["🏠 Dashboard", "📊 Cutoff Based", "🤖 ML Based", "👤 Profile", "ℹ️ About"].index(st.session_state.page)
)
st.session_state.page = menu

st.sidebar.markdown("---")
st.sidebar.info(f"👋 Logged in as **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    logout_user()
    st.rerun()

# =====================
# DASHBOARD
# =====================
if menu == "🏠 Dashboard":
    st.markdown("<h2 class='main-title'>🎯 KCET Predictor Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Select a prediction mode from the sidebar.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Colleges", df["College"].nunique())
    c2.metric("Total Branches", df["Branch"].nunique())
    c3.metric("Total Cities", df["Location"].nunique())

# =====================
# CUTOFF BASED
# =====================
elif menu == "📊 Cutoff Based":
    st.subheader("🎯 Cutoff-Based Prediction")

    rank = st.number_input("KCET Rank", min_value=1, value=1000)
    branch_list = st.multiselect("Preferred Branches", df["Branch"].unique())
    category = st.selectbox("Category", df.columns[4:], index=15)
    preferred_cities = st.multiselect("Preferred Cities", df["Location"].unique())

    if st.button("🔍 Predict"):
        if not branch_list:
            st.warning("⚠️ Please select at least one branch.")
            st.stop()

        seat_df = df[df["Branch"].isin(branch_list)].copy()

        if preferred_cities:
            seat_df = seat_df[seat_df["Location"].isin(preferred_cities)]

        seat_df["Cutoff"] = seat_df[category].astype(float)
        seat_df = seat_df[(seat_df["Cutoff"] > 0) & (rank <= seat_df["Cutoff"])]

        if seat_df.empty:
            st.warning("No colleges found for your rank.")
        else:
            st.success("✅ Eligible Colleges")
            st.dataframe(
                seat_df[["College", "Branch", "Location", "CETCode", "Cutoff"]]
                .sort_values("Cutoff")
            )

# =====================
# ML BASED
# =====================
elif menu == "🤖 ML Based":
    st.subheader("🤖 ML-Based Admission Prediction")

    rank = st.number_input("KCET Rank", min_value=1, value=1000)
    branch_list = st.multiselect("Preferred Branches", df["Branch"].unique())
    category = st.selectbox("Category", df.columns[4:], index=15)
    college_list = st.multiselect("Preferred Colleges (optional)", df["College"].unique())
    preferred_cities = st.multiselect("Preferred Cities (optional)", df["Location"].unique())

    if st.button("🚀 Predict"):
        if not branch_list:
            st.warning("⚠️ Please select at least one branch.")
            st.stop()

        df_long = df.melt(
            id_vars=["College", "Branch", "Location", "CETCode"],
            var_name="Category",
            value_name="Cutoff"
        )
        df_long.dropna(inplace=True)
        df_long["Admit"] = (df_long["Cutoff"] > 0).astype(int)

        features = ["College", "Branch", "Category", "Cutoff"]
        target = "Admit"

        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), ["College", "Branch", "Category"])],
            remainder="passthrough"
        )

        X = ct.fit_transform(df_long[features])
        y = df_long[target]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        rows = []
        colleges = college_list if college_list else df["College"].unique()

        for b in branch_list:
            for c in colleges:
                rows.append({
                    "College": c,
                    "Branch": b,
                    "Category": category,
                    "Cutoff": rank
                })

        user_df = pd.DataFrame(rows)

        if preferred_cities:
            user_df = user_df[user_df["College"].isin(
                df[df["Location"].isin(preferred_cities)]["College"].unique()
            )]

        if user_df.empty:
            st.warning("⚠️ No valid combinations found.")
            st.stop()

        user_df = user_df[features]
        user_X = ct.transform(user_df)

        user_df["Admission_Probability"] = model.predict_proba(user_X)[:, 1]

        st.success("✅ Prediction Results")
        st.dataframe(
            user_df.sort_values("Admission_Probability", ascending=False)
        )

# =====================
# PROFILE
# =====================
elif menu == "👤 Profile":
    user = st.session_state.users[st.session_state.username]
    cities = st.multiselect("Preferred Cities", df["Location"].unique(), default=user["cities"])

    if st.button("💾 Save"):
        user["cities"] = cities
        st.success("Saved!")

# =====================
# ABOUT
# =====================
elif menu == "ℹ️ About":
    st.markdown("""
    ### 📘 KCET Predictor Pro

    ![Course List Image](https://github.com/tk227/kcet/blob/main/Course%20list.png?raw=true)

    - Cutoff + ML based predictions  
    - Streamlit + Scikit-learn  
    - Secure login system  
    - Built for Karnataka CET aspirants 🎓
    """)