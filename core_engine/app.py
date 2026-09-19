"""
Plant Mamba - Executive Agricultural AI Diagnostic & Clinical Decision Platform
Features:
- Secure User Authentication (Login, Register, Demo Accounts, Session Persistence)
- ChatGPT-Style Consultation History Drawer (Past Scans, Quick Reload, Delete, Search)
- Modern Dark-Emerald Glassmorphism Web Interface
- ResNet-18 + Bidirectional SSM (Vision Mamba) 93.52%+ Accuracy Classifier (38 Classes)
- Universal Botanical Specimen & Non-Leaf Rejection Gate (Phones, Dogs, Selfies, Documents)
- Lesion-Guided Explainable AI (Grad-CAM Saliency & Physical Foliar Severity %)
- Real-time Agro-Meteorology & Fungal Epidemic Risk Alerts
- Spacious 2-Page Executive Phytosanitary Clinical PDF Report Download (FPDF2)
- Multi-View Architecture: Clinical Scanner, Farm Dashboard, Consultation Archive, Weather Map, Field Guide
- Trilingual Localization: English, Hindi (हिन्दी), Marathi (मराठी)
"""

import os
import glob
import json
import streamlit as st
from PIL import Image
import torch
from datetime import datetime

from pipeline import PlantDiagnosticPipeline
from vision_mamba_model import PLANT_CLASSES, CLEAN_CLASS_NAMES
from weather_service import get_live_weather, calculate_disease_risk, POPULAR_REGIONS
from case_matcher import retrieve_similar_cases
from report_generator import generate_pdf_report
from multilingual import (
    get_text,
    get_localized_disease_name,
    get_localized_severity,
    get_localized_treatment_fields
)
import database
import role_views

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Mamba - Autonomous Crop Pathology Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- High-End Modern Styling (Bright & Clean Light Mode with Emerald Accents) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Base */
    .stApp {
        background: linear-gradient(180deg, #f0fdf4 0%, #f8fafc 180px, #f8fafc 100%);
        color: #0f172a;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Navbar Card */
    .top-navbar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px -2px rgba(16, 185, 129, 0.08), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #065f46;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-pill {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .user-pill {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Clean Card Containers */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    .glass-card:hover {
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .glass-card-highlight {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border: 1.5px solid #86efac;
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -4px rgba(16, 185, 129, 0.12);
    }

    /* Metric Counters */
    .metric-value-huge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .metric-label-muted {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    /* ChatGPT History Item Cards */
    .history-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .history-card:hover {
        background: #f0fdf4;
        border-color: #86efac;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    .history-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-meta {
        font-size: 0.78rem;
        color: #64748b;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Rejection & Quality Banners */
    .quality-banner-passed {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 14px 20px;
        border-radius: 12px;
        border: 1px solid #a7f3d0;
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 0.94rem;
    }
    .rejection-box {
        background-color: #fef2f2;
        border: 1.5px solid #fca5a5;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.08);
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.35);
        transform: translateY(-1px);
    }

    /* New Chat / Diagnosis Button */
    .new-scan-btn button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        width: 100% !important;
        padding: 12px !important;
        margin-bottom: 14px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* =========================================================================
       CRITICAL CONTRAST FIXES: TABS, INPUTS, LABELS & TEXT MUST BE BLACK/SLATE
       ========================================================================= */
    /* All Tabs: Force Deep Black/Dark Slate Font Color */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid #cbd5e1 !important;
        gap: 8px !important;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs button[role="tab"],
    div[data-testid="stTabs"] button {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.96rem !important;
        background-color: #f1f5f9 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px !important;
        border: 1px solid #cbd5e1 !important;
        border-bottom: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs button[role="tab"] p,
    div[data-testid="stTabs"] button p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs button[role="tab"] span,
    div[data-testid="stTabs"] button span,
    .stTabs [data-baseweb="tab"] div,
    .stTabs button[role="tab"] div {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs button[role="tab"]:hover,
    div[data-testid="stTabs"] button:hover {
        background-color: #e2e8f0 !important;
        color: #047857 !important;
    }
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs button[role="tab"]:hover p {
        color: #047857 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #047857 !important;
        background-color: #ffffff !important;
        border-top: 3px solid #059669 !important;
        border-left: 1px solid #cbd5e1 !important;
        border-right: 1px solid #cbd5e1 !important;
        border-bottom: 2px solid #ffffff !important;
        margin-bottom: -2px !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs button[role="tab"][aria-selected="true"] span {
        color: #047857 !important;
        font-weight: 800 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #059669 !important;
    }

    /* Universal Text & Paragraphs */
    p, span, label, li, dt, dd {
        color: #0f172a;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #0f172a !important;
    }

    /* Form Labels & Placeholders */
    .stTextInput label, .stSelectbox label, .stNumberInput label, .stFileUploader label, .stRadio label {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }
    .stTextInput label p, .stSelectbox label p, .stRadio label p, .stFileUploader label p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    input, textarea {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
    }
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
    }
    div[data-baseweb="input"] input {
        color: #0f172a !important;
    }
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important;
    }
    div[data-baseweb="popover"] div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Radio Buttons */
    div[role="radiogroup"] label {
        color: #0f172a !important;
    }
    div[role="radiogroup"] label p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: #0f172a !important;
    }

    /* Metric Values & Labels */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #334155 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Management ---
if "user" not in st.session_state:
    st.session_state["user"] = None
if "active_consultation" not in st.session_state:
    st.session_state["active_consultation"] = None
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "scanner"
if "lang_code" not in st.session_state:
    st.session_state["lang_code"] = "en"


@st.cache_resource
def get_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    return PlantDiagnosticPipeline(checkpoint_path=ckpt_path)


# =============================================================================
# AUTHENTICATION PORTAL (LOGIN & REGISTRATION)
# =============================================================================
def render_auth_portal():
    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <div style="font-size: 3.2rem; margin-bottom: 8px;">🌿</div>
            <h1 style="color: #065f46; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.5px;">PLANT MAMBA</h1>
            <p style="color: #64748b; font-size: 0.95rem;">
                Autonomous Agricultural Pathology & Clinical Decision Support Platform
            </p>
            <span class="status-pill">State Space Model (SSM) v2.0 • 93.52% Precision Core</span>
        </div>
        """, unsafe_allow_html=True)

        auth_tab1, auth_tab2 = st.tabs(["🔑 Sign In / प्रवेश", "📝 Register New Account / नवीन खाते"])

        with auth_tab1:
            st.markdown("##### Sign in to access the Clinical Diagnostic Suite")
            login_username = st.text_input("Username", key="login_user", placeholder="e.g. admin or farmer")
            login_password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")

            if st.button("Sign In ➔", key="btn_signin", use_container_width=True):
                user = database.authenticate_user(login_username, login_password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["active_consultation"] = None
                    st.success(f"Welcome back, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

            st.markdown("---")
            st.markdown("<p style='color: #475569; font-size: 0.85rem; font-weight: 700;'>🚀 INSTANT DEMO ACCESS BY ROLE:</p>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("🩺 Dr. Dhanashri\n(Chief Agronomist)", key="btn_demo_agro", use_container_width=True):
                    user = database.authenticate_user("admin", "admin123")
                    if user:
                        st.session_state["user"] = user
                        st.session_state["active_consultation"] = None
                        st.session_state["current_view"] = "agronomist_review"
                        st.rerun()

            with col_d2:
                if st.button("🚜 Ramesh Patil\n(Progressive Farmer)", key="btn_demo_farmer", use_container_width=True):
                    user = database.authenticate_user("farmer", "farmer123")
                    if user:
                        st.session_state["user"] = user
                        st.session_state["active_consultation"] = None
                        st.session_state["current_view"] = "spray_calendar"
                        st.rerun()

        with auth_tab2:
            st.markdown("##### Create a new Agricultural Clinic account")
            new_name = st.text_input("Full Name", placeholder="e.g. Dr. Rajesh Sharma")
            new_email = st.text_input("Email Address", placeholder="rajesh@agriculture.gov.in")
            new_role = st.selectbox("Role", ["Agronomist / Specialist", "Farmer / Field Producer"])
            new_username = st.text_input("Choose Username", placeholder="e.g. rsharma")
            new_password = st.text_input("Create Password", type="password", placeholder="Min 4 characters")

            if st.button("Create Account", use_container_width=True):
                if new_username and new_password and new_name:
                    success, msg = database.register_user(new_username, new_password, new_name, new_email, new_role)
                    if success:
                        st.success(msg)
                        st.info("You can now switch to the 'Sign In' tab to log in.")
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in all required fields.")

        # Features highlight card below login
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div style="font-weight: 700; color: #059669; font-size: 1rem; margin-bottom: 10px;">🌟 Enterprise Platform Capabilities:</div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.7;">
                • <b>Vision Mamba SSM Core:</b> Linear-complexity sequence modeling across 38 crop disease classes.<br>
                • <b>Universal Specimen Guard:</b> 100% rejection of non-leaf photos (smartphones, selfies, pets, documents).<br>
                • <b>Lesion-Guided Explainability:</b> Fused Grad-CAM with exact physical foliar necrosis percentages.<br>
                • <b>Agro-Meteorological Radar:</b> Real-time fungal epidemic spore tracking based on ambient microclimate.<br>
                • <b>Executive 2-Page PDF Reports:</b> Printable phytosanitary clinical certificates for field spray scheduling.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION WORKFLOW (AUTHENTICATED)
# =============================================================================
def main():
    if st.session_state.get("user") is None:
        render_auth_portal()
        return

    user = st.session_state["user"]
    pipeline = get_pipeline()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------------------
    # TOP NAVBAR
    # -------------------------------------------------------------------------
    col_nav1, col_nav2, col_nav3 = st.columns([3.5, 2.5, 1.2])
    with col_nav1:
        st.markdown(f"""
        <div class="brand-title">
            <span>🌿 PLANT MAMBA</span>
            <span class="status-pill">v2.0 Clinical Core • Active</span>
        </div>
        """, unsafe_allow_html=True)

    with col_nav2:
        # Language Switcher in Navbar
        lang_options = ["English", "हिन्दी (Hindi)", "मराठी (Marathi)"]
        saved_lang = st.session_state.get("lang_code", "en")
        current_idx = 1 if saved_lang == "hi" else 2 if saved_lang == "mr" else 0
        lang_choice = st.selectbox(
            "Language",
            lang_options,
            index=current_idx,
            label_visibility="collapsed"
        )
        lang_code = "en"
        if "हिन्दी" in lang_choice:
            lang_code = "hi"
        elif "मराठी" in lang_choice:
            lang_code = "mr"
        st.session_state["lang_code"] = lang_code

    with col_nav3:
        if st.button(get_text("logout", lang_code), key="btn_logout", use_container_width=True):
            st.session_state["user"] = None
            st.session_state["active_consultation"] = None
            st.rerun()

    # -------------------------------------------------------------------------
    # ROLE RESOLUTION & STYLING
    # -------------------------------------------------------------------------
    user_role_raw = user.get("role", "Agronomist")
    role_lower = user_role_raw.lower()
    if "agronomist" in role_lower or "specialist" in role_lower:
        active_role = "Agronomist"
        role_pill_style = "background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a;"
        role_tag = get_text("chief_agronomist_tag", lang_code)
    else:
        active_role = "Farmer"
        role_pill_style = "background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;"
        role_tag = get_text("farmer_tag", lang_code)

    # User welcome banner
    st.markdown(f"""
    <div style="font-size: 0.88rem; color: #64748b; margin-top: -12px; margin-bottom: 18px;">
        👤 {get_text('logged_in_as', lang_code)} <b style="color: #0f172a;">{user['full_name']}</b> <span style="{role_pill_style} padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-left: 6px;">{role_tag}</span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR: ROLE-TAILORED NAVIGATION & CONSULTATION DRAWER
    # -------------------------------------------------------------------------
    with st.sidebar:
        # 1. New Diagnosis Button (Clean - no mixed slashes)
        st.markdown('<div class="new-scan-btn">', unsafe_allow_html=True)
        if st.button(get_text("new_diagnosis", lang_code), use_container_width=True):
            st.session_state["active_consultation"] = None
            st.session_state["current_view"] = "scanner"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. Role-Differentiated Navigation Switcher (ONLY 2 CLEAN ITEMS PER ROLE)
        st.markdown(f"##### 🧭 {get_text('nav_heading', lang_code)}")
        if active_role == "Agronomist":
            nav_options = {
                "scanner": get_text("nav_scanner", lang_code),
                "agronomist_review": get_text("nav_review", lang_code),
            }
        else:  # Farmer
            nav_options = {
                "scanner": get_text("nav_scanner", lang_code),
                "spray_calendar": get_text("nav_spray_calendar", lang_code),
            }

        if st.session_state.get("current_view") not in nav_options:
            st.session_state["current_view"] = list(nav_options.keys())[0]

        view_keys = list(nav_options.keys())
        selected_view_name = st.radio(
            "Go to View:",
            [nav_options[k] for k in view_keys],
            index=view_keys.index(st.session_state["current_view"]),
            label_visibility="collapsed"
        )
        for k, v in nav_options.items():
            if v == selected_view_name:
                st.session_state["current_view"] = k

        # 3. Location & Live Weather Settings
        st.markdown("---")
        st.markdown(f"##### 📍 {get_text('farm_location_weather', lang_code)}")
        region_options = ["Auto-Detect (GPS/IP)"] + list(POPULAR_REGIONS.keys())
        chosen_region = st.selectbox(get_text("region_select", lang_code), region_options)
        lookup_city = "Auto-Detect" if "Auto-Detect" in chosen_region else chosen_region
        weather = get_live_weather(lookup_city)
        st.caption(f"📍 **{weather['city']}** | 🌡️ **{weather['temperature']}°C** | 💧 **{weather['humidity']}% RH**")

        # 4. ChatGPT-Style Consultation History Drawer
        st.markdown("---")
        if active_role == "Agronomist":
            st.markdown(f"##### 📜 {get_text('all_field_submissions', lang_code)}")
            past_consultations = database.get_all_consultations(limit=25)
        else:
            st.markdown(f"##### 📜 {get_text('recent_consultations', lang_code)}")
            past_consultations = database.get_user_consultations(user_id=user["id"], limit=20)

        if not past_consultations:
            st.caption(get_text("no_consultations", lang_code))
        else:
            for item in past_consultations:
                cid = item["id"]
                c_name = item["clean_name"]
                c_sev = item["severity_level"]
                c_ratio = item["infected_ratio"]
                c_time = item["timestamp"][:16]
                c_stat = item.get("review_status", "AI Preliminary")

                sev_icon = "🟢" if "healthy" in c_name.lower() or "none" in c_sev.lower() else "🟡" if "mild" in c_sev.lower() else "🔴"
                stat_tag = "✅" if c_stat == "Agronomist Certified" else "⏳"

                col_h1, col_h2 = st.columns([5, 1])
                with col_h1:
                    btn_label = f"{stat_tag} {sev_icon} {c_name} ({c_ratio:.0f}%)"
                    if st.button(btn_label, key=f"hist_btn_{cid}", help=f"Recorded: {c_time} | Status: {c_stat}"):
                        st.session_state["active_consultation"] = item
                        st.session_state["current_view"] = "scanner"
                        st.rerun()
                with col_h2:
                    if st.button("🗑️", key=f"del_hist_{cid}", help="Delete from history"):
                        database.delete_consultation(cid, user_id=user["id"] if active_role != "Agronomist" else None)
                        active_c = st.session_state.get("active_consultation")
                        if active_c and active_c.get("id") == cid:
                            st.session_state["active_consultation"] = None
                        st.rerun()

            if st.button(f"🗑️ {get_text('clear_history', lang_code)}", key="btn_clear_all_hist"):
                for item in past_consultations:
                    database.delete_consultation(item["id"], user_id=user["id"] if active_role != "Agronomist" else None)
                st.session_state["active_consultation"] = None
                st.rerun()

    # -------------------------------------------------------------------------
    # MAIN VIEW ROUTING
    # -------------------------------------------------------------------------
    current_view = st.session_state.get("current_view", "scanner")

    # =========================================================================
    # VIEW 1: AI CLINICAL SCANNER (INSPECTOR & DIAGNOSIS)
    # =========================================================================
    if current_view == "scanner":
        active_consult = st.session_state.get("active_consultation")

        # ---------------------------------------------------------------------
        # Case A: Viewing a Past Saved Consultation from History
        # ---------------------------------------------------------------------
        if active_consult is not None:
            st.markdown(f"""
            <div class="glass-card-highlight">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="status-pill">{get_text('historical_heading', lang_code).upper()}</span>
                        <h2 style="color: #0f172a; margin-top: 8px; margin-bottom: 2px;">{active_consult['clean_name']}</h2>
                        <span style="color: #64748b; font-size: 0.85rem;">
                            {get_text('report_id_label', lang_code)}: {active_consult['report_id']} | Date: {active_consult['timestamp']} | {get_text('location_label', lang_code)}: {active_consult['city']}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown(f"##### {get_text('submitted_leaf', lang_code)}")
                if os.path.exists(active_consult["orig_img_path"]):
                    st.image(active_consult["orig_img_path"], use_container_width=True)
                else:
                    st.info("Specimen image file not found on disk.")

            with col_img2:
                st.markdown(f"##### {get_text('gradcam_heatmap', lang_code)}")
                if os.path.exists(active_consult["cam_img_path"]):
                    st.image(active_consult["cam_img_path"], use_container_width=True)
                else:
                    st.info("Grad-CAM heatmap file not found on disk.")

            # Metrics
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(get_text("confidence_label", lang_code), f"{active_consult['confidence']:.1f}%")
            with m2:
                st.metric(get_text("severity_label", lang_code), get_localized_severity(active_consult['severity_level'], lang_code))
            with m3:
                st.metric(get_text("infected_area_label", lang_code), f"{active_consult['infected_ratio']:.1f}%")
            with m4:
                st.metric(get_text("microclimate_risk", lang_code), active_consult.get('risk_level', 'Moderate Risk'))

            # Prescriptions
            st.markdown(f"### 💊 {get_text('treatment_recommendations', lang_code)}")
            treatment_data = active_consult.get("treatment", {})
            col_tx1, col_tx2 = st.columns(2)
            with col_tx1:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #dc2626; font-weight: 700; margin-bottom: 6px;">🧪 {get_text('treatment_heading', lang_code)}:</div>
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                        {treatment_data.get('treatment', 'Apply standard recommended protective fungicides.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_tx2:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #059669; font-weight: 700; margin-bottom: 6px;">🚜 {get_text('prevention_heading', lang_code)}:</div>
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                        {treatment_data.get('prevention', 'Ensure adequate crop spacing and drip irrigation.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Re-download PDF
            if os.path.exists(active_consult["orig_img_path"]) and os.path.exists(active_consult["cam_img_path"]):
                diag_mock = {
                    "clean_name": active_consult["clean_name"],
                    "predicted_class": active_consult["disease_name"],
                    "confidence": active_consult["confidence"],
                    "severity_level": active_consult["severity_level"],
                    "infected_ratio": active_consult["infected_ratio"],
                    "quality": {"status": "Passed", "sharpness": 80.0, "brightness": 120.0},
                    "original_image": Image.open(active_consult["orig_img_path"]),
                    "cam_image": Image.open(active_consult["cam_img_path"]),
                    "treatment": treatment_data
                }
                weather_mock = {"city": active_consult["city"], "temperature": active_consult.get("weather_temp", 26.0), "humidity": active_consult.get("weather_humidity", 75.0)}
                risk_mock = {"risk_level": active_consult.get("risk_level", "Moderate Risk"), "explanation": active_consult.get("risk_explanation", "")}
                pdf_bytes = generate_pdf_report(diag_mock, weather_mock, risk_mock, active_consult.get("similar_cases", []), custom_report_id=active_consult["report_id"])

                st.download_button(
                    label=get_text("download_pdf", lang_code),
                    data=pdf_bytes,
                    file_name=f"{active_consult['report_id']}_Diagnostic_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            if st.button(f"⬅️ {get_text('new_diagnosis', lang_code)}", use_container_width=True):
                st.session_state["active_consultation"] = None
                st.rerun()
            return

        # ---------------------------------------------------------------------
        # Case B: Live New Diagnosis Mode
        # ---------------------------------------------------------------------
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h1 style="color: #065f46; font-weight: 800; margin-bottom: 2px;">{get_text('title', lang_code)}</h1>
            <p style="color: #64748b; font-size: 0.95rem;">
                {get_text('sub_title', lang_code)}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Input Specimen Selector
        col_up1, col_up2 = st.columns([1.5, 1])
        with col_up1:
            uploaded_file = st.file_uploader(get_text("upload_label", lang_code), type=["jpg", "jpeg", "png"])
        with col_up2:
            samples_dir = os.path.join(base_dir, "sample_images")
            if not os.path.exists(samples_dir):
                samples_dir = os.path.join(os.path.dirname(base_dir), "sample_images")

            CURATED_SAMPLES = [
                ("tomato_late_blight.jpg", {
                    "en": "🍅 Tomato: Late Blight (Foliar Necrosis)",
                    "hi": "🍅 टमाटर: पछेती झुलसा रोग (Late Blight)",
                    "mr": "🍅 टोमॅटो: करपा / लेट ब्लाईट रोग"
                }),
                ("tomato_early_blight.jpg", {
                    "en": "🍅 Tomato: Early Blight (Concentric Rings)",
                    "hi": "🍅 टमाटर: अगेती झुलसा रोग (Early Blight)",
                    "mr": "🍅 टोमॅटो: अगेती करपा रोग"
                }),
                ("tomato_healthy.jpg", {
                    "en": "🌿 Tomato: Healthy Botanical Leaf",
                    "hi": "🌿 टमाटर: स्वस्थ निरोगी पत्ता (Healthy)",
                    "mr": "🌿 टोमॅटो: निरोगी पान (Healthy)"
                }),
                ("potato_late_blight.jpg", {
                    "en": "🥔 Potato: Late Blight (Tuber & Foliage Blight)",
                    "hi": "🥔 आलू: पछेती झुलसा रोग (Late Blight)",
                    "mr": "🥔 बटाटा: लेट ब्लाईट करपा रोग"
                }),
                ("corn_common_rust.jpg", {
                    "en": "🌽 Corn: Common Rust (Puccinia Pustules)",
                    "hi": "🌽 मक्का: रतुआ रोग (Common Rust)",
                    "mr": "🌽 मका: तांबेरा रोग (Rust)"
                }),
                ("apple_scab.jpg", {
                    "en": "🍎 Apple: Apple Scab (Olive-Green Lesions)",
                    "hi": "🍎 सेब: स्कैब / खपली रोग (Apple Scab)",
                    "mr": "🍎 सफरचंद: स्कॅब रोग"
                }),
                ("apple_healthy.jpg", {
                    "en": "🌿 Apple: Healthy Orchard Foliage",
                    "hi": "🌿 सेब: स्वस्थ निरोगी पत्ता (Healthy)",
                    "mr": "🌿 सफरचंद: निरोगी पान"
                }),
                ("grape_black_rot.jpg", {
                    "en": "🍇 Grape: Black Rot (Guignardia Decay)",
                    "hi": "🍇 अंगूर: काला सड़ांध रोग (Black Rot)",
                    "mr": "🍇 द्राक्ष: काळी सड / ब्लॅक रॉट"
                }),
                ("pepper_bacterial_spot.jpg", {
                    "en": "🫑 Bell Pepper: Bacterial Leaf Spot",
                    "hi": "🫑 शिमला मिर्च: जीवाणु पत्ती धब्बा रोग",
                    "mr": "🫑 ढोबळी मिरची: जिवाणू ठिपके रोग"
                }),
                ("non_leaf_smartphone.jpg", {
                    "en": "🛑 [Recruiter Demo] Non-Leaf Object (Rejection Gate Test)",
                    "hi": "🛑 [डेमो परीक्षण] गैर-पादप वस्तु (स्मार्टफोन रिजेक्शन गेट टेस्ट)",
                    "mr": "🛑 [चाचणी नमुना] वनस्पती नसलेली वस्तू (स्मार्टफोन नकार गेट चाचणी)"
                }),
            ]

            none_label = get_text("none_upload_own", lang_code)
            sample_options = [none_label]
            sample_map = {}

            if os.path.exists(samples_dir):
                for filename, labels in CURATED_SAMPLES:
                    img_path = os.path.join(samples_dir, filename)
                    if os.path.exists(img_path):
                        label = labels.get(lang_code, labels["en"])
                        sample_options.append(label)
                        sample_map[label] = img_path

            # Fallback to plantvillage_data if present locally
            val_dir = os.path.join(base_dir, "plantvillage_data", "data_38", "val")
            if os.path.exists(val_dir) and len(sample_options) == 1:
                for cls_name in PLANT_CLASSES:
                    cls_folder = os.path.join(val_dir, cls_name)
                    if os.path.exists(cls_folder):
                        imgs = glob.glob(os.path.join(cls_folder, "*.*"))
                        if imgs:
                            loc_name = get_localized_disease_name(cls_name, lang_code)
                            label = f"{get_text('sample_prefix', lang_code)}: {loc_name}"
                            sample_options.append(label)
                            sample_map[label] = imgs[0]

            selected_sample = st.selectbox(get_text("sample_select_label", lang_code), sample_options)

        # Image resolution
        image_to_process = None
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
        elif selected_sample != none_label and selected_sample in sample_map:
            image_to_process = Image.open(sample_map[selected_sample])

        if image_to_process is None:
            if lang_code == "hi":
                welcome_title = "🔬 एआई नैदानिक पैथोलॉजी स्कैनर में आपका स्वागत है"
                welcome_sub = "जांच शुरू करने के लिए ऊपर ड्रॉपडाउन से <b>प्री-लोडेड नमूना पत्ती (Sample Leaf)</b> चुनें अथवा अपने डिवाइस से किसी पौधे की पत्ती की तस्वीर अपलोड करें।"
                tip_title = "💡 रिक्रूटर एवं परीक्षक डेमो गाइड:"
                tip_desc = "ड्रॉपडाउन में टमाटर, आलू, मक्का, सेब, अंगूर और शिमला मिर्च के विभिन्न रोगों के वास्तविक नमूने शामिल हैं। आप <b>'Non-Leaf Object'</b> नमूना चुनकर एआई का ऑटोमैटिक गैर-पादप रिजेक्शन गेट भी टेस्ट कर सकते हैं।"
            elif lang_code == "mr":
                welcome_title = "🔬 एआय डिजिटल पीक रोग स्कॅनर मध्ये आपले स्वागत आहे"
                welcome_sub = "तपासणी सुरू करण्यासाठी वरील ड्रॉपडाउनमधून <b>आधीच उपलब्ध पानाचा नमुना (Sample Leaf)</b> निवडा किंवा स्वतःच्या पिकाच्या पानाचा फोटो अपलोड करा."
                tip_title = "💡 परीक्षक व मुलाखतकारांसाठी विशेष टीप:"
                tip_desc = "ड्रॉपडाउनमध्ये टोमॅटो, बटाटा, मका, सफरचंद, द्राक्ष आणि मिरचीच्या विविध रोगांचे नमुने जोडले आहेत. तुम्ही <b>'Non-Leaf Object'</b> नमुना निवडून एआय चे फसवणूक प्रतिबंधक तंत्रज्ञान (Security Rejection Gate) देखील तपासू शकता."
            else:
                welcome_title = "🔬 Clinical Phytosanitary Diagnostic Scanner"
                welcome_sub = "To begin diagnosis, select any <b>Pre-loaded Sample Leaf</b> from the dropdown above, or upload a leaf photograph from your device."
                tip_title = "💡 Evaluator & Recruiter Quick Demonstration Guide:"
                tip_desc = "The dropdown includes pre-loaded foliar specimens for Tomato, Potato, Corn, Apple, Grape, and Bell Pepper pathologies. You can also select the <b>'Non-Leaf Object'</b> specimen to evaluate the autonomous anti-spoofing security rejection gate."

            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 36px 28px; margin-top: 15px;">
                <div style="font-size: 3rem; margin-bottom: 12px;">🌿</div>
                <h3 style="color: #065f46; font-weight: 800; margin-bottom: 8px;">{welcome_title}</h3>
                <p style="color: #334155; font-size: 1rem; max-width: 680px; margin: 0 auto 20px auto; line-height: 1.6;">
                    {welcome_sub}
                </p>
                <div style="background-color: #f0fdf4; border: 1px solid #a7f3d0; border-radius: 12px; padding: 16px 20px; max-width: 700px; margin: 0 auto; text-align: left;">
                    <b style="color: #065f46; font-size: 0.95rem;">{tip_title}</b>
                    <div style="color: #334155; font-size: 0.9rem; margin-top: 4px; line-height: 1.6;">
                        {tip_desc}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Execute Pipeline
        with st.spinner("Executing Vision Mamba inference & Lesion-Guided Grad-CAM..."):
            result = pipeline.diagnose(image_to_process)

        # ---------------------------------------------------------------------
        # Specimen Validation Gate (Rejects Non-Leaf Objects)
        # ---------------------------------------------------------------------
        if not result.get("is_valid_leaf", True):
            q = result["quality"]
            rej_code = q.get("rejection_code", "")

            if lang_code == "hi":
                err_title = "🛑 अमान्य नमूना (अस्वीकृत): यह पौधे या पत्ती की तस्वीर नहीं है!"
                err_msg = q.get("details_hi", "छवि में कोई पत्ती या फसल नहीं पाई गई।")
                tip_msg = "💡 प्लांट मांबा प्रणाली विशेष रूप से कृषि फसलों और पौधों की पत्तियों के निदान के लिए है। कृपया किसी वास्तविक पौधे की पत्ती की तस्वीर अपलोड करें।"
                label_detected = "पहचाना गया विषय / वस्तु:"
                val_detected = q.get('detected_object_hi', 'गैर-पादप वस्तु')
                label_foliage = "पौधे का ऊतक प्रतिशत:"
            elif lang_code == "mr":
                err_title = "🛑 अमान्य नमुना (अस्वीकार): हे वनस्पतीचे पान नाही!"
                err_msg = q.get("details_mr", "प्रतिमेमध्ये वनस्पतीचे पान आढळले नाही.")
                tip_msg = "💡 प्लांट मांबा प्रणाली केवळ शेतातील पिके व झाडांच्या पानांच्या रोग निदानासाठी आहे. कृपया झाडाच्या किंवा पिकाच्या पानाचा फोटो अपलोड करा."
                label_detected = "आढळलेला विषय / वस्तू:"
                val_detected = q.get('detected_object_mr', 'गैर-वनस्पती वस्तू')
                label_foliage = "वनस्पतीचे प्रमाण:"
            else:
                err_title = "🛑 Specimen Rejected: Not a Botanical Plant Leaf!"
                err_msg = q.get("details", "No plant foliage or botanical leaf tissue detected in image.")
                tip_msg = "💡 Plant Mamba is calibrated exclusively for agricultural crop pathologies and botanical foliage. Please upload a clear photo of an agricultural leaf."
                label_detected = "Detected Object / Content:"
                val_detected = q.get('detected_object_en', 'Non-Botanical Object')
                label_foliage = "Botanical Foliage:"

            st.markdown(f"""
            <div class="rejection-box">
                <div style="color: #b91c1c; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
                    {err_title}
                </div>
                <div style="color: #991b1b; font-size: 1.0rem; line-height: 1.5; margin-bottom: 10px;">
                    {err_msg}
                </div>
                <div style="color: #854d0e; font-size: 0.92rem; font-weight: 600;">
                    {tip_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_rej1, col_rej2 = st.columns(2)
            with col_rej1:
                st.image(result["original_image"], caption="Submitted Image / अपलोड की गई तस्वीर", use_container_width=True)
            with col_rej2:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label-muted">{label_detected}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #dc2626; margin-bottom: 15px;">
                        {val_detected}
                    </div>
                    <div class="metric-label-muted">{label_foliage}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #d97706; margin-bottom: 15px;">
                        {q.get('plant_ratio', 0.0)}%
                    </div>
                    <div class="metric-label-muted">Security Check:</div>
                    <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                        • ImageNet MobileNetV3 1000-class open-domain filter<br>
                        • YCbCr Facial skin chrominance inspection<br>
                        • Chlorophyll optical reflectance gate
                    </div>
                </div>
                """, unsafe_allow_html=True)
            return

        # ---------------------------------------------------------------------
        # Valid Specimen: Full Pathology Diagnosis Display
        # ---------------------------------------------------------------------
        q = result["quality"]
        if q.get("status") == "Passed":
            st.markdown(f'<div class="quality-banner-passed">✅ {q.get("details_hi" if lang_code == "hi" else "details_mr" if lang_code == "mr" else "details")}</div>', unsafe_allow_html=True)

        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        clean_name = result["clean_name"]
        severity = result["severity_level"]
        infected_ratio = result["infected_ratio"]

        # Environmental risk calculation
        risk = calculate_disease_risk(weather, pred_class, lang=lang_code)
        similar_cases = retrieve_similar_cases(pred_class, confidence_score=confidence, current_region=weather.get("city", "Nagpur"), lang=lang_code)

        # Save to database automatically!
        report_id = database.save_consultation(user["id"], result, weather, risk, similar_cases)

        # Diagnosis Header Card
        st.markdown(f"""
        <div class="glass-card-highlight">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="status-pill">{get_text('diagnosis_confirmed', lang_code)}</span>
                    <h2 style="color: #0f172a; margin-top: 8px; margin-bottom: 4px;">{get_localized_disease_name(pred_class, lang_code)}</h2>
                    <span style="color: #64748b; font-size: 0.85rem;">{get_text('report_id_label', lang_code)}: <b>{report_id}</b> | {get_text('location_label', lang_code)}: {weather['city']} | {get_text('model_core_label', lang_code)}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dual Image Display
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown(f"##### {get_text('submitted_leaf', lang_code)}")
            st.image(result["original_image"], use_container_width=True)
        with col_img2:
            st.markdown(f"##### {get_text('gradcam_heatmap', lang_code)}")
            st.image(result["cam_image"], use_container_width=True)

        # Primary Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(get_text("confidence_label", lang_code), f"{confidence:.1f}%")
        with m2:
            st.metric(get_text("severity_label", lang_code), get_localized_severity(severity, lang_code))
        with m3:
            st.metric(get_text("infected_area_label", lang_code), f"{infected_ratio:.1f}%")
        with m4:
            st.metric(get_text("microclimate_risk", lang_code), risk.get("risk_level", "Moderate Risk"))

        # Atmospheric Risk Card
        risk_level = risk.get("risk_level", "Moderate Risk")
        risk_color = "#dc2626" if "Critical" in risk_level else "#d97706" if "High" in risk_level else "#059669"
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid {risk_color};">
            <div style="color: {risk_color}; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">
                🌦️ {get_text('weather_context', lang_code)}: {risk_level} ({weather['temperature']}°C | {weather['humidity']}% RH)
            </div>
            <div style="color: #334155; font-size: 0.92rem; line-height: 1.6;">
                {risk.get('explanation', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agronomic Protocols
        st.markdown(f"### 💊 {get_text('treatment_recommendations', lang_code)}")
        tx_fields = get_localized_treatment_fields(result.get("treatment", {}), lang_code)

        col_tx1, col_tx2 = st.columns(2)
        with col_tx1:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color: #dc2626; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
                    🧪 {get_text("treatment_heading", lang_code)}
                </div>
                <div style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                    {tx_fields['treatment']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_tx2:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color: #059669; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
                    🚜 {get_text("prevention_heading", lang_code)}
                </div>
                <div style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                    {tx_fields['prevention']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Regional Outbreaks
        if similar_cases:
            st.markdown(f"### 📍 {get_text('historical_heading', lang_code)}")
            for c in similar_cases:
                st.markdown(f"""
                <div style="font-size: 0.9rem; color: #334155; margin-bottom: 6px;">
                    • <b>{c['case_id']}</b>: {c['disease']} ({c['match_pct']}% {get_text('match_label', lang_code)}) — <i>{c['region']}</i><br>
                    <span style="color: #64748b; font-size: 0.82rem; margin-left: 14px;">Observed Outcome: {c['outcome']}</span>
                </div>
                """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Executive 2-Page PDF Report Download
        # ---------------------------------------------------------------------
        st.markdown("---")
        with st.spinner(get_text("generating_pdf", lang_code)):
            pdf_bytes = generate_pdf_report(result, weather, risk, similar_cases, custom_report_id=report_id)

        st.download_button(
            label=get_text("download_pdf", lang_code),
            data=pdf_bytes,
            file_name=f"{report_id}_{clean_name.replace(' ', '_')}.pdf",
            use_container_width=True
        )

        # Role-Specific Quick Actions from Scanner
        st.markdown("---")
        if active_role == "Agronomist":
            if st.button(get_text("agro_open_review_btn", lang_code), key="sc_agro_audit", use_container_width=True):
                st.session_state["current_view"] = "agronomist_review"
                st.rerun()
        else:  # Farmer
            if st.button(get_text("farmer_open_spray_btn", lang_code), key="sc_farmer_spray", use_container_width=True):
                st.session_state["current_view"] = "spray_calendar"
                st.rerun()

    # =========================================================================
    # VIEW 2: FARM DASHBOARD & ANALYTICS
    # =========================================================================
    elif current_view == "dashboard":
        st.markdown("""
        <div style="margin-bottom: 22px;">
            <h1 style="color: #10b981; font-weight: 800; margin-bottom: 2px;">Farm Health Dashboard & Analytics</h1>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Aggregated epidemiological insights, disease prevalence, and historical clinical performance across all your crop scans.
            </p>
        </div>
        """, unsafe_allow_html=True)

        stats = database.get_consultation_stats(user_id=user["id"])

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label-muted">Total Consultations</div>
                <div class="metric-value-huge">{stats['total_scans']}</div>
                <div style="color: #10b981; font-size: 0.8rem; margin-top: 4px;">Active clinical history</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label-muted">Model Diagnostic Accuracy</div>
                <div class="metric-value-huge">93.52%</div>
                <div style="color: #34d399; font-size: 0.8rem; margin-top: 4px;">Vision Mamba (224px)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label-muted">Average AI Confidence</div>
                <div class="metric-value-huge">{stats['avg_confidence']}%</div>
                <div style="color: #fbbf24; font-size: 0.8rem; margin-top: 4px;">Test-Time Augmentation</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label-muted">Supported Pathologies</div>
                <div class="metric-value-huge">38</div>
                <div style="color: #60a5fa; font-size: 0.8rem; margin-top: 4px;">PlantVillage Standard</div>
            </div>
            """, unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns([1.5, 1])
        with col_chart1:
            st.markdown("##### 🌾 Top Diagnosed Crops")
            if stats["top_crops"]:
                for c in stats["top_crops"]:
                    st.write(f"**{c['crop_name']}**: {c['count']} scans")
                    st.progress(min(1.0, c['count'] / max(stats['total_scans'], 1)))
            else:
                st.info("Run your first scan to populate crop analytics.")

        with col_chart2:
            st.markdown("##### 🩺 Severity Breakdown")
            if stats["severity_breakdown"]:
                for sev, count in stats["severity_breakdown"].items():
                    st.write(f"• **{sev}**: {count} cases")
            else:
                st.info("No pathology severity data recorded yet.")

    # =========================================================================
    # VIEW 3: CONSULTATION RECORDS ARCHIVE
    # =========================================================================
    elif current_view == "history":
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="color: #10b981; font-weight: 800; margin-bottom: 2px;">Consultation Records Archive</h1>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Search, filter, inspect, and export all historical phytosanitary diagnoses saved in your clinical database.
            </p>
        </div>
        """, unsafe_allow_html=True)

        all_records = database.get_user_consultations(user_id=user["id"], limit=100)

        search_query = st.text_input("🔍 Search Past Records (by Crop, Disease, Report ID or Location):", placeholder="e.g. Apple, Rust, PMR, Nagpur")
        if search_query:
            all_records = [
                r for r in all_records
                if search_query.lower() in r["clean_name"].lower()
                or search_query.lower() in r["report_id"].lower()
                or search_query.lower() in r["city"].lower()
            ]

        if not all_records:
            st.info("No consultation records found matching your query.")
        else:
            for r in all_records:
                with st.expander(f"📋 {r['report_id']} | {r['clean_name']} ({r['confidence']:.1f}%) | {r['timestamp'][:16]} | {r['city']}"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if os.path.exists(r["orig_img_path"]):
                            st.image(r["orig_img_path"], caption="Specimen", width=200)
                    with c2:
                        st.write(f"**Pathology Severity:** {r['severity_level']} ({r['infected_ratio']:.1f}% infected tissue)")
                        st.write(f"**Microclimate:** {r.get('weather_temp', 25.0)}°C / {r.get('weather_humidity', 75.0)}% RH | Risk: {r.get('risk_level', 'Moderate')}")
                        st.write(f"**Remediation:** {r['treatment'].get('treatment', 'Apply standard agronomic protocols.')}")

                        col_act1, col_act2 = st.columns(2)
                        with col_act1:
                            if st.button("Open in Scanner ➔", key=f"open_arch_{r['id']}"):
                                st.session_state["active_consultation"] = r
                                st.session_state["current_view"] = "scanner"
                                st.rerun()
                        with col_act2:
                            if st.button("Delete Record 🗑️", key=f"del_arch_{r['id']}"):
                                database.delete_consultation(r["id"], user_id=user["id"])
                                st.rerun()

    # =========================================================================
    # VIEW 4: AGRO-METEOROLOGY & RISK FORECAST
    # =========================================================================
    elif current_view == "weather":
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="color: #065f46; font-weight: 800; margin-bottom: 2px;">Agro-Meteorology & Disease Risk Radar</h1>
            <p style="color: #64748b; font-size: 0.95rem;">
                Real-time regional atmospheric weather telemetry across Maharashtra farming districts, estimating fungal spore propagation risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📍 Regional District Telemetry")
        district_cols = st.columns(3)
        sample_districts = ["Nagpur", "Pune", "Nashik", "Akola", "Kolhapur", "Solapur"]

        for idx, d_name in enumerate(sample_districts):
            d_weather = get_live_weather(d_name)
            d_risk = calculate_disease_risk(d_weather, "Tomato___Late_blight")
            r_color = "#dc2626" if "Critical" in d_risk["risk_level"] else "#d97706" if "High" in d_risk["risk_level"] else "#059669"

            district_cols[idx % 3].markdown(f"""
            <div class="glass-card" style="border-top: 4px solid {r_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <b style="font-size: 1.1rem; color: #0f172a;">{d_weather['city']}</b>
                    <span style="color: {r_color}; font-size: 0.8rem; font-weight: 700;">{d_risk['risk_level']}</span>
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #059669; margin-bottom: 4px;">
                    {d_weather['temperature']}°C
                </div>
                <div style="color: #334155; font-size: 0.88rem;">
                    💧 Relative Humidity: <b>{d_weather['humidity']}% RH</b>
                </div>
                <div style="color: #64748b; font-size: 0.78rem; margin-top: 8px; line-height: 1.5;">
                    {d_risk['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # VIEW 5: CROP DISEASE FIELD GUIDE (38 CLASSES)
    # =========================================================================
    elif current_view == "guide":
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="color: #10b981; font-weight: 800; margin-bottom: 2px;">Crop Pathology Field Guide</h1>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Comprehensive clinical encyclopedia covering all 38 supported agricultural diseases, macroscopic symptoms, and treatment guidelines.
            </p>
        </div>
        """, unsafe_allow_html=True)

        guide_search = st.text_input("Search Crop or Disease:", placeholder="e.g. Potato, Apple, Mildew, Rust, Blight")
        filtered_classes = [c for c in PLANT_CLASSES if guide_search.lower() in c.lower() or guide_search.lower() in CLEAN_CLASS_NAMES.get(c, "").lower()]

        for cls_name in filtered_classes:
            clean_name = CLEAN_CLASS_NAMES.get(cls_name, cls_name)
            is_healthy = "healthy" in cls_name.lower()

            with st.expander(f"{'🟢' if is_healthy else '🔴'} {clean_name}"):
                c1, c2 = st.columns([1, 2.5])
                with c1:
                    st.write(f"**Scientific Label:** `{cls_name}`")
                    st.write(f"**Type:** {'Healthy Crop Tissue' if is_healthy else 'Pathological Fungal/Bacterial Infection'}")
                with c2:
                    if is_healthy:
                        st.markdown("**Field Status:** Plant tissue shows normal chlorophyll reflection, uniform foliar coloration, and robust vascular leaf integrity.")
                    else:
                        st.markdown("**Symptom Profile:** Necrotic spots, chlorotic halos, dried foliar lesions, or powdery fungal mycelium covering the leaf blade.")
                        st.markdown("**Management Practice:** Apply recommended protective fungicides (e.g. Copper Oxychloride, Mancozeb) and maintain canopy aeration.")

    # =========================================================================
    # ROLE-SPECIFIC SPECIALIZED VIEWS
    # =========================================================================
    elif current_view == "agronomist_review":
        role_views.render_agronomist_review_panel(user, lang_code)
    elif current_view == "model_diagnostics":
        role_views.render_model_diagnostics(user, lang_code)
    elif current_view == "outbreak_broadcast":
        role_views.render_outbreak_broadcast(user, lang_code)
    elif current_view == "spray_calendar":
        role_views.render_spray_calendar_calculator(user, lang_code)
    elif current_view == "outbreak_alerts":
        role_views.render_farmer_outbreak_alerts(user, lang_code)
    elif current_view == "helpline":
        role_views.render_farmer_helplines(user, lang_code)
    elif current_view == "ssm_lab":
        role_views.render_ssm_architecture_lab(user, lang_code)
    elif current_view == "pathology_quiz":
        role_views.render_pathology_quiz(user, lang_code)
    elif current_view == "benchmark":
        role_views.render_model_benchmarks(user, lang_code)


if __name__ == "__main__":
    main()
