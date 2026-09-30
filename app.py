import streamlit as st


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CNC Digital Twin",
    page_icon="🏭",
    layout="wide"
)


# ==========================================
# PAGES
# ==========================================

login_page = st.Page(
    "dashboard/login.py",
    title="Login",
    icon="🔐"
)

dashboard_page = st.Page(
    "dashboard/cloud_dashboard.py",
    title="CNC Dashboard",
    icon="🏭"
)


# ==========================================
# NAVIGATION
# ==========================================

if st.session_state.get("logged_in", False):

    pg = st.navigation(
        [dashboard_page]
    )

else:

    pg = st.navigation(
        [login_page]
    )


# ==========================================
# RUN PAGE
# ==========================================

pg.run()