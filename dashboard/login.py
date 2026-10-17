
import os
import streamlit as st

st.set_page_config(
    page_title="CNC Digital Twin - Login",
    page_icon="🔐",
    layout="centered",
)

def get_credential(name):
    # Streamlit Community Cloud secrets
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    # Local environment variables
    return os.getenv(name.upper())

USERNAME = get_credential("dashboard_username")
PASSWORD = get_credential("dashboard_password")

st.title("🏭 CNC Digital Twin")
st.subheader("🔐 Login")
st.write("Please log in to access the CNC monitoring dashboard.")

if not USERNAME or not PASSWORD:
    st.error(
        "Login is not configured. Set dashboard_username and "
        "dashboard_password in Streamlit secrets or local environment variables."
    )
    st.stop()

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("🔓 Login", use_container_width=True)

if submitted:
    if username == USERNAME and password == PASSWORD:
        st.session_state["logged_in"] = True
        st.rerun()
    else:
        st.error("Invalid username or password.")
