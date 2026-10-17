
import os
import streamlit as st

st.set_page_config(
    page_title="CNC Digital Twin - Login",
    page_icon="🔐",
    layout="centered"
)

USERNAME = os.getenv("DASHBOARD_USERNAME")
PASSWORD = os.getenv("DASHBOARD_PASSWORD")

st.title("🏭 CNC Digital Twin")
st.subheader("🔐 Login")
st.write("Please log in to access the CNC monitoring dashboard.")

if not USERNAME or not PASSWORD:
    st.error("Login is not configured. Set the dashboard credentials in your environment.")
    st.stop()

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("🔓 Login", width="stretch"):
    if username == USERNAME and password == PASSWORD:
        st.session_state["logged_in"] = True
        st.rerun()
    else:
        st.error("Invalid username or password.")
