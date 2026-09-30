import streamlit as st


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CNC Digital Twin - Login",
    page_icon="🔐",
    layout="centered"
)


# ==========================================
# LOGIN DETAILS
# ==========================================

USERNAME = "admin"
PASSWORD = "cnc123"


# ==========================================
# LOGIN PAGE
# ==========================================

st.title("🏭 CNC Digital Twin")

st.subheader("🔐 Login")

st.write(
    "Please login to access the CNC monitoring dashboard."
)


# ==========================================
# INPUTS
# ==========================================

username = st.text_input(
    "Username"
)

password = st.text_input(
    "Password",
    type="password"
)


# ==========================================
# LOGIN BUTTON
# ==========================================

if st.button(
    "🔓 Login",
    use_container_width=True
):

    if username == USERNAME and password == PASSWORD:

        st.session_state["logged_in"] = True

        st.success(
            "✅ Login successful!"
        )

        st.rerun()

    else:

        st.error(
            "❌ Invalid username or password."
        )


# ==========================================
# DEMO CREDENTIALS
# ==========================================

st.divider()

st.caption(
    "Demo credentials"
)

st.code(
    "Username: admin\n"
    "Password: cnc123"
)