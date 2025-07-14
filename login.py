import streamlit as st

# Dummy credentials (you can replace with secure DB or API call)
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

# Initialize login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login form
def login_form():
    st.title("🔐 Login to the App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
   if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.stop()  # Better than rerun after login to avoid rerun issues
        else:
            st.error("Invalid username or password")


# Logout
def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# Main app
def main_app():
    st.title("📊 Welcome to the Dashboard")
    st.write(f"Hello, **{st.session_state.username}** 👋")
    logout_button()
    # Your main app logic here
    st.write("📈 Displaying your analytics and charts...")

# Routing
if not st.session_state.logged_in:
    login_form()
else:
    main_app()
