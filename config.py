# config.py
import streamlit as st

# --- API KEY MANAGEMENT ---
# Access secrets from Streamlit's secrets management.
# Ensure these are set in your Streamlit Cloud configuration.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
QDRANT_HOST = st.secrets.get("QDRANT_HOST", "")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = st.secrets.get("QDRANT_COLLECTION_NAME", "")
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = st.secrets.get("TWILIO_PHONE_NUMBER", "")

# --- PAGE SETUP ---
def set_page_config():
    """Sets the default Streamlit page configuration."""
    st.set_page_config(
        page_title="Chatbot for Weather Insights and Recommendations",
        layout="wide"
    )

# --- GLOBAL STYLES ---
def hide_default_streamlit_ui():
    """Injects CSS to hide default Streamlit UI elements."""
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
            div[data-testid="stToolbar"] {display: none;}
            header {visibility: hidden;}
            .block-container {padding-top: 1rem;}
        </style>
    """, unsafe_allow_html=True)

# --- CHATBOT CSS ---
CHATBOT_CSS = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

# --- PAGE-SPECIFIC BACKGROUNDS & STYLES ---
def apply_page_styling(page_name):
    """Applies a specific background and style based on the selected page."""
    background_styles = {
        "Description": """
            background-image: linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                              url("https://i.pinimg.com/736x/6a/00/92/6a009257f2f7b6a6fd62b9f63a849168.jpg");
        """,
        "Weather": """
            background-image: linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                              url("https://t4.ftcdn.net/jpg/02/66/38/15/360_F_266381525_alVrbw15u5EjhIpoqqa1eI5ghSf7hpz7.jpg");
        """,
        "Chatbot": """
            background-image: linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                              url("https://img.freepik.com/free-vector/background-monsoon-season_52683-115103.jpg?semt=ais_hybrid&w=740");
        """,
        "Results": """
            background-image: linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                              url("https://static.vecteezy.com/system/resources/thumbnails/020/069/751/small_2x/business-performance-monitoring-concept-businessman-using-smartphone-online-survey-filling-out-digital-form-checklist-blue-background-photo.jpg");
        """,
        "Notify": """
            background-image: linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                              url("https://t4.ftcdn.net/jpg/02/66/38/15/360_F_266381525_alVrbw15u5EjhIpoqqa1eI5ghSf7hpz7.jpg");
        """
    }

    style = f"""
        <style>
        .stApp {{
            {background_styles.get(page_name, "")}
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* General text color */
        .stApp, .stApp * {{
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7);
            font-weight: 600;
        }}
        /* Sidebar text should be clear */
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {{
            color: #111 !important;
            text-shadow: none !important;
            font-weight: 500 !important;
        }}
        /* Chat input text fix */
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] textarea::placeholder {{
            color: #111 !important;
            text-shadow: none !important;
            font-weight: 500 !important;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)