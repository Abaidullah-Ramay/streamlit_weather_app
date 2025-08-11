# pages/notify.py
import streamlit as st
from datetime import datetime, timedelta
from groq import Groq
from twilio.rest import Client
import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import re
from config import (
    GROQ_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER, OPENWEATHER_API_KEY
)

# --- HELPER FUNCTIONS ---

def send_whatsapp_message(phone_number, message_body):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(body=message_body, from_=TWILIO_PHONE_NUMBER, to=f'whatsapp:{phone_number}')
        return message.sid
    except Exception as e:
        st.error(f"Error sending message: {e}")
        return None

def get_current_weather(city, api_key):
    # ... (Keep the original get_current_weather function code)
    pass
    
def get_weather_forecast(city, api_key):
    # ... (Keep the original get_weather_forecast function code)
    pass

# --- MAIN PAGE LOGIC ---

def show_page():
    """Renders the Notification scheduler page."""
    # Initialize scheduler in session state if not present
    if 'scheduler' not in st.session_state:
        scheduler = BackgroundScheduler()
        scheduler.start()
        st.session_state.scheduler = scheduler

    PAKISTAN_TZ = pytz.timezone('Asia/Karachi')

    # Sidebar for configuration
    with st.sidebar:
        st.subheader("API Configuration")
        models = ["llama3-8b", "qwen2.5-7b", "deepseek-r1-8b", "gemma2-9b"]
        selected_llm_model = st.selectbox("Select AI Model:", models, index=0)
        
        model_map = {
            "llama3-8b": "llama3-8b-8192", "qwen2.5-7b": "qwen/qwen3-32b",
            "deepseek-r1-8b": "deepseek-r1-distill-llama-70b", "gemma2-9b": "gemma2-9b-it"
        }
        selected_model = model_map[selected_llm_model]

        if st.button("🚀 Initialize AI Client", key="init_groq"):
            try:
                st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)
                st.session_state.selected_model = selected_model
                st.success("AI client initialized successfully!")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
    
    # Main form for scheduling
    with st.form("schedule_form"):
        phone_number = st.text_input("Recipient's WhatsApp Number", "+92")
        user_prompt = st.text_area("User Prompt", height=150)
        weather_city = st.text_input("City for Weather (Required)")
        
        col1, col2 = st.columns(2)
        with col1: selected_date = st.date_input("Select Date", min_value=datetime.today())
        with col2: selected_time = st.time_input("Select Time", step=timedelta(minutes=1))
        
        b_col1, b_col2 = st.columns(2)
        with b_col1: schedule_button = st.form_submit_button("✉️ Schedule Message", use_container_width=True)
        with b_col2: instant_button = st.form_submit_button("⚡ Instant Message", use_container_width=True)

    if schedule_button or instant_button:
        # --- Form validation and processing logic ---
        # ... (Keep the entire original logic from the `if schedule_button or instant_button:` block)
        pass
        
    # Display scheduled jobs in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scheduled Jobs")
    if 'scheduler' in st.session_state and st.session_state.scheduler.get_jobs():
        for job in st.session_state.scheduler.get_jobs():
            job_time = job.next_run_time.astimezone(PAKISTAN_TZ)
            st.sidebar.write(f"• {job.id} - {job_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
        if st.sidebar.button("Clear All Jobs"):
            st.session_state.scheduler.remove_all_jobs()
            st.sidebar.success("All jobs cleared!")
            st.rerun()
    else:
        st.sidebar.write("No scheduled jobs")