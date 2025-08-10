import streamlit as st
import datetime as dt
import os
from streamlit_option_menu import option_menu
import base64
import requests
#from dotenv import load_dotenv
import json
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import speech_recognition as sr
from groq import Groq # For the client API (tool calling)
from langchain_community.vectorstores import Qdrant  # Fixed import
from langchain_community.embeddings import OpenAIEmbeddings
import qdrant_client
from dotenv import load_dotenv
import os
from groq import Groq
from langchain_groq import ChatGroq # For LangChain integration
# Add these imports at the top of your file with the other imports
import os
import streamlit as st
import requests
import time
import json
from typing import Dict
from datetime import datetime
from collections import defaultdict
from groq import Groq
from langchain_community.vectorstores import Qdrant
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MultiQueryRetriever
import pyttsx3
import streamlit as st
from datetime import datetime, time, timedelta
from groq import Groq
from twilio.rest import Client
import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger

# Load environment variables at the very beginning
#load_dotenv()
OPENAI_API_KEY_FROM_ENV = st.secrets["OPENAI_API_KEY"]
OPENWEATHER_API_KEY_FROM_ENV = st.secrets["OPENWEATHERMAP_API_KEY"]
GROQ_API_KEY_FROM_ENV = st.secrets["GROQ_API_KEY"]
# **Set page config as the first Streamlit command**
st.set_page_config(
    page_title="Chatbot for Weather Insights and Recommendations", layout="wide")

# Hide default Streamlit UI elements
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

selected = option_menu(None, ["Description", "Weather", "Chatbot", "Results", "Notify"],
    icons=["house", "cloud", "chat", "megaphone","whatsapp"],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "#2D3748",
            "width": "100%",
            "flex-wrap": "nowrap"
        },
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#4A5568",
            "color": "#FFFFFF",
            "white-space": "nowrap",
            "min-width": "fit-content",
            "flex-shrink": "0",
            "font-family": "sans-serif",
            "border-bottom": "2px solid transparent",
            "transition": "border-bottom 0.3s ease",
            ":hover": {
                "border-bottom": "2px solid #63B3ED"
            }
        },
        "nav-link-selected": {
            "background-color": "#4299E1",
            "color": "#FFFFFF",
            "flex-shrink": "0"
        },
    }
)


##### css style ######
css = '''
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
'''



############ Weather function ###############
def get_weather_data(city, weather_api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + weather_api_key + "&q=" + city
    response = requests.get(complete_url)
    return response.json()

def get_weekly_forecast(weather_api_key, lat, lon):
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = f"{base_url}lat={lat}&lon={lon}&appid={weather_api_key}"
    response = requests.get(complete_url)
    return response.json()

def plot_weather_data(list_days, list_temp, list_humidity, list_pressure, list_wind_speed, list_cloudiness):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Weather Parameters Over Days", fontsize=16)

    axs[0, 0].plot(list_days, list_temp, marker='o', color='red')
    axs[0, 0].set_title("Temperature (°C)")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].tick_params(axis='x', rotation=45)

    axs[0, 1].plot(list_days, list_humidity, marker='o', color='blue')
    axs[0, 1].set_title("Humidity (%)")
    axs[0, 1].set_ylabel("Humidity (%)")
    axs[0, 1].tick_params(axis='x', rotation=45)

    axs[1, 0].plot(list_days, list_pressure, marker='o', color='green')
    axs[1, 0].set_title("Pressure (hPa)")
    axs[1, 0].set_ylabel("Pressure (hPa)")
    axs[1, 0].tick_params(axis='x', rotation=45)

    axs[1, 1].plot(list_days, list_wind_speed, marker='o', color='purple')
    axs[1, 1].set_title("Wind Speed (m/s)")
    axs[1, 1].set_ylabel("Wind Speed (m/s)")
    axs[1, 1].tick_params(axis='x', rotation=45)

    axs[2, 0].plot(list_days, list_cloudiness, marker='o', color='grey')
    axs[2, 0].set_title("Cloudiness (%)")
    axs[2, 0].set_ylabel("Cloudiness (%)")
    axs[2, 0].tick_params(axis='x', rotation=45)

    fig.delaxes(axs[2, 1])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def display_weekly_forecast(data):
    try:
        st.write("=" * 88)
        st.write("### Weekly Weather Forecast")
        displayed_dates = set()
        
        # Create column headers using proper formatting
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("Day")
        with c2:
            st.subheader("Description")
        with c3:
            st.subheader("Min Temp (°C)")
        with c4:
            st.subheader("Max Temp (°C)")

        list_days = []
        list_temp = []
        list_humidity = []
        list_pressure = []
        list_wind_speed = []
        list_cloudiness = []

        timezone_offset = data['city']['timezone']
        local_tz = dt.timezone(dt.timedelta(seconds=timezone_offset))

        for day in data['list']:
            utc_dt = dt.datetime.fromtimestamp(day['dt'], tz=dt.timezone.utc)
            local_dt = utc_dt.astimezone(local_tz)
            date = local_dt.strftime('%A %B %d')

            if date not in displayed_dates:
                displayed_dates.add(date)
                list_days.append(date)

                min_temp = day['main']['temp_min'] - 273.15
                max_temp = day['main']['temp_max'] - 273.15
                description = day['weather'][0]['description']

                list_temp.append(numpy.round(day['main']['temp'] - 273.15, 2))
                list_pressure.append(day['main']['pressure'])
                list_humidity.append(day['main']['humidity'])
                list_wind_speed.append(day['wind']['speed'])
                list_cloudiness.append(day['clouds']['all'])

                with c1:
                    st.write(f"{date}")
                with c2:
                    st.write(f"{description.capitalize()}")
                with c3:
                    st.write(f"{min_temp:.1f}°C")
                with c4:
                    st.write(f"{max_temp:.1f}°C")

        st.title("Weather Parameters Over Days")
        fig = plot_weather_data(list_days, list_temp, list_humidity,
                                list_pressure, list_wind_speed, list_cloudiness)
        st.pyplot(fig)

    except Exception as e:
        st.error("Error in displaying weekly forecast: " + str(e))

###############  End of weather functions  #################


def first_page():
    def image_to_base64(img_path):
        if not os.path.exists(img_path):
            st.error(f"Image not found: {img_path}. Please ensure it's in the correct location.")
            return "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        with open(img_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()

    bg_img_path = "logos/uni_background.jpg"
    uni_logo_path = "logos/Nust_new_logo.png"
    dept_logo_path = "logos/Sines_new_logo.png"

    bg_base64 = image_to_base64(bg_img_path)
    uni_logo_base64 = image_to_base64(uni_logo_path)
    dept_logo_base64 = image_to_base64(dept_logo_path)

    st.markdown(f"""
        <style>
            .header-container {{
                background-image: url("data:image/jpeg;base64,{bg_base64}");
                background-size: cover;
                background-position: center;
                padding: 30px 10px;
                border-radius: 12px;
            }}
            .header-overlay {{
                background-color: rgba(255, 255, 255, 0.6);
                padding: 20px;
                border-radius: 10px;
            }}
            .logo-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .logo-row img {{
                height: 90px;
            }}
            .header-title h1 {{
                text-align: center;
                color: #2E86C1; /* This is the title inside the overlay, can be kept or changed */
            }}
        </style>

        <div class="header-container">
            <div class="header-overlay">
                <div class="logo-row">
                    <img src="data:image/png;base64,{uni_logo_base64}" alt="University Logo">
                    <img src="data:image/png;base64,{dept_logo_base64}" alt="Department Logo">
                </div>
                <div class="header-title">
                    <h1>Chatbot for Weather Insights and Recommendations</h1>
                </div>
            </div>
        </div>
        <hr style='border:2px solid #2E86C1'>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='color:white'>

    ### Institutional Details
    - **University:** National University of Science and Technology (NUST), Pakistan
    - **Student Name:** Abaidullah Ramay
    - **Supervisor:** Dr. Muhammad Tariq Saeed
    - **Department:** School of Interdisciplinary Engineering and Sciences (SINES)
    - **Project Title:** Chatbot for Weather Insights and Recommendations
    - **Project Duration:** 6 Months

    ### Project Overview
    This project explores the use of Large Language Model (LLM) as a chatbot, equipped with Retrieval Augmented Generation (RAG) pipeline. Data used is vast set of websites scraped from internet on Weather and Climatic studies. This project covers complete pipeline of RAG starting from **loading scraped data in vector database** to **retrieval** of relevant chunks to **generation** by LLM.
    The chatbot is designed to answer questions related to weather and climate, providing insights and recommendations based on the retrieved data. 
    ### Features
    - **Weather Forecasting:** Provides current weather updates and forecasts for any city.
    - **Chatbot Functionality:** Engages users in conversation, answering questions about weather and climate.
    - **RAG Pipeline:** Utilizes a Retrieval Augmented Generation pipeline to enhance the chatbot's responses with relevant data.
    - **Data Visualization:** Displays weather data in a user-friendly format, including graphs and charts.
    ### Technologies Used
    - **Streamlit:** For building the web application interface.
    - **Ollama:** For accessing the LLM capabilities.
    - **OpenWeather API:** For fetching real-time weather data.
    - **Chromadb:** For vector storage and retrieval of weather-related data.
    ### Future Work
    - **Enhancing Data Sources:** Integrating more diverse data sources for improved accuracy and insights.
    - **Improving User Experience:** Adding more interactive features and improving the chatbot's conversational abilities.
    - **Expanding Functionality:** Including more advanced weather analytics and predictions.
    </div>
    """, unsafe_allow_html=True)

def second_page():
    st.sidebar.title("Weather forecasting with LLM") # Sidebar title color is usually handled by theme
    city = st.sidebar.text_input("Enter city name", "Islamabad")

    weather_api_key = OPENWEATHER_API_KEY_FROM_ENV

    if not weather_api_key:
        st.sidebar.error("OpenWeather API key not found. Please set OPENWEATHER_API_KEY in your .env file.")
        st.markdown("<p style='color: red;'>Weather forecast functionality is unavailable due to missing API key.</p>", unsafe_allow_html=True)
        return

    submit = st.sidebar.button("Get Weather")

    if submit:
        try:
            weather_main_data = get_weather_data(city, weather_api_key)

            if weather_main_data and weather_main_data.get("cod") != "404" and weather_main_data.get("cod") != 404:
                st.sidebar.write(f"{city} Latitude", weather_main_data["coord"]["lat"])
                st.sidebar.write(f"{city} Longitude", weather_main_data["coord"]["lon"])
                st.sidebar.write("Sunrise Time", dt.datetime.utcfromtimestamp(weather_main_data["sys"]["sunrise"] + weather_main_data["timezone"]))
                st.sidebar.write("Sunset Time", dt.datetime.utcfromtimestamp(weather_main_data["sys"]["sunset"] + weather_main_data["timezone"]))
                
                current_datetime = dt.datetime.now()
                st.sidebar.write(current_datetime.date().strftime("%d-%m-%Y"))
                st.sidebar.write(current_datetime.time().strftime("%H:%M:%S"))

                # CHANGED: st.title to st.markdown for white color
                st.markdown(f"<h1 style='color: white; text-align: center;'>Weather updates for {city} is:</h1>", unsafe_allow_html=True)
                
                with st.spinner("Fetching weather data..."):
                    col1, col2 = st.columns(2)
                    # Metric values might need their own styling if they don't inherit or if the background is too light
                    # For now, assuming default metric styling is acceptable or will inherit if possible.
                    # If metric text is also dull, they'd need custom HTML like:
                    # st.markdown(f"<h3 style='color: white;'>Temperature 🌡</h3><p style='color: white; font-size: 1.5em;'>{weather_main_data['main']['temp'] - 273.15:.2f} °C</p>", unsafe_allow_html=True)
                    # But st.metric is convenient. Let's see how it looks first.
                    with col1:
                        st.metric("Temperature 🌡",
                                  f"{weather_main_data['main']['temp'] - 273.15:.2f} °C")
                        st.metric("Humidity 💧",
                                  f"{weather_main_data['main']['humidity']}%")
                    with col2:
                        st.metric(
                            "Pressure", f"{weather_main_data['main']['pressure']} hPa")
                        st.metric("Wind Speed 🍃 🍂",
                                  f"{weather_main_data['wind']['speed']} m/s")

                    lat = weather_main_data['coord']['lat']
                    lon = weather_main_data['coord']['lon']

                    forecast_data = get_weekly_forecast(weather_api_key, lat, lon)
                    if forecast_data and "list" in forecast_data:
                        # Assuming display_weekly_forecast handles its own text colors
                        # or if it uses st.write, st.markdown, etc., those might need adjustment too.
                        display_weekly_forecast(forecast_data)
                    else:
                        st.markdown("<p style='color: red;'>Error fetching weekly forecast data or data format incorrect.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: red;'>City '{city}' not found or an error occurred: {weather_main_data.get('message', 'Unknown error')}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<p style='color: red;'>An error occurred while fetching weather data: {e}</p>", unsafe_allow_html=True)
            # st.error(traceback.format_exc()) # Keep for debugging if needed

def third_page():
    import os
    import tempfile
    import base64
    import json
    import openai
    import streamlit as st

    st.write(css, unsafe_allow_html=True)  # keep your css variable

    # --- Secrets / OpenAI ---
    groq_api_key_to_use = st.secrets.get("GROQ_API_KEY")
    qdrant_host = st.secrets.get("QDRANT_HOST")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    qdrant_collection = st.secrets.get("QDRANT_COLLECTION_NAME")
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    # --- helper: safe summary for audio_info (so we can render in sidebar) ---
    def summarize_audio_info(ai):
        if ai is None:
            return "None"
        summary = {}
        for k, v in ai.items():
            try:
                if isinstance(v, (bytes, bytearray)):
                    summary[k] = {"type": "bytes", "len": len(v)}
                elif isinstance(v, str):
                    summary[k] = {"type": "str", "len": len(v), "preview": v[:200] + ("..." if len(v) > 200 else "")}
                elif isinstance(v, (int, float, bool)):
                    summary[k] = {"type": type(v).__name__, "value": v}
                else:
                    # fallback repr (safe)
                    summary[k] = {"type": type(v).__name__, "repr": repr(v)[:200]}
            except Exception as e:
                summary[k] = {"type": type(v).__name__, "repr_error": str(e)}
        return summary

    # --- Session state init ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input_value" not in st.session_state:
        st.session_state.user_input_value = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = None
    if "input_reset_key" not in st.session_state:
        st.session_state.input_reset_key = 0
    if "transcription_success" not in st.session_state:
        st.session_state.transcription_success = False

    # --- title ---
    st.markdown("<h2 style='color: white; text-align: center;'> 🌦️ Smart Weather Assistant</h2>", unsafe_allow_html=True)

    # --- Sidebar: config + STT + debug output ---
    with st.sidebar:
        st.subheader("Configuration")
        models = ["llama3-8b", "qwen2.5-7b", "deepseek-r1-8b", "gemma2-9b"]
        selected_llm_model = st.selectbox("Select Model:", models, key="model_select")
        # map selection
        if selected_llm_model == "llama3-8b":
            selected_model = "llama3-8b-8192"
        elif selected_llm_model == "qwen2.5-7b":
            selected_model = "qwen/qwen3-32b"
        elif selected_llm_model == "deepseek-r1-8b":
            selected_model = "deepseek-r1-distill-llama-70b"
        else:
            selected_model = "gemma2-9b-it"

        if st.button("🚀 Initialize Chat", key="init_chat"):
            with st.spinner("Initializing AI components..."):
                try:
                    from langchain_community.embeddings import OpenAIEmbeddings
                    from langchain_qdrant import Qdrant
                    import qdrant_client

                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    qdrant_client_instance = qdrant_client.QdrantClient(qdrant_host, api_key=qdrant_api_key)
                    st.session_state.vector_store = Qdrant(
                        client=qdrant_client_instance,
                        collection_name=qdrant_collection,
                        embeddings=embeddings
                    )

                    from groq import Groq
                    st.session_state.groq_client = Groq(api_key=groq_api_key_to_use)

                    st.session_state.conversation = True
                    st.sidebar.success("Chat initialized successfully!")
                except Exception as e:
                    st.sidebar.error(f"Initialization failed: {e}")

        st.markdown("---")
        st.subheader("Voice Input (STT)")

        # try to import the recorder component
        try:
            from streamlit_mic_recorder import mic_recorder  # component
        except Exception as e:
            st.error("streamlit-mic-recorder NOT installed or failed to import. Please add to requirements.txt.")
            st.stop()

        # --- recorder call ---
        audio_info = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key="mic_recorder",
            format="wav"
        )

        # --- Debug: raw audio_info summary ---
        st.sidebar.markdown("### Raw mic_recorder output (DEBUG)")
        try:
            st.sidebar.json(summarize_audio_info(audio_info))
        except Exception:
            st.sidebar.write("Could not render audio_info summary (non-serializable).")

        # --- Extract bytes robustly from many possible shapes ---
        def get_audio_bytes_from_audio_info(ai):
            if not ai:
                return None, "audio_info is empty/None"

            # Common keys returned by different forks: 'bytes', 'blob', 'base64', 'dataURL', 'audio'
            # Try them in order of likelihood.
            # 1) bytes / bytearray
            if "bytes" in ai and isinstance(ai["bytes"], (bytes, bytearray)):
                return bytes(ai["bytes"]), None

            # 2) base64 string (raw)
            for k in ("base64", "base_64", "b64"):
                if k in ai and isinstance(ai[k], str):
                    try:
                        return base64.b64decode(ai[k]), None
                    except Exception as e:
                        return None, f"base64 decode error for key {k}: {e}"

            # 3) blob / dataURL like "data:audio/wav;base64,AAAA..."
            if "blob" in ai and isinstance(ai["blob"], str):
                s = ai["blob"]
                if s.startswith("data:"):
                    try:
                        header, b64 = s.split(",", 1)
                        return base64.b64decode(b64), None
                    except Exception as e:
                        return None, f"dataURL decode error: {e}"
                else:
                    # might be plain base64 string
                    try:
                        return base64.b64decode(s), None
                    except Exception as e:
                        return None, f"unknown blob format: {e}"

            # 4) some components put audio under 'audio' or 'recording'
            for k in ("audio", "recording"):
                if k in ai:
                    v = ai[k]
                    if isinstance(v, (bytes, bytearray)):
                        return bytes(v), None
                    if isinstance(v, str):
                        try:
                            return base64.b64decode(v), None
                        except Exception as e:
                            return None, f"decoding {k} string failed: {e}"

            return None, "No recognized audio bytes in audio_info"

        audio_bytes, audio_err = get_audio_bytes_from_audio_info(audio_info)

        # Show whether we found bytes
        if audio_bytes:
            st.sidebar.success(f"Found audio bytes — {len(audio_bytes)} bytes")
            # Playback in sidebar so you can confirm
            try:
                st.sidebar.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.sidebar.write(f"Could not play audio in sidebar: {e}")
        else:
            st.sidebar.warning(f"No audio bytes extracted: {audio_err}")

        # --- Provide an uploader test so you can check transcription independent of recorder ---
        st.markdown("#### Test transcription with upload (helps isolate issues)")
        test_file = st.file_uploader(
            "Upload a WAV/MP3 to test transcription", 
            type=["wav", "mp3", "m4a", "ogg"],
            key="test_file_uploader"
        )

        # --- If we have bytes, save to temp file and call OpenAI (robust) ---
        def robust_transcribe(tmp_path):
            """
            Try multiple OpenAI client call patterns to handle different SDK versions.
            Returns (text_or_none, error_or_none)
            """
            last_exc = None
            # try 1: openai.Audio.transcribe(model, file)  (common example)
            try:
                with open(tmp_path, "rb") as f:
                    res = openai.Audio.transcribe("whisper-1", f)
                # res may be a dict or object
                if isinstance(res, dict):
                    text = res.get("text")
                else:
                    text = getattr(res, "text", None) or (res.get("text") if isinstance(res, dict) else None)
                if text:
                    return text, None
            except Exception as e:
                last_exc = e

            # try 2: openai.Audio.transcriptions.create(...)
            try:
                with open(tmp_path, "rb") as f:
                    # some SDKs expose this
                    res = openai.Audio.transcriptions.create(model="whisper-1", file=f)
                if isinstance(res, dict):
                    text = res.get("text")
                else:
                    text = getattr(res, "text", None)
                if text:
                    return text, None
            except Exception as e:
                last_exc = e

            # try 3: openai.audio.transcriptions.create(...) (lowercase package var)
            try:
                with open(tmp_path, "rb") as f:
                    res = openai.audio.transcriptions.create(model="whisper-1", file=f)
                if isinstance(res, dict):
                    text = res.get("text")
                else:
                    text = getattr(res, "text", None)
                if text:
                    return text, None
            except Exception as e:
                last_exc = e

            # try 4: older pattern (just in case)
            try:
                with open(tmp_path, "rb") as f:
                    res = openai.Transcription.create(model="whisper-1", file=f)
                if isinstance(res, dict):
                    text = res.get("text")
                else:
                    text = getattr(res, "text", None)
                if text:
                    return text, None
            except Exception as e:
                last_exc = e

            return None, f"All transcription attempts failed. Last exception: {repr(last_exc)}"

        # If we have audio bytes available OR test_file was uploaded -> attempt transcription
        do_transcribe = False
        tmp_audio_path = None

        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                tf.write(audio_bytes)
                tmp_audio_path = tf.name
            do_transcribe = True
        elif test_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(test_file.name)[1]) as tf:
                tf.write(test_file.read())
                tmp_audio_path = tf.name
            do_transcribe = True

        if do_transcribe and tmp_audio_path:
            st.sidebar.info(f"Saved temp audio to {tmp_audio_path} — attempting transcription...")
            text, err = robust_transcribe(tmp_audio_path)

            if text:
                st.sidebar.success("Transcription successful!")
                st.sidebar.text_area("Transcript (DEBUG)", value=text, height=150)
                
                # Set as user input
                st.session_state.user_input_value = text
                st.session_state.input_reset_key += 1
                st.session_state.transcription_success = True
                
                # Reset recorder states to prevent reprocessing
                if audio_bytes:
                    st.session_state.mic_recorder = None
                if test_file is not None:
                    st.session_state.test_file_uploader = None
                
                # Cleanup temp file
                try:
                    os.remove(tmp_audio_path)
                except Exception:
                    pass
            else:
                st.sidebar.error(f"Transcription failed: {err}")
                # cleanup file
                try:
                    os.remove(tmp_audio_path)
                except Exception:
                    pass

    # --- Main area: show chat history ---
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if st.button(f"🔊", key=f"tts_button_{i}", help="Play assistant's response"):
                    # reuse your text to speech
                    clean_text = message['content'].split("**Document Sources:**")[0].strip()
                    try:
                        from gtts import gTTS
                        from io import BytesIO
                        tts = gTTS(text=clean_text, lang='en')
                        fp = BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        audio_b64 = base64.b64encode(fp.read()).decode()
                        st.components.v1.html(f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}"></audio>', height=0)
                    except Exception as e:
                        st.error(f"TTS error: {e}")

    # --- Handle input with transcription success check ---
    prompt = None
    if st.session_state.transcription_success:
        # Clear success flag to prevent reprocessing
        st.session_state.transcription_success = False
        # Use transcribed text as prompt
        prompt = st.session_state.user_input_value
        st.session_state.user_input_value = None  # Reset after use
    else:
        # Regular input handling
        input_placeholder = st.empty()
        chat_key = f"chat_input_{st.session_state.input_reset_key}"
        
        try:
            # try to prefill chat_input (works on newer Streamlit)
            prompt = input_placeholder.chat_input(
                "Ask about documents...",
                value=st.session_state.user_input_value or "",
                key=chat_key
            )
        except TypeError:
            # older Streamlit where chat_input doesn't accept value
            st.sidebar.info("Streamlit version doesn't support chat_input(value=...). Using text_input fallback.")
            prompt = input_placeholder.text_input(
                "Ask about documents...",
                value=st.session_state.user_input_value or "",
                key=f"text_input_fallback_{st.session_state.input_reset_key}"
            )
        except Exception as e:
            st.error(f"Unexpected error rendering input widget: {e}")
            prompt = input_placeholder.text_input(
                "Ask about documents...",
                value=st.session_state.user_input_value or "",
                key=f"text_input_final_fallback_{st.session_state.input_reset_key}"
            )

        # Clear the stored prefill after we read it (so next time it's not reused)
        if prompt:
            st.session_state.user_input_value = None

    # --- Process prompt ---
    if prompt:
        if st.session_state.conversation and st.session_state.vector_store and st.session_state.groq_client:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Assistant is thinking..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant and professional writer. "
                                "Always use Markdown formatting with headings, bullet points, and short paragraphs. "
                                "Highlight key terms in bold. Use the following context:\n"
                                f"{context}\n"
                                "If the context is not enough, use your general knowledge."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ]

                    response = st.session_state.groq_client.chat.completions.create(
                        messages=messages,
                        model=selected_model,
                    )

                    final_answer = response.choices[0].message.content
                    if docs:
                        sources = "\n\n**Document Sources:**\n" + "\n".join(
                            [f"- {doc.metadata.get('source', 'Unknown source')}" for doc in docs]
                        )
                        final_answer += sources

                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                except Exception as e:
                    error_message = f"Sorry, an unexpected error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            st.warning("Chat not initialized. Please select a model and click 'Initialize Chat' in the sidebar.")

if selected == "Description":
    st.markdown("""
        <style>
        .stApp {
            background-image: 
                linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                url("https://i.pinimg.com/736x/6a/00/92/6a009257f2f7b6a6fd62b9f63a849168.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp, .stApp * {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7);
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    first_page()
elif selected == "Weather":
    st.markdown("""
        <style>
        .stApp {
            background-image: 
                linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                url("https://t4.ftcdn.net/jpg/02/66/38/15/360_F_266381525_alVrbw15u5EjhIpoqqa1eI5ghSf7hpz7.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main-vibrant, .main-vibrant * {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7);
            font-weight: 600;
        }
        /* Make sidebar text black and remove text-shadow for clarity */
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
            color: #111 !important;
            text-shadow: none !important;
            font-weight: 500 !important;
        }
        /* Make st.metric values vibrant */
        div[data-testid="stMetricValue"] {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7) !important;
            font-weight: 700 !important;
        }
        /* Make st.metric label vibrant */
        div[data-testid="stMetricLabel"] {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7) !important;
            font-weight: 600 !important;
        }
        /* Make st.subheader and st.header vibrant */
        h1, h2, h3, h4, h5, h6, .stMarkdown {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7) !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-vibrant">', unsafe_allow_html=True)
    second_page()
    st.markdown('</div>', unsafe_allow_html=True)


elif selected == "Chatbot":
    st.markdown("""
        <style>
        .stApp {
            background-image: 
                linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                url("https://img.freepik.com/free-vector/background-monsoon-season_52683-115103.jpg?semt=ais_hybrid&w=740");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp, .stApp * {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7);
            font-weight: 600;
        }
        /* Make sidebar text black and remove text-shadow for clarity */
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
            color: #111 !important;
            text-shadow: none !important;
            font-weight: 500 !important;
        }
        /* Make chat input text and placeholder black */
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #111 !important;
            text-shadow: none !important;
            font-weight: 500 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    third_page()
elif selected == "Results":
    st.markdown("""
        <style>
        .stApp {
            background-image: 
                linear-gradient(rgba(20,20,20,0.65), rgba(20,20,20,0.65)),
                url("https://static.vecteezy.com/system/resources/thumbnails/020/069/751/small_2x/business-performance-monitoring-concept-businessman-using-smartphone-online-survey-filling-out-digital-form-checklist-blue-background-photo.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp, .stApp * {
            color: #fff !important;
            text-shadow: 0 1px 4px rgba(0,0,0,0.7);
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    # General style for the page: default text to white
    # We will use st.markdown with inline styles for text elements we want to be white.
    # Avoid a global CSS override that might break dataframes/plots.

    # Helper function to display text in white
    def white_text(text_content, level="p", text_align="left"):
        if level.startswith("h"):
            st.markdown(f"<{level} style='color: white; text-align: {text_align};'>{text_content}</{level}>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color: white; text-align: {text_align};'>{text_content}</p>", unsafe_allow_html=True)

    # --- RAGAS Evaluation Section ---
    white_text("RAGAS Evaluation", level="h1", text_align="center") # Centered title
    white_text("Synthetic questions and models responses", level="h2")
    white_text("We first generate synthetic testset for over data using RAGAS library. We use GPT-4o-mini model to generate testset comprising of 50 questions and referance answer. We then pass these questions to four models namely DeepSeek-R1, Qwen2.5, Gemma2, Llama3, to generate responses. Given below are 10 samples out of our 50 set:")

    # Define questions and answers
    questions = [
        "Q1. How does the climate of Gilgit-Baltistan differ from other regions in Pakistan?",
        "Q2. What are the climatic challenges faced by Balochistan, particularly regarding drought?",
        "Q3. How has climate change affected Karachi in recent years?"
    ]

    answers = {
        "Q1. How does the climate of Gilgit-Baltistan differ from other regions in Pakistan?": {
            "Reference Answer": "The climate of Gilgit-Baltistan is characterized as a continental type, which is distinct from other regions in Pakistan. This area experiences extreme variations in temperature due to its high altitudes, with cold, snow-covered northern mountains influencing its weather patterns.",
            "DeepSeek Answer": "Climate change is caused by human activities like burning fossil fuels, leading to global warming and changes in weather patterns.",
            "Qwen Answer": "It refers to shifts in global or regional climate patterns, often linked to increased levels of atmospheric carbon dioxide.",
            "Gemma Answer": "A long-term change in Earth's climate, including rising temperatures and extreme weather events, primarily due to greenhouse gas emissions.",
            "LLaMA Answer": "Climate change describes sustained changes in climate patterns over decades, affecting ecosystems and human societies."
        },
        "Q2. What are the climatic challenges faced by Balochistan, particularly regarding drought?": {
            "Reference Answer": "Balochistan faces significant climatic challenges, particularly drought, which has become a frequent phenomenon in the region. The drought from 1998 to 2002 is considered the worst in 50 years, severely stretching the coping abilities of existing systems and leading to unmet water needs for 40 percent of the country's requirements. This drought is noted as one of the most significant factors affecting the growth performance of the region.",
            "DeepSeek Answer": "AI enhances meteorology by utilizing deep learning models to analyze historical and real-time data for better forecasting.",
            "Qwen Answer": "It leverages neural networks to interpret satellite images, sensor data, and atmospheric conditions for precise weather predictions.",
            "Gemma Answer": "AI-driven models process meteorological data efficiently, allowing for high-resolution climate forecasting.",
            "LLaMA Answer": "By using data-driven methods, AI refines predictive models, increasing accuracy in climate simulations and weather forecasting."
        },
        "Q3. How has climate change affected Karachi in recent years?": {
            "Reference Answer": "Climate change has significantly impacted Karachi, contributing to increased heat, drought, and extreme weather events. The city, like the rest of Pakistan, is highly vulnerable to these changes, which have been linked to severe natural disasters, including floods. The 2022 floods, for instance, were exacerbated by climate change and had devastating effects on the population and infrastructure.",
            "DeepSeek Answer": "It results in melting glaciers, increased droughts, hurricanes, and shifts in weather patterns.",
            "Qwen Answer": "Global warming impacts include more frequent wildfires, habitat destruction, and intensified heatwaves.",
            "Gemma Answer": "Severe consequences include ocean acidification, food shortages, and forced human migration due to climate-related disasters.",
            "LLaMA Answer": "It contributes to ecosystem imbalances, rising global temperatures, and changes in precipitation patterns affecting water supply."
        }
    }

    scores_bank = {
        "Q1. Scores": {
            "DeepSeek Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.600, "Answer Relevancy": 0.964},
            "Qwen Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.300, "Answer Relevancy": 1.000},
            "Gemma Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.875, "Answer Relevancy": 1.000},
            "LLaMA Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.714, "Answer Relevancy": 1.000}
        },
        "Q2. Scores": {
            "DeepSeek Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.313, "Answer Relevancy": 0.873},
            "Qwen Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.500, "Answer Relevancy": 0.957},
            "Gemma Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.333, "Answer Relevancy": 0.906},
            "LLaMA Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.375, "Answer Relevancy": 0.906}
        },
        "Q3. Scores": {
            "DeepSeek Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.000, "Answer Relevancy": 0.996},
            "Qwen Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.333, "Answer Relevancy": 0.966},
            "Gemma Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.556, "Answer Relevancy": 0.945},
            "LLaMA Scores": {"Context Precision": 1.000, "Context Recall": 1.000, "Faithfulness": 0.167, "Answer Relevancy": 1.000}
        }
    }

    st.markdown("""
    <style>
        div[data-testid="stExpander"] summary {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    for question_text in questions:
        with st.expander(question_text):
            q_number_key = question_text.split('.')[0] + "." 
            score_key = f"{q_number_key} Scores" 

            for model_type, response_text in answers.get(question_text, {}).items():
                st.markdown(f"<p style='color: white;'><strong>{model_type}:</strong> {response_text}</p>", unsafe_allow_html=True)

                if model_type == "Reference Answer":
                    continue

                model_name_for_score = model_type.replace(" Answer", "") 
                model_score_key = f"{model_name_for_score} Scores"
                scores_data = scores_bank.get(score_key, {}).get(model_score_key, {})

                if scores_data:
                    df_scores = pd.DataFrame([scores_data])
                    st.dataframe(df_scores) 
                else:
                    st.markdown("<p style='color: white;'>No scores available for this model.</p>", unsafe_allow_html=True)

    st.divider()

    white_text("Generate Average Scores", level="h2")
    white_text("We generate Average of our scores computed for 50 set. Given below is average model scores:")

    data = {
        "Models": ["Deep Seek-r1:8b", "Qwen2.5:7b", "Llama3:8b", "Gemma:7b"],
        "Context Precision": [1.0000, 1.0000, 1.0000, 1.0000],
        "Context Recall": [0.9585, 0.9644, 0.9546, 0.9644],
        "Faithfulness": [0.5723, 0.6650, 0.5926, 0.6504],
        "Answer Relevancy": [0.7988, 0.7281, 0.6335, 0.7049]
    }
    df_avg_scores = pd.DataFrame(data)
    st.dataframe(df_avg_scores, hide_index=True)

    st.markdown("<p style='color: white; font-size: 0.9em; text-align: center;'><i>Average Context Precision, Average Context Recall, Average Faithfulness, Average Answer Relevancy</i></p>", unsafe_allow_html=True)

    st.divider()

    # --- Plot Section ---
    white_text("Generating Plot", level="h2")
    white_text("We generate side by side bar plot for analysis of average scores among our four models.")

    # Store default rcParams to restore later
    default_rcParams = plt.rcParams.copy()

    try:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Set base rcParams (some might still be useful for elements not explicitly set)
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white' # Fallback if direct set_xlabel/ylabel color is removed
        plt.rcParams['xtick.color'] = 'white'    # For tick marks primarily now
        plt.rcParams['ytick.color'] = 'white'    # For tick marks primarily now
        plt.rcParams['axes.titlecolor'] = 'white'  # Fallback if direct set_title color is removed

        n_models = len(df_avg_scores)
        n_metrics = 4
        bar_width = 0.2
        index = np.arange(n_models)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        metrics_to_plot = ['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Relevancy']

        for i, metric in enumerate(metrics_to_plot):
            ax.bar(index + i * bar_width, df_avg_scores[metric], bar_width, label=metric, color=colors[i])

        # Explicitly set colors for title, axis labels, and tick labels
        ax.set_xlabel('Models', fontsize=12, color='white')
        ax.set_ylabel('Scores', fontsize=12, color='white')
        ax.set_title('Model Performance Comparison', fontsize=14, color='white')

        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        # Set x-tick labels and their color
        ax.set_xticklabels(df_avg_scores['Models'], rotation=45, ha="right", fontsize=10, color='white')

        # Set y-tick label color
        ax.tick_params(axis='y', labelcolor='white', labelsize=10)
        
        # Ensure tick marks themselves are also white (rcParams xtick.color/ytick.color should handle this,
        # but explicit tick_params can also set the 'color' of the tick lines)
        ax.tick_params(axis='x', color='white') # For the x-tick lines
        ax.tick_params(axis='y', color='white') # For the y-tick lines

        # Create the legend and ensure its text is white
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='none', edgecolor='white')
        for text_obj in legend.get_texts():
            text_obj.set_color('white')
        
        ax.grid(True, linestyle='--', alpha=0.6, color='gray') # Grid lines kept gray for contrast
        fig.patch.set_alpha(0) # Make figure background transparent
        ax.set_facecolor('none') # Make axes background transparent

        # Set the color of the spines (axis border lines)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        st.pyplot(fig)

    finally:
        # Reset to default matplotlib styles to avoid affecting other plots in the app
        plt.rcParams.update(default_rcParams)


    st.divider()

    # --- Analysis and Conclusion Sections ---
    white_text("Analysis", level="h2")
    analysis_text = """
    <ol>
    <li>  <strong>Context Precision</strong>: All four models have the same Context Precision score of 1.0000, indicating they retrieve highly relevant context for the given queries. </li>
    <li> <strong>Context Recall</strong>:
        <ul>
        <li>   DeepSeek-r1:8B (0.9585) and Qwen2.5:7b (0.9644), Gemma:7b (0.9644) show strong recall. Llama3:8b (0.9546) is slightly lower. </li>
        </ul>
    </li>
    <li>  <strong>Faithfulness</strong>:
        <ul>
          <li>  Qwen2.5:7b (0.6650) and Gemma:7b (0.6504) have the highest Faithfulness. </li>
           <li> Deep Seek-r1:8b (0.5723) has the lowest Faithfulness among these. </li>
        </ul>
    </li>
    <li>  <strong>Answer Relevancy</strong>:
        <ul>
         <li>   DeepSeek-r1:8B (0.7988) has the highest Answer Relevancy. </li>
         <li>   Llama3:8B (0.6335) has the lowest in this set for relevancy. </li>
        </ul>
        </li>
    </ol>
    """
    st.markdown(f"<div style='color: white;'>{analysis_text}</div>", unsafe_allow_html=True)

    white_text("Comparative Analysis", level="h2")
    comparative_text = """
    <ol>
    <li>  DeepSeek-r1:8B shows strength in Answer Relevancy and good Context Precision/Recall but lower Faithfulness. </li>
    <li>  Gemma:7B and Qwen2.5:7b lead in Faithfulness and have strong Context Recall. </li>
    <li>  Qwen2.5:7B also shows good Answer Relevancy, making it a strong contender. </li>
    <li>  Llama3:8B is generally competitive but doesn't top any single metric in this dataset. </li>
    </ol>
    """
    st.markdown(f"<div style='color: white;'>{comparative_text}</div>", unsafe_allow_html=True)

    white_text("Conclusion", level="h2")
    conclusion_text = """
    <ul>
    <li>  If <strong>Accuracy (Faithfulness)</strong> is the priority, <strong>Qwen2.5:7b</strong> or <strong>Gemma:7b</strong> are strong choices. </li>
    <li>  If overall <strong>Response Relevance</strong> is key, <strong>DeepSeek-r1:8B</strong> stands out. </li>
    <li>   Consider <strong>Qwen2.5:7b</strong> for a good balance of Faithfulness and Answer Relevancy. </li>
    </ul>
    """
    st.markdown(f"<div style='color: white;'>{conclusion_text}</div>", unsafe_allow_html=True)
elif selected == "Notify":
    def notify():
        # Initialize Groq client with API key
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # ⚠️ Replace with your actual Groq API key

        # Initialize scheduler in session state
        if 'scheduler' not in st.session_state:
            scheduler = BackgroundScheduler()
            scheduler.start()
            st.session_state.scheduler = scheduler

        # Set Pakistan timezone as default
        PAKISTAN_TZ = pytz.timezone('Asia/Karachi')

        # Sidebar configuration
        with st.sidebar:
            st.subheader("API Configuration")
            
            # Model selection
            #groq_models = ["llama3-8b-8192", "qwen-qwq-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
            groq_models = ["llama3-8b", "qwen2.5-7b", "deepseek-r1-8b", "gemma2-9b"]
            selected_ll_model = st.selectbox("Select AI Model:", groq_models, index=0)
            if selected_ll_model == "llama3-8b":
                selected_model = "llama3-8b-8192"
            elif selected_ll_model == "qwen2.5-7b":
                selected_model = "qwen/qwen3-32b"
            elif selected_ll_model == "deepseek-r1-8b":
                selected_model = "deepseek-r1-distill-llama-70b"
            elif selected_ll_model == "gemma2-9b":
                selected_model = "gemma2-9b-it"
            
            # Initialize button
            if st.button("🚀 Initialize AI Client", key="init_groq"):
                try:
                    st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)
                    st.session_state.selected_model = selected_model
                    st.success("AI client initialized successfully!")
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
            
            TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]  # ⚠️ Replace with your actual Twilio Account SID
            TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]  # ⚠️ Replace with your actual Twilio Auth Token
            TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]

            OPENWEATHER_API_KEY = st.secrets["OPENWEATHERMAP_API_KEY"]  # ⚠️ Replace with your actual OpenWeatherMap API key

        # WhatsApp sending function
        def send_whatsapp_message(phone_number, message_body):
            try:
                twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                message = twilio_client.messages.create(
                    body=message_body,
                    from_=TWILIO_PHONE_NUMBER,
                    to=f'whatsapp:{phone_number}'
                )
                return message.sid
            except Exception as e:
                st.error(f"Error sending message: {str(e)}")
                return None

        # Weather API functions
        def get_current_weather(city, api_key):
            """Get current weather data for a city"""
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    return {
                        "temp": data["main"]["temp"],
                        "feels_like": data["main"]["feels_like"],
                        "humidity": data["main"]["humidity"],
                        "description": data["weather"][0]["description"],
                        "wind_speed": data["wind"]["speed"],
                        "city": data["name"],
                        "country": data["sys"]["country"]
                    }
                else:
                    st.error(f"Weather API error: {data.get('message', 'Unknown error')}")
                    return None
            except Exception as e:
                st.error(f"Failed to get weather: {str(e)}")
                return None

        def get_weather_forecast(city, api_key):
            """Get 5-day weather forecast with daily summaries"""
            try:
                url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    now = datetime.now(PAKISTAN_TZ)
                    today = now.date()
                    
                    daily_forecasts = {}
                    for forecast in data["list"]:
                        forecast_time = datetime.fromtimestamp(forecast["dt"], tz=pytz.UTC).astimezone(PAKISTAN_TZ)
                        forecast_date = forecast_time.date()
                        
                        if forecast_date < today:
                            continue
                            
                        if forecast_date not in daily_forecasts:
                            daily_forecasts[forecast_date] = {
                                "min_temp": float('inf'),
                                "max_temp": float('-inf'),
                                "descriptions": [],
                                "date_obj": forecast_date
                            }
                        
                        temp = forecast["main"]["temp"]
                        daily_forecasts[forecast_date]["min_temp"] = min(daily_forecasts[forecast_date]["min_temp"], temp)
                        daily_forecasts[forecast_date]["max_temp"] = max(daily_forecasts[forecast_date]["max_temp"], temp)
                        daily_forecasts[forecast_date]["descriptions"].append(forecast["weather"][0]["description"])
                    
                    forecast_days = sorted(daily_forecasts.keys())
                    forecast_summaries = []
                    
                    for i, day in enumerate(forecast_days):
                        if i >= 5:
                            break
                            
                        day_data = daily_forecasts[day]
                        
                        description_counts = {}
                        for desc in day_data["descriptions"]:
                            description_counts[desc] = description_counts.get(desc, 0) + 1
                        most_common_desc = max(description_counts, key=description_counts.get)
                        
                        if day == today:
                            date_str = "Today"
                        elif day == today + timedelta(days=1):
                            date_str = "Tomorrow"
                        else:
                            date_str = day.strftime("%A, %b %d")
                        
                        forecast_summaries.append({
                            "date": date_str,
                            "min_temp": day_data["min_temp"],
                            "max_temp": day_data["max_temp"],
                            "description": most_common_desc
                        })
                    
                    return forecast_summaries
                else:
                    st.error(f"Forecast API error: {data.get('message', 'Unknown error')}")
                    return None
            except Exception as e:
                st.error(f"Failed to get forecast: {str(e)}")
                return None

        # Main form
        with st.form("schedule_form"):
            phone_number = st.text_input(
                "Recipient's WhatsApp Number (with country code):",
                placeholder="e.g., +923001234567",
                max_chars=20,
                value="+92"
            )
            
            user_prompt = st.text_area(
                "User Prompt:",
                placeholder="e.g., 'What's the weather in Karachi today?' or 'Send me the 5-day weather forecast for Lahore'",
                height=150,
                help="Ask about weather or request a forecast. Mention the city name."
            )
            
            weather_city = st.text_input(
                "City for Weather (Required):",
                placeholder="e.g., Karachi",
                help="Specify city for weather information"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                selected_date = st.date_input("Select Date", min_value=datetime.today())
            with col2:
                selected_time = st.time_input("Select Time", step=60)
            
            st.markdown(f"**Timezone:** Asia/Karachi (PKT)")
            
            # --- START: MODIFIED CODE ---
            # Create two columns for the buttons
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                schedule_button = st.form_submit_button("✉️ Schedule Message", use_container_width=True)
            with b_col2:
                instant_button = st.form_submit_button("⚡ Instant Message", use_container_width=True)
            # --- END: MODIFIED CODE ---

        # After form submission logic for both buttons
        if schedule_button or instant_button:
            # Common validation for both actions
            errors = []
            if not phone_number:
                errors.append("Please enter a phone number")
            if not user_prompt.strip():
                errors.append("Please enter a message prompt")
            if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
                errors.append("Twilio credentials are incomplete")
            
            # Combine date and time
            try:
                scheduled_datetime = PAKISTAN_TZ.localize(datetime.combine(selected_date, selected_time))
            except ValueError:
                scheduled_datetime = PAKISTAN_TZ.localize(datetime.combine(selected_date, selected_time), is_dst=None)
            
            current_datetime = datetime.now(PAKISTAN_TZ)
            
            # Schedule-specific validation
            if schedule_button and scheduled_datetime <= current_datetime:
                errors.append("Please select a future time for scheduling")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Common logic for weather fetching and AI message generation
                weather_info = None
                forecast_info = None
                city_to_use = weather_city.strip() if weather_city.strip() else None

                # Try to extract city from user_prompt if not provided
                if not city_to_use:
                    import re
                    # Simple regex to find a city name after "in" or "for"
                    match = re.search(r"(?:in|for)\s+([A-Za-z\s]+)", user_prompt, re.IGNORECASE)
                    if match:
                        city_to_use = match.group(1).strip()

                if not city_to_use:
                    st.error("Please specify a city for weather information (either in the city field or in your prompt).")
                else:
                    if OPENWEATHER_API_KEY:
                        weather_info = get_current_weather(city_to_use, OPENWEATHER_API_KEY)
                        forecast_info = get_weather_forecast(city_to_use, OPENWEATHER_API_KEY)
                    else:
                        st.error("OpenWeather API key is missing.")
                
                with st.spinner("Generating message with AI..."):
                    final_message = ""
                    weather_report_string = ""
                    if weather_info:
                        weather_report_string += (
                            f"Here is the current weather for {weather_info['city']}, {weather_info['country']}:\n"
                            f"- Temperature: {weather_info['temp']}°C (feels like {weather_info['feels_like']}°C)\n"
                            f"- Conditions: {weather_info['description'].capitalize()}\n\n"
                        )

                    if forecast_info:
                        weather_report_string += f"Here's the 5-day forecast for {city_to_use}:\n"
                        for day in forecast_info:
                            # If date is "Today" or "Tomorrow", add the actual date in parentheses
                            if day['date'] == "Today":
                                actual_date = datetime.now(PAKISTAN_TZ).strftime("%A, %d %b %Y")
                                date_str = f"Today ({actual_date})"
                            elif day['date'] == "Tomorrow":
                                actual_date = (datetime.now(PAKISTAN_TZ) + timedelta(days=1)).strftime("%A, %d %b %Y")
                                date_str = f"Tomorrow ({actual_date})"
                            else:
                                date_str = day['date']  # Already a date string like "Monday, Jul 29"
                            weather_report_string += f"- {date_str}: {day['description'].capitalize()}, {day['min_temp']:.1f}°C to {day['max_temp']:.1f}°C\n"
                    
                    if "groq_client" in st.session_state:
                        try:
                            system_prompt = """
                            You are an assistant who formats messages for WhatsApp.
                            You will be given a user's request and pre-formatted weather information.
                            Your task is to combine these into a friendly and concise message.
                            DO NOT change, rephrase, or summarize the weather data or the dates provided.
                            You MUST copy the weather information and dates EXACTLY as given between ---BEGIN WEATHER--- and ---END WEATHER---.
                            If the user asks for a summary, still use the exact dates and weather data as provided.
                            Start with a greeting based on the user's prompt and then present the weather information.
                            """
                            
                            user_content = (
                                f"The user wants to know: '{user_prompt}'.\n\n"
                                "Please create a friendly WhatsApp message that includes the following weather information exactly as written:\n"
                                f"---BEGIN WEATHER---\n{weather_report_string}\n---END WEATHER---"
                            )

                            if not weather_report_string:
                                user_content = f"Create a friendly WhatsApp message for the following request: {user_prompt}"

                            response = st.session_state.groq_client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_content}
                                ],
                                model=st.session_state.selected_model,
                                temperature=0.7,
                                max_tokens=400
                            )
                            # Split LLM response at the weather delimiter, keep only the greeting/intro
                            llm_response = response.choices[0].message.content.strip()
                            if "---BEGIN WEATHER---" in llm_response:
                                greeting = llm_response.split("---BEGIN WEATHER---")[0].strip()
                                final_message = f"{greeting}\n\n{weather_report_string}"
                            else:
                                # If LLM doesn't use delimiter, just prepend greeting
                                final_message = f"{llm_response}\n\n{weather_report_string}"
                            # After building final_message (right after the if/else block that sets final_message)
                            MAX_MESSAGE_LENGTH = 1600
                            if len(final_message) > MAX_MESSAGE_LENGTH:
                                # Truncate without breaking words if possible
                                truncated = final_message[:MAX_MESSAGE_LENGTH]
                                last_newline = truncated.rfind('\n')
                                if last_newline > 0 and last_newline > MAX_MESSAGE_LENGTH - 100:
                                    truncated = truncated[:last_newline]
                                final_message = truncated + "\n\n[Message truncated due to length limit]"

                        except Exception as e:
                            st.error(f"AI generation failed: {str(e)}")
                            final_message = weather_report_string if weather_report_string else user_prompt
                    else:
                        final_message = weather_report_string if weather_report_string else user_prompt
                        st.warning("AI client not initialized. Using direct weather data.")
                
                # --- START: MODIFIED CODE ---
                # Action-specific logic: Schedule or Send Instantly
                if schedule_button:
                    with st.spinner(f"Scheduling message for {scheduled_datetime.strftime('%Y-%m-%d %I:%M %p %Z')}..."):
                        try:
                            job = st.session_state.scheduler.add_job(
                                send_whatsapp_message,
                                trigger=DateTrigger(run_date=scheduled_datetime),
                                args=[phone_number, final_message],
                                id=f"whatsapp_{scheduled_datetime.timestamp()}"
                            )
                            
                            st.success(f"📨 Message Scheduled Successfully for {scheduled_datetime.strftime('%Y-%m-%d %I:%M %p %Z')}!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Scheduled Details")
                                st.write(f"**To:** {phone_number}")
                                st.write(f"**Scheduled Time:** {scheduled_datetime.strftime('%Y-%m-%d %I:%M %p %Z')}")
                                st.write(f"**AI Model:** {st.session_state.get('selected_model', 'Not used')}")
                                if city_to_use:
                                    st.write(f"**Weather City:** {city_to_use}")
                            
                            with col2:
                                st.subheader("Message Preview")
                                st.info(final_message)
                            
                            st.markdown("---")
                            st.info("ℹ️ The message will be automatically sent at the scheduled time. Keep this app running.")
                            
                        except Exception as e:
                            st.error(f"Scheduling failed: {str(e)}")

                elif instant_button:
                    with st.spinner("Sending message instantly..."):
                        message_sid = send_whatsapp_message(phone_number, final_message)
                        if message_sid:
                            st.success(f"⚡ Message sent successfully!")
                            st.subheader("Message Details")
                            st.write(f"**To:** {phone_number}")
                            st.write(f"**Sent Time:** {datetime.now(PAKISTAN_TZ).strftime('%Y-%m-%d %I:%M %p %Z')}")
                            if selected_model== "llama3-8b-8192":
                                selected_model = "llama3-8b"
                            elif selected_model == "qwen/qwen3-32b":
                                selected_model = "qwen2.5-7b"
                            elif selected_model == "deepseek-r1-distill-llama-70b":
                                selected_model = "deepseek-r1-8b"
                            elif selected_model == "gemma2-9b-it":
                                selected_model = "gemma2-9b"
                            st.session_state['selected_model'] = selected_model
                            st.write(f"**AI Model:** {st.session_state.get('selected_model', 'Not used')}")
                            if city_to_use:
                                st.write(f"**Weather City:** {city_to_use}")
                            st.subheader("Message Content")
                            st.info(final_message)
                        else:
                            st.error("Failed to send message. Please check the logs or your Twilio credentials.")
                # --- END: MODIFIED CODE ---

        # Add footer
        st.markdown("---")
        st.caption("LLM WhatsApp Scheduler v1.3 | Weather Powered by OpenWeatherMap | Pakistan Time (PKT)")

        # Display scheduled jobs
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
    notify()