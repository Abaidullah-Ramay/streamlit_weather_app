# chatbot_utils.py
import streamlit as st
from gtts import gTTS
from io import BytesIO
import base64

def text_to_speech(text):
    """Generates audio from text and plays it in the Streamlit app."""
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
        st.components.v1.html(audio_html, height=0)
        
    except Exception as e:
        st.error(f"TTS Error: {e}")