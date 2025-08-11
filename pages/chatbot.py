# pages/chatbot.py
import streamlit as st
from groq import Groq
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_qdrant import Qdrant
import qdrant_client
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from chatbot_utils import text_to_speech
from config import (
    CHATBOT_CSS, GROQ_API_KEY, QDRANT_HOST, QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME, OPENAI_API_KEY
)

def show_page():
    """Renders the Chatbot page."""
    st.write(CHATBOT_CSS, unsafe_allow_html=True)

    # --- Session State Management ---
    if "conversation" not in st.session_state: st.session_state.conversation = None
    if "messages" not in st.session_state: st.session_state.messages = []
    if "user_input_value" not in st.session_state: st.session_state.user_input_value = None
    if "vector_store" not in st.session_state: st.session_state.vector_store = None
    if "groq_client" not in st.session_state: st.session_state.groq_client = None
    if "input_reset_key" not in st.session_state: st.session_state.input_reset_key = 0
        
    # --- UI Layout ---
    st.markdown("<h2 style='color: white; text-align: center;'> 🌦️ Smart Weather Assistant</h2>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Configuration")
        models = ["llama3-8b", "qwen2.5-7b", "deepseek-r1-8b", "gemma2-9b"]
        selected_llm_model = st.selectbox("Select Model:", models, key="model_select")
        
        model_map = {
            "llama3-8b": "llama3-8b-8192",
            "qwen2.5-7b": "qwen/qwen3-32b",
            "deepseek-r1-8b": "deepseek-r1-distill-llama-70b",
            "gemma2-9b": "gemma2-9b-it"
        }
        selected_model = model_map.get(selected_llm_model)

        if st.button("🚀 Initialize Chat", key="init_chat"):
            with st.spinner("Initializing AI components..."):
                try:
                    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                    qdrant_client_instance = qdrant_client.QdrantClient(QDRANT_HOST, api_key=QDRANT_API_KEY)
                    st.session_state.vector_store = Qdrant(
                        client=qdrant_client_instance,
                        collection_name=QDRANT_COLLECTION_NAME,
                        embeddings=embeddings
                    )
                    st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)
                    st.session_state.conversation = True
                    st.sidebar.success("Chat initialized successfully!")
                except Exception as e:
                    st.sidebar.error(f"Initialization failed: {str(e)}")

        st.subheader("Voice Input (STT)")
        try:
            audio_info = mic_recorder(start_prompt="🎤 Start", stop_prompt="⏹️ Stop", key="mic_recorder", format="wav")
            if audio_info and 'bytes' in audio_info:
                st.sidebar.info("Transcribing audio...")
                recognizer = sr.Recognizer()
                audio_data = sr.AudioData(audio_info['bytes'], audio_info['sample_rate'], 2)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.session_state.user_input_value = text
                    st.session_state.input_reset_key += 1
                    st.rerun()
                except sr.UnknownValueError:
                    st.sidebar.error("Could not understand audio")
                except sr.RequestError as e:
                    st.sidebar.error(f"Recognition error: {e}")
        except ImportError:
            st.sidebar.warning("Required libraries for voice input are not installed.")
        except Exception as e:
            st.sidebar.error(f"Voice error: {e}")

    # --- Chat Interface ---
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if st.button(f"🔊", key=f"tts_button_{i}", help="Play assistant's response"):
                    clean_text_for_speech = message['content'].split("**Document Sources:**")[0].strip()
                    with st.spinner("Preparing audio..."): 
                        text_to_speech(clean_text_for_speech)

    input_placeholder = st.empty()
    if st.session_state.user_input_value:
        prompt = input_placeholder.chat_input("Ask about documents...", value=st.session_state.user_input_value, key=f"chat_input_{st.session_state.input_reset_key}")
    else:
        prompt = input_placeholder.chat_input("Ask about documents...", key=f"chat_input_{st.session_state.input_reset_key}")
    
    prompt = prompt or st.session_state.user_input_value
    st.session_state.user_input_value = None

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
                        {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question:\n{context}\nIf the context is not enough, use your general knowledge."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = st.session_state.groq_client.chat.completions.create(messages=messages, model=selected_model)
                    final_answer = response.choices[0].message.content
                    if docs:
                        sources = "\n\n**Document Sources:**\n" + "\n".join([f"- {doc.metadata.get('source', 'Unknown source')}" for doc in docs])
                        final_answer += sources
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                except Exception as e:
                    error_message = f"Sorry, an unexpected error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.rerun()
        else:
            st.warning("Chat not initialized. Please select a model and click 'Initialize Chat' in the sidebar.")