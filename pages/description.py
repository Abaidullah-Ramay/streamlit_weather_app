# pages/description.py
import streamlit as st
import base64
import os

def image_to_base64(img_path):
    """Converts a local image to a base64 string."""
    if not os.path.exists(img_path):
        st.error(f"Image not found: {img_path}. Please ensure it's in the correct location.")
        return "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" # 1x1 transparent gif
    with open(img_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

def show_page():
    """Renders the Description page."""
    bg_img_path = "logos/uni_background.jpg"
    uni_logo_path = "logos/Nust_new_logo.png"
    dept_logo_path = "logos/Sines_new_logo.png"

    # Assume logos are in a 'logos' subdirectory. Create it if needed.
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
                color: #2E86C1;
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
    This project explores the use of a Large Language Model (LLM) as a chatbot, equipped with a Retrieval Augmented Generation (RAG) pipeline. The data used is a vast set of websites scraped from the internet on Weather and Climatic studies. This project covers the complete pipeline of RAG, starting from **loading scraped data into a vector database** to **retrieval** of relevant chunks to **generation** by the LLM.
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
    - **Qdrant:** For vector storage and retrieval of weather-related data.

    ### Future Work
    - **Enhancing Data Sources:** Integrating more diverse data sources for improved accuracy and insights.
    - **Improving User Experience:** Adding more interactive features and improving the chatbot's conversational abilities.
    - **Expanding Functionality:** Including more advanced weather analytics and predictions.
    </div>
    """, unsafe_allow_html=True)