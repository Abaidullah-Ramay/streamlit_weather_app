# main.py
import streamlit as st
from streamlit_option_menu import option_menu
from config import set_page_config, hide_default_streamlit_ui, apply_page_styling
from pages import description, weather, chatbot, results, notify

# --- Initial Page Setup ---
set_page_config()
hide_default_streamlit_ui()

# --- Main Navigation Menu ---
selected = option_menu(
    menu_title=None,
    options=["Description", "Weather", "Chatbot", "Results", "Notify"],
    icons=["house", "cloud", "chat", "megaphone", "whatsapp"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#2D3748", "width": "100%", "flex-wrap": "nowrap"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px", "text-align": "left", "margin": "0px",
            "--hover-color": "#4A5568", "color": "#FFFFFF", "white-space": "nowrap",
            "border-bottom": "2px solid transparent",
            ":hover": {"border-bottom": "2px solid #63B3ED"}
        },
        "nav-link-selected": {"background-color": "#4299E1", "color": "#FFFFFF"},
    }
)

# --- Apply Background and Styles for the Selected Page ---
apply_page_styling(selected)

# --- Page Routing ---
if selected == "Description":
    description.show_page()
elif selected == "Weather":
    weather.show_page()
elif selected == "Chatbot":
    chatbot.show_page()
elif selected == "Results":
    results.show_page()
elif selected == "Notify":
    notify.show_page()