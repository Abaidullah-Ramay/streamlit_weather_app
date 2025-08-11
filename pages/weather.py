# pages/weather.py
import streamlit as st
import datetime as dt
from weather_utils import get_weather_data, get_weekly_forecast, display_weekly_forecast
from config import OPENWEATHER_API_KEY

def show_page():
    """Renders the Weather page."""
    st.sidebar.title("Weather forecasting with LLM")
    city = st.sidebar.text_input("Enter city name", "Islamabad")

    if not OPENWEATHER_API_KEY:
        st.sidebar.error("OpenWeather API key not found. Please set it in your Streamlit secrets.")
        st.markdown("<p style='color: red;'>Weather forecast functionality is unavailable due to a missing API key.</p>", unsafe_allow_html=True)
        return

    submit = st.sidebar.button("Get Weather")

    if submit:
        try:
            weather_main_data = get_weather_data(city, OPENWEATHER_API_KEY)

            if weather_main_data and weather_main_data.get("cod") == 200:
                st.sidebar.write(f"{city} Latitude", weather_main_data["coord"]["lat"])
                st.sidebar.write(f"{city} Longitude", weather_main_data["coord"]["lon"])
                st.sidebar.write("Sunrise Time", dt.datetime.utcfromtimestamp(weather_main_data["sys"]["sunrise"] + weather_main_data["timezone"]))
                st.sidebar.write("Sunset Time", dt.datetime.utcfromtimestamp(weather_main_data["sys"]["sunset"] + weather_main_data["timezone"]))

                current_datetime = dt.datetime.now()
                st.sidebar.write(current_datetime.date().strftime("%d-%m-%Y"))
                st.sidebar.write(current_datetime.time().strftime("%H:%M:%S"))

                st.markdown(f"<h1 style='color: white; text-align: center;'>Weather updates for {city} is:</h1>", unsafe_allow_html=True)

                with st.spinner("Fetching weather data..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Temperature 🌡", f"{weather_main_data['main']['temp'] - 273.15:.2f} °C")
                        st.metric("Humidity 💧", f"{weather_main_data['main']['humidity']}%")
                    with col2:
                        st.metric("Pressure", f"{weather_main_data['main']['pressure']} hPa")
                        st.metric("Wind Speed 🍃 🍂", f"{weather_main_data['wind']['speed']} m/s")

                    lat = weather_main_data['coord']['lat']
                    lon = weather_main_data['coord']['lon']

                    forecast_data = get_weekly_forecast(OPENWEATHER_API_KEY, lat, lon)
                    if forecast_data and "list" in forecast_data:
                        display_weekly_forecast(forecast_data)
                    else:
                        st.markdown("<p style='color: red;'>Error fetching weekly forecast data or data format incorrect.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: red;'>City '{city}' not found or an error occurred: {weather_main_data.get('message', 'Unknown error')}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<p style='color: red;'>An error occurred while fetching weather data: {e}</p>", unsafe_allow_html=True)