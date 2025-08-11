# weather_utils.py
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy
import datetime as dt

def get_weather_data(city, weather_api_key):
    """Fetches current weather data from OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={weather_api_key}&q={city}"
    response = requests.get(complete_url)
    return response.json()

def get_weekly_forecast(weather_api_key, lat, lon):
    """Fetches 5-day/3-hour forecast data."""
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = f"{base_url}lat={lat}&lon={lon}&appid={weather_api_key}"
    response = requests.get(complete_url)
    return response.json()

def plot_weather_data(list_days, list_temp, list_humidity, list_pressure, list_wind_speed, list_cloudiness):
    """Creates and returns a matplotlib figure with weather plots."""
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
    """Displays the weekly forecast and plots in Streamlit."""
    try:
        st.write("=" * 88)
        st.write("### Weekly Weather Forecast")
        displayed_dates = set()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.subheader("Day")
        with c2: st.subheader("Description")
        with c3: st.subheader("Min Temp (°C)")
        with c4: st.subheader("Max Temp (°C)")

        list_days, list_temp, list_humidity, list_pressure, list_wind_speed, list_cloudiness = [], [], [], [], [], []

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

                with c1: st.write(f"{date}")
                with c2: st.write(f"{description.capitalize()}")
                with c3: st.write(f"{min_temp:.1f}°C")
                with c4: st.write(f"{max_temp:.1f}°C")

        st.title("Weather Parameters Over Days")
        fig = plot_weather_data(list_days, list_temp, list_humidity, list_pressure, list_wind_speed, list_cloudiness)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in displaying weekly forecast: {e}")