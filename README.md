## Chatbot for Weather Insights and Recommendations

This is a multi-page Streamlit web application that serves as an intelligent weather assistant. It combines real-time weather data, a Retrieval Augmented Generation (RAG) powered chatbot, performance analytics, and a notification system to provide a comprehensive user experience.

### ❯ Key Features

* **Real-time Weather Forecast**: Fetches and displays current weather conditions and a 5-day forecast for any city using the OpenWeatherMap API. Includes visual plots for temperature, humidity, wind speed, and more.

* **RAG-Powered Chatbot:** An interactive chatbot that answers user questions about weather and climate. It uses a RAG pipeline with a Qdrant vector database and the Groq API for fast LLM inference to provide context-aware responses.

* **Voice-to-Text Input:** Users can interact with the chatbot using their voice, which is transcribed into text for a hands-free experience.

* **Model Performance Analysis:** A dedicated "Results" page displays a comparative analysis of different LLMs (Llama3, Qwen, Gemma2, DeepSeek) using RAGAS evaluation metrics like Faithfulness and Answer Relevancy.

* **WhatsApp Notification Scheduler:** Allows users to schedule weather updates and custom messages to be sent to any WhatsApp number at a specific date and time, powered by the Twilio API.

* **Modular & Clean Interface:** A multi-page application with a clean, intuitive UI built with Streamlit and streamlit-option-menu.

### ❯ Tech Stack

* **Frontend:** Streamlit, Streamlit Option Menu

* **AI & Machine Learning:**

    * **LLM Inference:** Groq API (for Llama3, Gemma2, etc.)
    * **Framework:** LangChain
    * **Embeddings:** OpenAI

    * **Vector Database:** Qdrant

* **Data APIs:**

    * **Weather:** OpenWeatherMap API

    * **Notifications:** Twilio API for WhatsApp

* **Utilities:** Pandas, Matplotlib, gTTS (for Text-to-Speech), APScheduler (for scheduling)

### ❯ Project Structure

The project is organized into a modular structure for better readability and maintenance.

 ``` 

your_project_directory/
├── main.py                 # Main application entry point
├── config.py               # API keys, page setup, and global CSS
├── weather_utils.py        # Functions for fetching and plotting weather data
├── chatbot_utils.py        # Helper functions for the chatbot page
├── requirements.txt        # List of project dependencies
└── pages/
    ├── __init__.py         # Makes 'pages' a Python package
    ├── description.py      # UI code for the "Description" page
    ├── weather.py          # UI code for the "Weather" page
    ├── chatbot.py          # UI code for the "Chatbot" page
    ├── results.py          # UI code for the "Results" page
    └── notify.py           # UI code for the "Notify" page
```
### ❯ Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**
```
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

**2. Create a Virtual Environment**

It's recommended to create a virtual environment to manage project dependencies.

```
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies

Install all the required Python libraries from the requirements.txt file.
```
pip install -r requirements.txt
```

### 4. Configure API Keys and Secrets

This project requires several API keys to function correctly. Streamlit's secrets management is used to handle them securely.

Create a directory named .streamlit in the root of your project folder, and inside it, create a file named secrets.toml.

```
your_project_directory/
└── .streamlit/
    └── secrets.toml
```
Add the following content to your secrets.toml file and replace the placeholder values with your actual credentials.

```
# .streamlit/secrets.toml

# OpenAI API Key (for text embeddings)
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# OpenWeatherMap API Key
OPENWEATHERMAP_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Groq API Key (for LLM inference)
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Qdrant Vector Database Credentials
QDRANT_HOST = "your-qdrant-cluster-url.qdrant.tech"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION_NAME = "your_collection_name"

# Twilio Credentials (for WhatsApp notifications)
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886" # Your Twilio sandbox or purchased number
```
### ❯ How to Run the Application

Once the dependencies are installed and the secrets.toml file is configured, you can run the application with a single command:
```
streamlit run main.py
```
The application will open in your default web browser.

### ❯ Page Descriptions

* **Description:** The landing page providing an overview of the project, its objectives, and institutional details.

* **Weather:** An interactive page to get real-time weather data and multi-day forecasts for any city.

* **Chatbot:** The core conversational AI where you can ask weather and climate-related questions. Remember to click "Initialize Chat" first.

* **Results:** A page dedicated to showcasing the performance metrics and comparative analysis of the different LLMs used in the RAG pipeline.

* **Notify:** A powerful tool to schedule and send weather alerts or custom messages via WhatsApp to any phone number.

### ❯ Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

&nbsp;1. Fork the repository.

&nbsp;2. Create a new branch ```(git checkout -b feature/YourFeature)```.

&nbsp;3. Commit your changes ```(git commit -m 'Add some feature')```.

&nbsp;4. Push to the branch ```(git push origin feature/YourFeature)```.

&nbsp;5. Open a Pull Request.

### ❯ License

This project is licensed under the MIT License. See the LICENSE file for more details.