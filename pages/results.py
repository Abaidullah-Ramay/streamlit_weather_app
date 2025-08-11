# pages/results.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def white_text(text_content, level="p", text_align="left"):
    """Helper to display text in white."""
    st.markdown(f"<{level} style='color: white; text-align: {text_align};'>{text_content}</{level}>", unsafe_allow_html=True)

def show_page():
    """Renders the RAGAS evaluation results page."""
    white_text("RAGAS Evaluation", level="h1", text_align="center")
    white_text("Synthetic questions and models responses", level="h2")
    white_text("We first generate a synthetic test set for our data using the RAGAS library. We use the GPT-4o-mini model to generate a test set comprising 50 questions and reference answers. We then pass these questions to four models, namely DeepSeek-R1, Qwen2.5, Gemma2, Llama3, to generate responses. Given below are 3 samples out of our 50-set:")

    # Define questions, answers, and scores data
    questions = [
        "Q1. How does the climate of Gilgit-Baltistan differ from other regions in Pakistan?",
        "Q2. What are the climatic challenges faced by Balochistan, particularly regarding drought?",
        "Q3. How has climate change affected Karachi in recent years?"
    ]
    # ... (Keep the original answers and scores_bank dictionaries)
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


    st.markdown("<style>div[data-testid='stExpander'] summary {color: white !important;}</style>", unsafe_allow_html=True)

    for question_text in questions:
        with st.expander(question_text):
            q_key = question_text.split('.')[0] + "."
            score_key = f"{q_key} Scores"
            for model_type, response_text in answers.get(question_text, {}).items():
                st.markdown(f"<p style='color: white;'><strong>{model_type}:</strong> {response_text}</p>", unsafe_allow_html=True)
                if model_type != "Reference Answer":
                    model_score_key = f"{model_type.replace(' Answer', '')} Scores"
                    scores_data = scores_bank.get(score_key, {}).get(model_score_key, {})
                    if scores_data:
                        st.dataframe(pd.DataFrame([scores_data]))

    st.divider()
    white_text("Generate Average Scores", level="h2")
    avg_data = {"Models": ["Deep Seek-r1:8b", "Qwen2.5:7b", "Llama3:8b", "Gemma:7b"], "Context Precision": [1.0, 1.0, 1.0, 1.0], "Context Recall": [0.9585, 0.9644, 0.9546, 0.9644], "Faithfulness": [0.5723, 0.6650, 0.5926, 0.6504], "Answer Relevancy": [0.7988, 0.7281, 0.6335, 0.7049]}
    df_avg_scores = pd.DataFrame(avg_data)
    st.dataframe(df_avg_scores, hide_index=True)

    st.divider()
    white_text("Generating Plot", level="h2")
    fig, ax = plt.subplots(figsize=(12, 7))
    df_avg_scores.plot(x='Models', y=['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Relevancy'], kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison', color='white')
    ax.set_xlabel('Models', color='white')
    ax.set_ylabel('Scores', color='white')
    ax.tick_params(colors='white', which='both')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    st.pyplot(fig)

    st.divider()
    white_text("Analysis", level="h2")
    st.markdown("<div style='color: white;'>...</div>", unsafe_allow_html=True) # Add analysis text
    
    white_text("Conclusion", level="h2")
    st.markdown("<div style='color: white;'>...</div>", unsafe_allow_html=True) # Add conclusion text