import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from keybert import KeyBERT

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize pipelines and models
zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-base"
)

summarizer = pipeline("summarization", model="facebook/bart-base")

kw_model = KeyBERT()

# Define the function for Zero-Shot Classification
def zero_shot_classification():
    st.subheader("Zero-Shot Classification with Hugging Face Transformers")

    # User inputs for the sequence and candidate labels
    sequence = st.text_input("Enter the sequence to classify:", "Can you order some Pizza & book an Uber to the nearest cinema House at 10 PM?")
    candidate_labels = st.text_input("Enter the candidate labels (comma-separated):", "Flight Travel, Cabs Travel, Reminders, Food, Movies")

    # Convert the candidate labels to a list
    candidate_labels_list = [label.strip() for label in candidate_labels.split(",")]

    # Classification button
    if st.button("Classify"):
        # Perform zero-shot classification
        result = zero_shot_classifier(
            sequences=sequence,
            candidate_labels=candidate_labels_list,
            multi_label=True
        )
        
        # Display the classification results as a bar chart
        st.write("### Classification Results")
        fig, ax = plt.subplots()
        ax.bar(result["labels"], result["scores"])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylabel("Confidence Score")
        ax.set_xlabel("Labels")
        ax.set_title("Zero-Shot Classification Results")
        st.pyplot(fig)

# Define the function for Text Summarization
def text_summarization():
    st.subheader("Text Summarization App")

    # Input text box
    text = st.text_area("Enter text to summarize", 
                        "Artificial Intelligence (AI) is a rapidly advancing field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence...")

    # Parameters for summarization
    max_length = st.slider("Max length of summary", 20, 100, 60)
    min_length = st.slider("Min length of summary", 5, 50, 25)

    # Summarize on button click
    if st.button("Summarize"):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        st.write("Summary:")
        st.write(summary[0]['summary_text'])

# Define the function for Keyword Extraction
def keyword_extraction():
    st.subheader("Keyword Extraction with n-grams and MMR")

    # Text input from the user
    text = st.text_area("Enter text for keyword extraction", 
                        "Artificial Intelligence is revolutionizing the tech industry by enabling machines to learn from data and make decisions with minimal human intervention.")

    # Parameters input
    ngram_range = st.slider("Select n-gram range", 1, 3, (1, 2))  # min value, max value, default value
    top_n = st.number_input("Number of keywords to extract", min_value=1, max_value=20, value=5)
    diversity = st.slider("Select MMR diversity", 0.0, 1.0, 0.7)

    # Keyword extraction on button click
    if st.button("Extract Keywords"):
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            use_mmr=True,
            diversity=diversity,
            stop_words='english',
            top_n=top_n
        )

        # Display extracted keywords
        st.write("Extracted Keywords:")
        for keyword in keywords:
            st.write(f"- {keyword[0]}")

# Main routing logic
def main():
    st.title("Multi-Function Streamlit App")

    # Create a sidebar for navigation
    option = st.sidebar.selectbox(
        "Choose a function",
        ("Zero-Shot Classification", "Text Summarization", "Keyword Extraction")
    )

    # Route to the selected function
    if option == "Zero-Shot Classification":
        zero_shot_classification()
    elif option == "Text Summarization":
        text_summarization()
    elif option == "Keyword Extraction":
        keyword_extraction()

# Run the app
if __name__ == "__main__":
    main()
