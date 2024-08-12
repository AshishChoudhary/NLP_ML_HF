import streamlit as st
from transformers import pipeline

# Initialize the summarization pipeline
#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-base")

# Streamlit app title
st.title("Text Summarization App")

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
