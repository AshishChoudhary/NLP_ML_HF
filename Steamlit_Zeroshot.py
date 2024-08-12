import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1-3)

# Use a CPU-only model with the updated multi_label argument
zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",  # Explicitly specifying the model
)

# Streamlit app title
st.title("Zero-Shot Classification with Hugging Face Transformers")

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
        multi_label=True  # Updated argument name
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

# Footer
st.write("Powered by [Hugging Face Transformers](https://huggingface.co/transformers/) and [Streamlit](https://streamlit.io/).")
