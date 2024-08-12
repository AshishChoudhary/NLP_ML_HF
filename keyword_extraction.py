import streamlit as st
from keybert import KeyBERT

# Initialize the KeyBERT model
kw_model = KeyBERT()

# Title of the Streamlit app
st.title("Keyword Extraction with n-grams and MMR")

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

# Footer
st.write("Powered by KeyBERT and Streamlit")
