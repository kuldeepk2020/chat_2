import streamlit as st
from gramformer import Gramformer
import spacy
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure the spaCy model is available
try:
    spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    spacy.load('en_core_web_sm')

# Load the Gramformer model
def load_model():
    gf = Gramformer(models=1, use_gpu=False)
    return gf

# Function to correct user input using Gramformer
def correct_english(gf, input_text):
    corrected_sentences = gf.correct(input_text)
    return list(corrected_sentences)[0] if corrected_sentences else input_text

# Streamlit UI
def main():
    st.title("High-Level English Learning Chat App")
    st.write("Welcome to the English Learning Chat App!")
    
    # Load the model once at the beginning
    gf = load_model()

    user_input = st.text_input("Enter your sentence:")
    
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a sentence.")
        else:
            corrected_sentence = correct_english(gf, user_input)
            st.text_area("Corrected Sentence:", value=corrected_sentence, height=100)
    
if __name__ == "__main__":
    main()
