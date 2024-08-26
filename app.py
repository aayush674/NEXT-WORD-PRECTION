import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import speech_recognition as sr
import pickle

# Load the model
model = tf.keras.models.load_model('next_word_predictor.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    mytokenizer = pickle.load(handle)

# Function to predict the next words and generate multiple suggestions
def predict_next_words(model, tokenizer, text, max_seq_len, num_suggestions=3):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_indices = np.argsort(predicted_probs)[-num_suggestions:][::-1]
    
    suggestions = []
    for index in predicted_indices:
        output_word = ""
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                output_word = word
                break
        suggestions.append(text + " " + output_word)
    
    return suggestions

# Custom CSS for background image and styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
    body {
        background: url('../image/images.jpg') no-repeat center center fixed;
        height: 100vh;
        background-size: cover;
        margin: 0;
        overflow: hidden;
        font-family: 'Pacifico', cursive;
    }
    .title {
        text-align: center;
        font-size: 3rem;
        color: #ffffff;
        margin-bottom: 20px;
        font-family: 'Pacifico', cursive;
        text-shadow: 2px 2px 5px red;
    }
    .input-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
        font-family: 'Times New Roman', sans-serif;
    }
    .input-box {
        width: 300px;
        padding: 10px;
        font-size: 1rem;
        border: 1px solid #ccc;
        border-radius: 5px;
        margin-right: 10px;
        text-shadow: 1px 1px 3px red;
    }
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<div class="title">Next Word Predictor</div>', unsafe_allow_html=True)

# Take text input from the user
input_text = st.text_input('Enter your text here:')

# Button to predict the next words
if st.button('Predict Next Words'):
    if input_text:
        max_seq_len = model.input_shape[1] + 1  # assuming input shape is (None, max_seq_len-1)
        predictions = predict_next_words(model, mytokenizer, input_text, max_seq_len)
        for idx, suggestion in enumerate(predictions):
            st.write(f'Suggestion {idx+1}: {suggestion}')

    else:
        st.write('Please enter some text to predict the next words.')

# Audio recognition
if st.button('Record Audio'):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Adjusting noise and recording for 4 seconds...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recorded_audio = recognizer.listen(source, timeout=4)
        st.write("Done recording.")
    try:
        text = recognizer.recognize_google(recorded_audio, language="en-US")
        st.write(f"Recognized Text: {text}")
        if text:
            max_seq_len = model.input_shape[1] + 1  # assuming input shape is (None, max_seq_len-1)
            predictions = predict_next_words(model, mytokenizer, text, max_seq_len)
            for idx, suggestion in enumerate(predictions):
                st.write(f'Suggestion {idx+1}: {suggestion}')
    except Exception as e:
        st.write(f"Error: {str(e)}")
