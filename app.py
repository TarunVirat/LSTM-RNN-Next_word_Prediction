import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# If model files don’t exist, run training to generate them
if not (os.path.exists("next_word_lstm.weights.h5") and os.path.exists("model_architecture.json")):
    import train

# Load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load config
with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)

total_words = config["total_words"]
max_sequence_len = config["max_sequence_len"]

# Load model architecture
with open("model_architecture.json", "r") as json_file:
    model_json = json_file.read()
model = keras.models.model_from_json(model_json)

# Load weights
model.load_weights("next_word_lstm.weights.h5")

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")
if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {next_word}")
