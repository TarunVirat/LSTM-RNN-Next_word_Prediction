import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
with open("hamlet.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# Save tokenizer for app.py
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Prepare sequences
input_sequences = []
for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Define model
model = keras.Sequential([
    keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1),
    keras.layers.LSTM(150),
    keras.layers.Dense(total_words, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model (keep epochs small for quick training on Streamlit Cloud)
model.fit(X, y, epochs=5, verbose=1)

# Save model architecture
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("next_word_lstm.weights.h5")

# Save metadata (vocab size and sequence length)
with open("model_config.pkl", "wb") as f:
    pickle.dump({"total_words": total_words, "max_sequence_len": max_sequence_len}, f)
