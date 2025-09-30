# Next Word Prediction with LSTM and Streamlit

🔗 **Live App**: [Next Word Prediction Demo](https://lstm-rnn-nextwordprediction-4dssdkutrhoqotngddtqbz.streamlit.app/)

## 📌 Overview
This project demonstrates a **Next Word Prediction** model trained on *Shakespeare’s Hamlet*. It uses an **LSTM neural network** to learn sequences of words and predict the most likely next word given a text input. The app is deployed using **Streamlit Cloud** for an interactive web interface.

## 🚀 Features
- Trains an LSTM model on a text dataset (`hamlet.txt`).
- Uses **Tokenizer + Embedding layer** for text preprocessing.
- Predicts the next word in a sequence.
- Interactive **Streamlit UI** for testing predictions.
- Automatically retrains if model files are missing.

## 🏗 Project Structure
```
├── app.py                   # Streamlit app (loads model + tokenizer, runs predictions)
├── train.py                 # Script to train the LSTM and save model files
├── hamlet.txt               # Training dataset (Shakespeare’s Hamlet)
├── tokenizer.pickle         # Saved tokenizer
├── model_config.pkl         # Contains vocab size & sequence length
├── model_architecture.json  # Saved model architecture
├── next_word_lstm.weights.h5 # Saved model weights
├── requirements.txt         # Python dependencies
```

## ⚙️ Setup Instructions
1. Clone Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```
2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```
3. Train the Model (if needed)
```bash
python train.py
```
This generates `tokenizer.pickle`, `model_config.pkl`, `model_architecture.json`, and `next_word_lstm.weights.h5`.

4. Run Streamlit App
```bash
streamlit run app.py
```

## 📊 Example Usage
- Input: `To be or not to`  
- Output: `Next word: be`

## 🛠 Requirements
Key dependencies:
- `tensorflow` >= 2.15
- `streamlit`
- `numpy`
- `pickle-mixin`

Install with:
```bash
pip install -r requirements.txt
```

## 📡 Deployment
- Deployed on **Streamlit Cloud**
- Auto-runs training if model files are missing

Live demo 👉 [Streamlit App](https://lstm-rnn-nextwordprediction-4dssdkutrhoqotngddtqbz.streamlit.app/)
