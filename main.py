import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('models/RNN_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="centered")

# Page Title and Description
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown(
    """
    Enter a movie review to classify it as positive or negative. This app uses a pre-trained Recurrent Neural Network (RNN) model 
    to analyze the sentiment of the review. The model has been trained on the IMDB dataset.
    """
)

# User input
user_input = st.text_area(
    'Movie Review',
    height=200,
    placeholder="Type your movie review here..."
)

# Classification Button
if st.button('Classify'):
    if user_input:
        # Make predictions
        sentiment, prediction = predict_sentiment(user_input)
        # Display results
        st.subheader('Prediction Result')
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')
        
        # Add an emoji based on sentiment
        if sentiment == 'Positive':
            st.write("😊")
        else:
            st.write("😞")
    else:
        st.warning('Please enter a movie review to classify.')

# Add a footer
st.markdown(
    """
    ---
    Developed by [Veera Prathap](https://your-website.com). 
    For more information, visit [GitHub](https://github.com/veeraprathp/Movie_review_sentiment_prediction).
    """
)
