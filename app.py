import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    text = word_tokenize(text)

    y = []
    # Filter out non-alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stem words
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess input message
    transformed_sms = transform_text(input_sms)
    # Vectorize preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    # Ensure that the model is fitted before making predictions
    if hasattr(model, 'predict'):
        # Predict
        result = model.predict(vector_input)[0]
        # Display prediction
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Model is not fitted. Please load a fitted model.")
