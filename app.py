import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]  # Remove special characters
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords & punctuation
    y = [ps.stem(i) for i in y]  # Apply stemming

    return " ".join(y)

# Load trained vectorizer & model
with open('vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.header("Spam" if result == 1 else "Not Spam")
    else:
        st.warning("Please enter a message before predicting.")
