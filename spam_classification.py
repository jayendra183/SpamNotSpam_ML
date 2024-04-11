import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ps = PorterStemmer()


def transform_text(message):
    # 1) converting capital letter to small letter
    message = message.lower()
    # 2) tokenization
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)
    message = y[:]  # cloning the list y
    y.clear()
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return "  ".join(y)


cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title("E-Mail/Sms Classifier")
input_sms = st.text_input("enter the message")
if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = cv.transform([transform_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")

