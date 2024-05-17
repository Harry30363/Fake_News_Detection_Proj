import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

true = pd.read_csv('true.csv')
fake = pd.read_csv('fake.csv')
true['label'] = 1  # True DataSet
fake['label'] = 0
news = pd.concat([fake, true], axis=0)
news = news.drop(['title', 'subject', 'date'], axis=1)

news = news.sample(frac=1)
news.reset_index(inplace=True)
news.drop(['index'], axis=1, inplace=True)


def wordopt(text):
    # Convert into lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d', '', text)

    # Remove newline characters
    text = re.sub(r'\n', ' ', text)

    return text


news['text'] = news['text'].apply(wordopt)

x = news['text']  # Dependent variable
y = news['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)
# #
# #
# # DTC = DecisionTreeClassifier()
# # DTC.fit(xv_train, y_train)
# #
# # rfc = RandomForestClassifier()
# # rfc.fit(xv_train, y_train)
#
#
# gbc = GradientBoostingClassifier()
# gbc.fit(xv_train, y_train)
st.set_page_config(layout="wide")
st.write("##")
st.header("Hello Guys :wave: Welcome to Fake News Detection")
st.write("Here you can check whether a news is Fake or True")
st.title('Have a Doubt ?')
input_text = st.text_input('Enter news Article')


def prediction(input_text):
    input_data = vectorization.transform([input_text])
    prediction = LR.predict(input_data)
    return prediction[0]


if input_text:
    pred = prediction(input_text)
    if pred == 0:
        st.write('The News is Fake')
    else:
        st.write('The News is True')

        # To Launch Streamlit Webpage Run this command in the terminal -> streamlit run streamlit.py

