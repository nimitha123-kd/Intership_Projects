# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install  streamlit pandas numpy seaborn matplotlib scikit-learn
import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("Email Spam Classifier")

# Load dataset
df = pd.read_csv("email.csv")

# Show columns (to debug)
st.write("Columns in dataset:", df.columns)

# Drop missing values
df = df.dropna()

# Change these based on your dataset
X_text = df['Message']   # first column = text
y = df['Category']       # second column = label

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Train
model = MultinomialNB()
model.fit(X, y)

# Input
user_input = st.text_input("Enter text:")

if user_input:
    test = vectorizer.transform([user_input])
    prediction = model.predict(test)
    st.write("Prediction:", prediction[0])

