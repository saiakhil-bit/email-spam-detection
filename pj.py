import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and prepare the data
#@st.cache
def load_data():
    df = pd.read_csv("data.csv")
    df["Category"] = df["Category"].apply(lambda x: 0 if x == "ham" else 1)
    return df

# Load data and prepare the model
df = load_data()
cv = CountVectorizer()
x = cv.fit_transform(df["Message"])
y = df["Category"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
model = MultinomialNB()
model.fit(x_train, y_train)

# Streamlit application
st.title("Email Spam Detection")

# Input form for new messages
new_message = st.text_area("Enter the message you want to classify:")

if st.button("Predict"):
    if new_message:
        # Transform the input message using the same vectorizer
        x_new = cv.transform([new_message])
        # Predict the category of the new message
        prediction = model.predict(x_new)
        # Display the result
        result = "Spam" if prediction[0] == 1 else "Ham"
        st.write(f"The message is classified as: {result}")
    else:
        st.write("Please enter a message to classify.")
