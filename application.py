import streamlit as st
import pickle

st.header("Sentiment")
page_description = """Bu model so'zlarni sentimentga ajratadi"""
st.markdown(page_description)

model = None
matn = st.number_input("Ingliz tilida soz kiriting", min_value=0.0, max_value=30.0, step=1.0, value=.00)

with open("sentiment_model.pkl","rb") as fl:
    pr = pickle.load(fl)

if st.button("Tekshirish "):
    sentiment_natija = pr.predict([[matn]])
    if sentiment_natija == 'negative':
        st.write("Negativ!")
    elif sentiment_natija == 'positive':
        st.write("Positive! ")
    else:
        st.write('Neutral!')
