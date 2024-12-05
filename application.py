import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

st.header("Sentiment")
page_description = """Bu model so'zlarni sentimentga ajratadi"""
st.markdown(page_description)

# Matnni kiritish uchun text_input ishlatish
matn = st.text_input("Ingliz tilida soz kiriting:")

# Model va vektorizatorni yuklash
with open("sentiment_model.pkl", "rb") as fl:
    pr = pickle.load(fl)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Tekshirish tugmasi
if st.button("Tekshirish"):
    if matn:  # Foydalanuvchi matn kiritsa
        # Matnni vektor shaklida o'zgartirish
        matn_vektor = vectorizer.transform([matn])
        
        # Modelga yuborish
        sentiment_natija = pr.predict(matn_vektor)
        
        if sentiment_natija == 'negative':
            st.write("Negativ!")
        elif sentiment_natija == 'positive':
            st.write("Positive!")
        else:
            st.write('Neutral!')
    else:
        st.write("Iltimos, matn kiriting.")
