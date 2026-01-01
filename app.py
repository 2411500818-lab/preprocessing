import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="NLP Twitter", layout="wide")
st.title("🧠 NLP Twitter Analysis")

file = st.file_uploader("Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Jumlah data:", len(df))

    col = st.selectbox("Pilih kolom teks", df.columns)

    def clean_text(text):
        text = str(text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return text.lower()

    if st.button("Proses"):
        df["clean_text"] = df[col].apply(clean_text)
        st.dataframe(df[[col, "clean_text"]].head())
