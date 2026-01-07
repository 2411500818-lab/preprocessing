import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===================== CONFIG =====================
st.set_page_config(page_title="NLP Pipeline", layout="wide")
st.title("🧠 NLP Text Analysis Pipeline")

# ===================== NORMALIZATION =====================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "bgus": "bagus", "mantul": "mantap",
    "bgs": "bagus", "apk": "aplikasi"
}

stopwords = set([
    "dan","yang","di","ke","dari","ini","itu","saya","aku","kamu",
    "dia","adalah","untuk","dengan","pada","tidak","ya"
])

# ===================== UPLOAD =====================
file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Jumlah data:", len(df))
    text_col = st.selectbox("Pilih kolom teks", df.columns)

    # ===================== CLEAN TEXT =====================
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        words = text.split()
        words = [normalisasi.get(w, w) for w in words if w not in stopwords]
        return " ".join(words)

    df["clean_text"] = df[text_col].apply(clean_text)

    # ===================== TABS =====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Preprocessing",
        "2️⃣ Feature Extraction (TF-IDF)",
        "3️⃣ Pemodelan",
        "4️⃣ Evaluasi Performa"
    ])

    # ===================== TAB 1 =====================
    with tab1:
        st.subheader("Hasil Preprocessing")
        st.dataframe(df[[text_col, "clean_text"]])

    # ===================== TF-IDF MANUAL =====================
    docs = df["clean_text"].tolist()
    N = len(docs)

    tf_list = []
    df_counter = Counter()

    for doc in docs:
        tf = Counter(doc.split())
        tf_list.append(tf)
        for w in tf:
            df_counter[w] += 1

    vocab = list(df_counter.keys())
    vocab_index = {w: i for i, w in enumerate(vocab)}

    tfidf_matrix = np.zeros((N, len(vocab)))

    for i, tf in enumerate(tf_list):
        for w, c in tf.items():
            idf = math.log((N + 1) / (df_counter[w] + 1)) + 1
            tfidf_matrix[i, vocab_index[w]] = c * idf

    tfidf_scores = tfidf_matrix.sum(axis=0)
    tfidf_df = pd.DataFrame({
        "Kata": vocab,
        "Skor_TFIDF": tfidf_scores
    }).sort_values(by="Skor_TFIDF", ascending=False)

    # ===================== TAB 2 =====================
    with tab2:
        st.subheader("Top 10 Kata TF-IDF")
        st.dataframe(tfidf_df.head(10))

        fig, ax = plt.subplots()
        top10 = tfidf_df.head(10)
        ax.barh(top10["Kata"], top10["Skor_TFIDF"])
        ax.invert_yaxis()
        st.pyplot(fig)

        st.subheader("WordCloud")
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(
            dict(zip(tfidf_df["Kata"], tfidf_df["Skor_TFIDF"]))
        )
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # ===================== SENTIMENT LABEL =====================
    positive_words = ["bagus","baik","mantap","suka","senang","puas","keren","cepat"]
    negative_words = ["buruk","jelek","lambat","error","kecewa","parah","lemot"]

    def sentiment(text):
        score = 0
        for w in text.split():
            if w in positive_words:
                score += 1
            elif w in negative_words:
                score -= 1
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["clean_text"].apply(sentiment)

    # Encode label
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    y = df["sentiment"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=0.2, random_state=42
    )

    # ===================== TAB 3 =====================
    with tab3:
        st.subheader("Pemodelan")

        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)

        acc_nb = accuracy_score(y_test, nb_pred)
        acc_knn = accuracy_score(y_test, knn_pred)

        st.metric("Accuracy Naive Bayes", f"{acc_nb*100:.2f}%")
        st.metric("Accuracy KNN", f"{acc_knn*100:.2f}%")

    # ===================== TAB 4 =====================
    with
