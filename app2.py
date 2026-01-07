import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ================= CONFIG =================
st.set_page_config(page_title="NLP Research App", layout="wide")
st.title("🧠 Analisis Teks NLP Berbasis Metode Penelitian")

# ================= NORMALISASI =================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "apk": "aplikasi", "bgus": "bagus"
}

stopwords = {
    "dan","yang","di","ke","dari","ini","itu","saya","aku",
    "kamu","dia","adalah","untuk","dengan","pada","tidak"
}

# ================= UPLOAD DATA =================
file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📄 Data Asli")
    st.write("Jumlah data:", len(df))
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    # ================= 1. PREPROCESSING =================
    st.sidebar.header("1️⃣ Preprocessing")

    casefold = st.sidebar.checkbox("Case Folding", True)
    stopword_removal = st.sidebar.checkbox("Stopword Removal", True)
    normalisasi_kata = st.sidebar.checkbox("Normalisasi Kata", True)

    def preprocessing(text):
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        if casefold:
            text = text.lower()
        tokens = text.split()

        if normalisasi_kata:
            tokens = [normalisasi.get(w, w) for w in tokens]

        if stopword_removal:
            tokens = [w for w in tokens if w not in stopwords]

        return tokens

    if st.button("🚀 Jalankan Analisis"):
        # ================= HASIL PREPROCESSING =================
        df["tokens"] = df[text_col].apply(preprocessing)
        df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

        st.subheader("🔹 Hasil Preprocessing")
        st.dataframe(df[[text_col, "clean_text"]])

        # ================= 2. FEATURE EXTRACTION =================
        st.subheader("2️⃣ Feature Extraction (TF-IDF Manual)")

        docs = df["clean_text"].tolist()
        N = len(docs)

        tf_list = []
        df_counter = Counter()

        for doc in docs:
            tf = Counter(doc.split())
            tf_list.append(tf)
            for w in tf:
                df_counter[w] += 1

        tfidf = {}
        for tf in tf_list:
            for w, c in tf.items():
                idf = math.log((N + 1) / (df_counter[w] + 1)) + 1
                tfidf[w] = tfidf.get(w, 0) + c * idf

        tfidf_df = pd.DataFrame(tfidf.items(), columns=["Kata", "Skor TF-IDF"])
        tfidf_df = tfidf_df.sort_values(by="Skor TF-IDF", ascending=False)

        st.dataframe(tfidf_df.head(10))

        # ================= VISUAL TF-IDF =================
        fig, ax = plt.subplots()
        top10 = tfidf_df.head(10)
        ax.barh(top10["Kata"], top10["Skor TF-IDF"])
        ax.invert_yaxis()
        st.pyplot(fig)

        # ================= WORDCLOUD =================
        st.subheader("☁️ WordCloud")
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(tfidf)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # ================= 3. PEMODELAN =================
        st.subheader("3️⃣ Pemodelan (Naive Bayes Manual)")

        positive = {"bagus","mantap","puas","suka","baik","cepat"}
        negative = {"buruk","jelek","lambat","error","kecewa"}

        def label_sentiment(text):
            score = 0
            for w in text.split():
                if w in positive: score += 1
                if w in negative: score -= 1
            if score > 0: return "Positive"
            if score < 0: return "Negative"
            return "Neutral"

        df["sentiment"] = df["clean_text"].apply(label_sentiment)

        labels = df["sentiment"].unique()
        label_count = df["sentiment"].value_counts().to_dict()

        word_count = {l: Counter() for l in labels}
        total_words = {l: 0 for l in labels}

        for _, r in df.iterrows():
            for w in r["clean_text"].split():
                word_count[r["sentiment"]][w] += 1
                total_words[r["sentiment"]] += 1

        V = len(tfidf)

        def predict_nb(text):
            scores = {}
            for l in labels:
                prob = math.log(label_count[l] / len(df))
                for w in text.split():
                    prob += math.log((word_count[l].get(w, 0) + 1) / (total_words[l] + V))
                scores[l] = prob
            return max(scores, key=scores.get)

        df["prediksi_nb"] = df["clean_text"].apply(predict_nb)

        # ================= 4. EVALUASI =================
        st.subheader("4️⃣ Evaluasi Performa")
        acc = (df["sentiment"] == df["prediksi_nb"]).mean() * 100
        st.metric("Accuracy", f"{acc:.2f}%")

        st.dataframe(df[[text_col, "sentiment", "prediksi_nb"]])

        st.download_button(
            "⬇️ Download Hasil Analisis",
            df.to_csv(index=False),
            "hasil_nlp_penelitian.csv",
            "text/csv"
        )
