import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="NLP Text Analysis", layout="wide")
st.title("🧠 NLP Text Analysis – Metode Penelitian")

# ===================== DATA =====================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "bgus": "bagus", "bgs": "bagus",
    "mantul": "mantap", "apk": "aplikasi"
}

stopwords = set([
    "dan","yang","di","ke","dari","ini","itu","saya","aku",
    "kamu","dia","adalah","untuk","dengan","pada","tidak","ya"
])

file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    text_col = st.selectbox("Pilih kolom teks", df.columns)

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [normalisasi.get(w, w) for w in tokens]
        tokens = [w for w in tokens if w not in stopwords]
        return tokens

    if st.button("🚀 Jalankan Analisis"):

        # ===================== PREPROCESSING =====================
        df["tokens"] = df[text_col].apply(preprocess)
        df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

        # ===================== TF-IDF =====================
        docs = df["tokens"].tolist()
        N = len(docs)

        tf_list = []
        df_counter = Counter()

        for doc in docs:
            tf = Counter(doc)
            tf_list.append(tf)
            for w in tf:
                df_counter[w] += 1

        tfidf = {}
        for tf in tf_list:
            for w, c in tf.items():
                idf = math.log((N + 1) / (df_counter[w] + 1)) + 1
                tfidf[w] = tfidf.get(w, 0) + c * idf

        tfidf_df = pd.DataFrame(tfidf.items(), columns=["Kata", "Skor_TFIDF"])
        tfidf_df = tfidf_df.sort_values(by="Skor_TFIDF", ascending=False)

        # ===================== SENTIMENT =====================
        positive = ["bagus","baik","mantap","suka","senang","puas","keren","cepat"]
        negative = ["buruk","jelek","lambat","error","kecewa","parah","lemot"]

        def sentiment(text):
            score = 0
            for w in text.split():
                if w in positive:
                    score += 1
                elif w in negative:
                    score -= 1
            if score > 0:
                return "Positive"
            elif score < 0:
                return "Negative"
            return "Neutral"

        df["sentiment"] = df["clean_text"].apply(sentiment)

        # ===================== NAIVE BAYES =====================
        labels = df["sentiment"].unique()
        label_count = df["sentiment"].value_counts().to_dict()

        word_count = {l: Counter() for l in labels}
        total_words = {l: 0 for l in labels}

        for _, r in df.iterrows():
            for w in r["clean_text"].split():
                word_count[r["sentiment"]][w] += 1
                total_words[r["sentiment"]] += 1

        V = len(tfidf_df)
        total_docs = len(df)

        def predict_nb(text):
            scores = {}
            for l in labels:
                prob = math.log(label_count[l] / total_docs)
                for w in text.split():
                    prob += math.log((word_count[l].get(w, 0) + 1) / (total_words[l] + V))
                scores[l] = prob
            return max(scores, key=scores.get)

        df["nb_prediction"] = df["clean_text"].apply(predict_nb)
        accuracy = (df["sentiment"] == df["nb_prediction"]).mean()

        # ===================== TABS NAVIGASI =====================
        tab1, tab2, tab3, tab4 = st.tabs([
            "1️⃣ Preprocessing",
            "2️⃣ Feature Extraction (TF-IDF)",
            "3️⃣ Pemodelan (Naive Bayes)",
            "4️⃣ Evaluasi Performa"
        ])

        # ---------- TAB 1 ----------
        with tab1:
            st.subheader("Preprocessing")
            st.write("Tokenisasi, Stopword Removal, Normalisasi")
            st.dataframe(df[[text_col, "clean_text"]])

        # ---------- TAB 2 ----------
        with tab2:
            st.subheader("TF-IDF")
            st.dataframe(tfidf_df.head(10))

            fig, ax = plt.subplots()
            top10 = tfidf_df.head(10)
            ax.barh(top10["Kata"], top10["Skor_TFIDF"])
            ax.invert_yaxis()
            st.pyplot(fig)

            wc = WordCloud(width=800, height=400, background_color="white")
            wc.generate_from_frequencies(tfidf)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc)
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        # ---------- TAB 3 ----------
        with tab3:
            st.subheader("Naive Bayes")
            st.dataframe(df[[text_col, "sentiment", "nb_prediction"]])

        # ---------- TAB 4 ----------
        with tab4:
            st.subheader("Evaluasi Performa")
            st.metric("Accuracy", f"{accuracy*100:.2f}%")

            st.write("📊 Statistik TF-IDF")
            st.write(f"• Total kata: {len(tfidf)}")
            st.write(f"• Skor tertinggi: {tfidf_df['Skor_TFIDF'].max():.4f}")
            st.write(f"• Skor terendah: {tfidf_df['Skor_TFIDF'].min():.4f}")
            st.write(f"• Rata-rata: {tfidf_df['Skor_TFIDF'].mean():.4f}")

        st.download_button(
            "⬇️ Download Hasil CSV",
            df.to_csv(index=False),
            "hasil_nlp.csv",
            "text/csv"
        )
