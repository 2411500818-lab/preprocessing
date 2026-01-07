import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===================== CONFIG =====================
st.set_page_config(page_title="NLP Text Analysis", layout="wide")
st.title("🧠 NLP Text Analysis – Metode Penelitian")

# ===================== SIDEBAR METODE =====================
st.sidebar.title("📌 Metode Penelitian")
st.sidebar.markdown("""
### Preprocessing
 Tokenisasi  
 Stopword Removal  
 Normalisasi  

### Feature Extraction
 TF-IDF  

### Pemodelan
 Naive Bayes  

### Evaluasi Performa
 Accuracy  
""")

# ===================== NORMALISASI & STOPWORD =====================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "bgus": "bagus", "bgs": "bagus",
    "mantul": "mantap", "apk": "aplikasi"
}

stopwords = set([
    "dan","yang","di","ke","dari","ini","itu","saya","aku",
    "kamu","dia","adalah","untuk","dengan","pada","tidak","ya"
])

# ===================== UPLOAD DATA =====================
file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📄 Data Asli")
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    # ===================== PREPROCESSING =====================
    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)

        tokens = text.split()
        tokens = [normalisasi.get(w, w) for w in tokens]
        tokens = [w for w in tokens if w not in stopwords]

        return tokens

    if st.button("🚀 Jalankan Analisis"):
        df["tokens"] = df[text_col].apply(preprocess)
        df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

        # ===================== HASIL PREPROCESSING =====================
        st.subheader("1️⃣ Preprocessing – Hasil Tokenisasi")
        st.dataframe(df[[text_col, "clean_text"]])

        # ===================== TF-IDF MANUAL =====================
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

        # ===================== FEATURE EXTRACTION =====================
        st.subheader("2️⃣ Feature Extraction – TF-IDF")
        st.dataframe(tfidf_df.head(10))

        fig, ax = plt.subplots()
        top10 = tfidf_df.head(10)
        ax.barh(top10["Kata"], top10["Skor_TFIDF"])
        ax.invert_yaxis()
        st.pyplot(fig)

        # ===================== WORDCLOUD =====================
        st.subheader("☁️ WordCloud TF-IDF")
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(tfidf)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # ===================== SENTIMENT LEXICON =====================
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

        # ===================== PEMODELAN NAIVE BAYES =====================
        labels = df["sentiment"].unique()
        label_count = df["sentiment"].value_counts().to_dict()

        word_count = {l: Counter() for l in labels}
        total_words = {l: 0 for l in labels}

        for _, r in df.iterrows():
            for w in r["clean_text"].split():
                word_count[r["sentiment"]][w] += 1
                total_words[r["sentiment"]] += 1

        vocab = set(tfidf_df["Kata"])
        V = len(vocab)
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

        st.subheader("3️⃣ Pemodelan – Naive Bayes")
        st.dataframe(df[[text_col, "sentiment", "nb_prediction"]])

        # ===================== EVALUASI =====================
        acc = (df["sentiment"] == df["nb_prediction"]).mean()

        st.subheader("4️⃣ Evaluasi Performa")
        st.metric("Accuracy", f"{acc*100:.2f}%")

        # ===================== STATISTIK AKHIR =====================
        st.subheader("📊 Statistik Hasil Analisis")
        st.write(f"• Total kata dianalisis: {len(tfidf)}")
        st.write(f"• Skor TF-IDF tertinggi: {tfidf_df['Skor_TFIDF'].max():.4f}")
        st.write(f"• Skor TF-IDF terendah: {tfidf_df['Skor_TFIDF'].min():.4f}")
        st.write(f"• Skor TF-IDF rata-rata: {tfidf_df['Skor_TFIDF'].mean():.4f}")

        # ===================== DOWNLOAD =====================
        st.download_button(
            "⬇️ Download Hasil CSV",
            df.to_csv(index=False),
            "hasil_nlp.csv",
            "text/csv"
        )
