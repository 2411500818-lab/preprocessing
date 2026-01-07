import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===================== CONFIG =====================
st.set_page_config(page_title="NLP Text Analysis", layout="wide")
st.title("🧠 NLP Text Analysis")

# ===================== SIDEBAR METODE =====================
st.sidebar.title("📌 Metode Penelitian")
st.sidebar.markdown("""
### 1️⃣ Preprocessing
✔ Tokenisasi  
✔ Stopword Removal  
✔ Normalisasi  

### 2️⃣ Feature Extraction
✔ TF-IDF  

### 3️⃣ Pemodelan
✔ Naive Bayes  
✔ KNN  

### 4️⃣ Evaluasi Performa
✔ Accuracy  
""")

# ===================== NORMALISASI & STOPWORD =====================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "bgus": "bagus", "mantul": "mantap",
    "apk": "aplikasi"
}

stopwords = set([
    "dan","yang","di","ke","dari","ini","itu","saya","aku","kamu",
    "dia","adalah","untuk","dengan","pada","ya"
])

# ===================== UPLOAD DATA =====================
file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Jumlah data:", len(df))
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    # ===================== PREPROCESSING =====================
    st.subheader("1️⃣ Preprocessing")

    def clean_text(text):
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = text.lower()
        words = text.split()
        words = [normalisasi.get(w, w) for w in words]
        words = [w for w in words if w not in stopwords]
        return " ".join(words)

    if st.button("🚀 Jalankan Analisis"):
        df["clean_text"] = df[text_col].apply(clean_text)

        st.write("📄 Hasil Clean Text")
        st.dataframe(df[[text_col, "clean_text"]].head(10))

        # ===================== TOKENISASI =====================
        st.subheader("Tokenisasi")
        df["tokens"] = df["clean_text"].apply(lambda x: x.split())
        st.dataframe(df["tokens"].head(10))

        # ===================== STATISTIK =====================
        st.subheader("📊 Statistik")
        total_words = sum(len(t) for t in df["tokens"])
        st.write("Total kata:", total_words)

        # ===================== TF-IDF =====================
        st.subheader("2️⃣ Feature Extraction (TF-IDF Manual)")

        docs = df["clean_text"].tolist()
        N = len(docs)

        tf_list = []
        df_count = Counter()

        for doc in docs:
            tf = Counter(doc.split())
            tf_list.append(tf)
            for w in tf:
                df_count[w] += 1

        tfidf = {}
        for tf in tf_list:
            for w, c in tf.items():
                idf = math.log((N + 1) / (df_count[w] + 1)) + 1
                tfidf[w] = tfidf.get(w, 0) + c * idf

        tfidf_df = pd.DataFrame(tfidf.items(), columns=["Kata", "TF-IDF"])
        tfidf_df = tfidf_df.sort_values(by="TF-IDF", ascending=False)

        st.dataframe(tfidf_df.head(10))

        # ===================== GRAFIK TF-IDF =====================
        fig, ax = plt.subplots()
        top10 = tfidf_df.head(10)
        ax.barh(top10["Kata"], top10["TF-IDF"])
        ax.invert_yaxis()
        st.pyplot(fig)

        # ===================== WORDCLOUD =====================
        st.subheader("☁️ WordCloud")
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(tfidf)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # ===================== SENTIMENT =====================
        st.subheader("😊 Analisis Sentimen")

        positive_words = ["bagus","baik","mantap","suka","senang","puas","keren"]
        negative_words = ["buruk","jelek","lambat","error","kecewa","parah"]

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
        st.dataframe(df[[text_col, "sentiment"]])

        # ===================== NAIVE BAYES =====================
        st.subheader("3️⃣ Pemodelan (Naive Bayes Manual)")

        labels = df["sentiment"].unique()
        label_count = df["sentiment"].value_counts().to_dict()

        word_count = {l: Counter() for l in labels}
        total_words_label = {l: 0 for l in labels}

        for _, row in df.iterrows():
            for w in row["clean_text"].split():
                word_count[row["sentiment"]][w] += 1
                total_words_label[row["sentiment"]] += 1

        V = len(tfidf_df)

        def predict_nb(text):
            scores = {}
            for l in labels:
                prob = math.log(label_count[l] / len(df))
                for w in text.split():
                    prob += math.log((word_count[l].get(w, 0) + 1) /
                                     (total_words_label[l] + V))
                scores[l] = prob
            return max(scores, key=scores.get)

        df["NB_Pred"] = df["clean_text"].apply(predict_nb)

        nb_acc = (df["NB_Pred"] == df["sentiment"]).mean() * 100
        st.write("Accuracy Naive Bayes:", f"{nb_acc:.2f}%")

        # ===================== KNN MANUAL =====================
        st.subheader("KNN Manual (k=3)")

        def cosine_sim(a, b):
            return sum(a[w]*b[w] for w in a if w in b) / (
                math.sqrt(sum(v*v for v in a.values())) *
                math.sqrt(sum(v*v for v in b.values())) + 1e-9
            )

        tf_vectors = []
        for doc in docs:
            vec = Counter(doc.split())
            tf_vectors.append(vec)

        def predict_knn(idx, k=3):
            sims = []
            for i, vec in enumerate(tf_vectors):
                if i != idx:
                    sims.append((cosine_sim(tf_vectors[idx], vec), df.iloc[i]["sentiment"]))
            sims.sort(reverse=True)
            top = sims[:k]
            return Counter([s[1] for s in top]).most_common(1)[0][0]

        df["KNN_Pred"] = [predict_knn(i) for i in range(len(df))]
        knn_acc = (df["KNN_Pred"] == df["sentiment"]).mean() * 100
        st.write("Accuracy KNN:", f"{knn_acc:.2f}%")

        # ===================== GRAFIK AKURASI =====================
        st.subheader("4️⃣ Evaluasi Performa")
        fig_acc, ax_acc = plt.subplots()
        ax_acc.bar(["Naive Bayes", "KNN"], [nb_acc, knn_acc])
        ax_acc.set_ylabel("Accuracy (%)")
        st.pyplot(fig_acc)

        # ===================== DOWNLOAD =====================
        st.download_button(
            "⬇️ Download Hasil",
            df.to_csv(index=False),
            "hasil_nlp_final.csv",
            "text/csv"
        )
