import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===================== CONFIG =====================
st.set_page_config(page_title="NLP Text Analysis", layout="wide")
st.title("🧠 NLP Text Analysis (TF-IDF, Sentiment, WordCloud, Naive Bayes)")

# ===================== NORMALIZATION DICT =====================
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
file = st.file_uploader("Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Jumlah data:", len(df))
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    st.sidebar.header("⚙️ Preprocessing")
    casefold = st.sidebar.checkbox("Case Folding", True)
    remove_url = st.sidebar.checkbox("Remove URL", True)
    remove_symbol = st.sidebar.checkbox("Remove Symbol", True)
    remove_number = st.sidebar.checkbox("Remove Number", True)
    use_normalisasi = st.sidebar.checkbox("Normalisasi Kata", True)
    remove_stopword = st.sidebar.checkbox("Stopword Removal", True)

    def clean_text(text):
        text = str(text)
        if remove_url:
            text = re.sub(r"http\S+|www\S+", "", text)
        if remove_number:
            text = re.sub(r"\d+", "", text)
        if remove_symbol:
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
        if casefold:
            text = text.lower()
        words = text.split()
        if use_normalisasi:
            words = [normalisasi.get(w, w) for w in words]
        if remove_stopword:
            words = [w for w in words if w not in stopwords]
        return " ".join(words)

    if st.button("🚀 Proses"):
        df["clean_text"] = df[text_col].apply(clean_text)

        # ===================== STATISTIK =====================
        st.subheader("📊 Statistik")
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Data", len(df))
        col2.metric("Total Kata", df["clean_text"].str.split().str.len().sum())
        col3.metric("Rata-rata Panjang Teks", round(df["clean_text"].str.len().mean(), 2))

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

        tfidf = {}
        for tf in tf_list:
            for w, c in tf.items():
                idf = math.log((N + 1) / (df_counter[w] + 1)) + 1
                tfidf[w] = tfidf.get(w, 0) + c * idf

        tfidf_df = pd.DataFrame(tfidf.items(), columns=["Kata", "Skor_TFIDF"])
        tfidf_df = tfidf_df.sort_values(by="Skor_TFIDF", ascending=False)

        st.subheader("🔟 10 Kata Teratas TF-IDF")
        st.dataframe(tfidf_df.head(10))

        fig, ax = plt.subplots()
        top10 = tfidf_df.head(10)
        ax.barh(top10["Kata"], top10["Skor_TFIDF"])
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

        # ===================== SENTIMENT LEXICON =====================
        positive_words = ["bagus","baik","mantap","suka","senang","puas","keren","cepat"]
        negative_words = ["buruk","jelek","lambat","error","kecewa","parah","lemot"]

        def sentiment(text):
            score = 0
            for w in text.split():
                if w in positive_words:
                    score += 1
                if w in negative_words:
                    score -= 1
            if score > 0:
                return "Positive"
            elif score < 0:
                return "Negative"
            return "Neutral"

        df["sentiment"] = df["clean_text"].apply(sentiment)

        st.subheader("😊 Distribusi Sentimen")
        sent_count = df["sentiment"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(sent_count, labels=sent_count.index, autopct="%1.1f%%")
        st.pyplot(fig2)

        # ===================== NAIVE BAYES MANUAL =====================
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

        st.subheader("🤖 Hasil Naive Bayes")
        st.dataframe(df[[text_col, "sentiment", "nb_prediction"]])

        acc = (df["sentiment"] == df["nb_prediction"]).mean() * 100
        st.metric("Akurasi Naive Bayes", f"{acc:.2f}%")

        st.download_button(
            "⬇️ Download Hasil",
            df.to_csv(index=False),
            "hasil_nlp.csv",
            "text/csv"
        )
