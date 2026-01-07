import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import Word2Vec

# ===================== CONFIG =====================
st.set_page_config(page_title="NLP Twitter Analysis", layout="wide")
st.title("🧠 Analisis NLP Teks (Preprocessing, TF-IDF, WordCloud, Sentiment, Naive Bayes)")

# ===================== NORMALISASI & STOPWORD =====================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "bgus": "bagus", "mantul": "mantap",
    "bgs": "bagus", "apk": "aplikasi"
}

stopwords = set([
    "dan","yang","di","ke","dari","ini","itu","saya","aku","kamu",
    "dia","adalah","untuk","dengan","pada","tidak","ya"
])

positive_words = ["bagus","baik","mantap","suka","senang","puas","keren","cepat"]
negative_words = ["buruk","jelek","lambat","error","kecewa","parah","lemot"]

# ===================== UPLOAD DATA =====================
file = st.file_uploader("📤 Upload file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📄 Data Awal")
    st.write("Jumlah data:", len(df))
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    # ===================== PREPROCESSING OPTIONS =====================
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
        tokens = text.split()
        if use_normalisasi:
            tokens = [normalisasi.get(t, t) for t in tokens]
        if remove_stopword:
            tokens = [t for t in tokens if t not in stopwords]
        return tokens

    if st.button("🚀 Jalankan Analisis"):

        # ===================== CASE FOLDING =====================
        st.header("1️⃣ Case Folding")
        df["case_folding"] = df[text_col].str.lower()
        st.dataframe(df[[text_col, "case_folding"]].head())

        # ===================== TOKENISASI =====================
        st.header("2️⃣ Tokenisasi")
        df["token"] = df["case_folding"].apply(lambda x: str(x).split())
        st.write(df["token"].head())

        # ===================== NORMALISASI + STOPWORD =====================
        st.header("3️⃣ Normalisasi & Stopword Removal")
        df["clean_token"] = df[text_col].apply(clean_text)
        st.write(df["clean_token"].head())

        # ===================== CLEAN TEXT =====================
        st.header("🧹 Clean Text Final")
        df["clean_text"] = df["clean_token"].apply(lambda x: " ".join(x))
        st.dataframe(df[[text_col, "clean_text"]].head())

        # ===================== STATISTIK DATA =====================
        st.header("📊 Statistik Data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Data", len(df))
        c2.metric("Total Kata", sum(len(x) for x in df["clean_token"]))
        c3.metric("Rata-rata Panjang Teks", round(df["clean_text"].str.len().mean(), 2))

        # ===================== TF-IDF MANUAL =====================
        st.header("📐 Feature Extraction: TF-IDF")

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
        ax.barh(tfidf_df.head(10)["Kata"], tfidf_df.head(10)["Skor_TFIDF"])
        ax.invert_yaxis()
        st.pyplot(fig)

        # ===================== WORDCLOUD =====================
        st.header("☁️ WordCloud")
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(tfidf)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # ===================== WORD EMBEDDINGS =====================
        st.header("📐 Word Embeddings (Word2Vec)")
        sentences = df["clean_token"].tolist()
        w2v = Word2Vec(sentences, vector_size=50, window=5, min_count=1)
        st.write("Contoh vektor kata:")
        st.write(w2v.wv[sentences[0][0]])

        # ===================== SENTIMENT ANALYSIS =====================
        st.header("😊 Analisis Sentimen")

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

        sent_count = df["sentiment"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(sent_count, labels=sent_count.index, autopct="%1.1f%%")
        st.pyplot(fig2)

        # ===================== NAIVE BAYES MANUAL =====================
        st.header("🤖 Pemodelan: Naive Bayes")

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

        # ===================== EVALUASI MODEL =====================
        st.header("📊 Evaluasi Model")

        accuracy = (df["sentiment"] == df["nb_prediction"]).mean()

        TP = sum((df["sentiment"]=="Positive") & (df["nb_prediction"]=="Positive"))
        FP = sum((df["sentiment"]!="Positive") & (df["nb_prediction"]=="Positive"))
        FN = sum((df["sentiment"]=="Positive") & (df["nb_prediction"]!="Positive"))

        precision = TP / (TP + FP) if TP+FP else 0
        recall = TP / (TP + FN) if TP+FN else 0
        f1 = 2 * precision * recall / (precision + recall) if precision+recall else 0

        st.metric("Accuracy", f"{accuracy*100:.2f}%")
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1-Score", f"{f1:.2f}")

        # ===================== STATISTIK TF-IDF =====================
        st.header("📊 Statistik Hasil TF-IDF")
        st.write(f"• Total kata yang dianalisis: {len(tfidf_df)}")
        st.write(f"• Skor TF-IDF tertinggi: {tfidf_df['Skor_TFIDF'].max():.4f}")
        st.write(f"• Skor TF-IDF terendah: {tfidf_df['Skor_TFIDF'].min():.4f}")
        st.write(f"• Skor TF-IDF rata-rata: {tfidf_df['Skor_TFIDF'].mean():.4f}")

        # ===================== DOWNLOAD =====================
        st.download_button(
            "⬇️ Download Hasil Analisis",
            df.to_csv(index=False),
            "hasil_nlp_lengkap.csv",
            "text/csv"
        )
