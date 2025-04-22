import streamlit as st
import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction import text
from sentiment_utils import preprocess

# App title
st.title("🧠 Sentiment Analysis App (Custom Dataset)")

# File uploader
uploaded_file = st.file_uploader("📥 Upload your Excel file with 'Text' and 'Sentiment' columns", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Sample Data")
    st.dataframe(df.head())

    df["Cleaned_Text"] = df["Text"].astype(str).apply(preprocess)
    df["Sentiment"] = df["Sentiment"].map({"Positive": 1, "Negative": 0})

    st.subheader("📊 Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Sentiment", ax=ax1, palette="pastel")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig1)

    st.subheader("📚 Top 10 Frequent Words")
    all_words = ' '.join(df["Cleaned_Text"]).split()
    common_words = Counter(all_words).most_common(10)
    words_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])

    fig2, ax2 = plt.subplots()
    sns.barplot(data=words_df, x="Frequency", y="Word", palette="viridis", ax=ax2)
    st.pyplot(fig2)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Cleaned_Text"], df["Sentiment"], test_size=0.25, random_state=42, stratify=df["Sentiment"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

    st.subheader("🤖 Model Performance")
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for name, clf in models.items():
        pipeline = make_pipeline(vectorizer, clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.markdown(f"**{name} Accuracy:** {round(acc * 100, 2)}%")
        st.text(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=1))

    st.subheader("📝 Try Your Own Text")
    user_input = st.text_area("Enter a sentence to analyze sentiment")

    if user_input:
        model = make_pipeline(vectorizer, LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        prediction = model.predict([preprocess(user_input)])
        label = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        st.markdown(f"**Prediction:** {label}")
