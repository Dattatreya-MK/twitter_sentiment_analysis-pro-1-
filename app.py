# app.py — FINAL STABLE VERSION (Error-free)

import os
import re
from collections import Counter
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from helper.fetch import fetch_tweets as fetch_tweets_df
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

# ---------------------------
# Basic toxicity fallback
# ---------------------------
BASIC_BADWORDS = {
    "fuck", "shit", "bitch", "asshole", "damn",
    "stupid", "fucking", "idiot", "suck"
}

def simple_toxicity_scores(texts):
    scores = []
    for t in texts:
        tokens = re.findall(r"\b[\w']+\b", str(t).lower())
        bad = sum(1 for w in tokens if w in BASIC_BADWORDS)
        score = min(1.0, bad / max(1, len(tokens) ** 0.5))
        scores.append(score)
    return scores


# ---------------------------
# Model loaders
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def load_transformer(model="cardiffnlp/twitter-roberta-base-sentiment"):
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model=model)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_emotion():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    except Exception:
        return None


# ---------------------------
# Utils
# ---------------------------
def compute_eii(score):
    return 50 * (float(score) + 1)

@st.cache_data(ttl=600)
def cached_fetch(query, mode, limit, lang, sample_path):
    return fetch_tweets_df(
        query=query,
        limit=limit,
        mode=mode,
        lang=lang,
        sample_path=sample_path
    )


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙ Settings")

data_source = st.sidebar.radio(
    "Data Source",
    ["Live (Twitter API v2)", "Sample dataset"],
    index=1
)

query = st.sidebar.text_input("Search Query", placeholder="bitcoin, ai, elections")
max_tweets = st.sidebar.slider("Number of Tweets", 10, 500, 100, 10)
lang = st.sidebar.selectbox("Language", ["en", "hi", "kn", "ta", "te"], 0)

st.sidebar.markdown("---")
use_vader = st.sidebar.checkbox("VADER Sentiment", True)
use_transformer = st.sidebar.checkbox("Transformer Sentiment", False)
use_emotion = st.sidebar.checkbox("Emotion Detection", False)
use_toxicity = st.sidebar.checkbox("Toxicity Detection", False)
filter_toxic = st.sidebar.checkbox("Remove Toxic Tweets (≥0.5)", False)

sentiment_weight = st.sidebar.slider("Fusion Weight (VADER ↔ Transformer)", 0.0, 1.0, 0.6)


# ---------------------------
# Main
# ---------------------------
st.title("Twitter Sentiment Analysis System")

if st.button("Fetch & Analyze"):

    mode = "sample" if data_source.startswith("Sample") else "twitter_api"
    sample_path = f"sample_{lang}.json"

    if mode == "twitter_api" and not query.strip():
        st.error("Enter a query for live Twitter analysis.")
        st.stop()

    try:
        with st.spinner("Fetching tweets..."):
            df = cached_fetch(query, mode, max_tweets, lang, sample_path)

        if df.empty:
            st.warning("No tweets found.")
            st.stop()

        st.success(f"Fetched {len(df)} tweets")
        st.dataframe(df.head(50), use_container_width=True)

        df2 = df.copy()
        df2["text"] = df2["text"].astype(str)

        # ---------------------------
        # Analysis
        # ---------------------------
        with st.spinner("Running analysis..."):

            # VADER
            if use_vader:
                vader = load_vader()
                df2["vader"] = df2["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
            else:
                df2["vader"] = 0.0

            # Transformer
            if use_transformer:
                model = load_transformer()
                scores = []
                for t in df2["text"]:
                    try:
                        r = model(t[:512])[0]
                        s = r["score"]
                        if "NEG" in r["label"].upper():
                            s = -s
                    except Exception:
                        s = 0.0
                    scores.append(s)
                df2["transformer"] = scores
            else:
                df2["transformer"] = 0.0

            # Emotion
            if use_emotion:
                emo = load_emotion()
                df2["emotion"] = [
                    emo(t[:512])[0]["label"] if emo else None
                    for t in df2["text"]
                ]

            # Toxicity
            if use_toxicity:
                try:
                    from detoxify import Detoxify
                    tox = Detoxify("original")
                    res = tox.predict(df2["text"].tolist())
                    df2["toxicity"] = pd.Series(res["toxicity"]).astype(float).fillna(0.0)
                except Exception:
                    df2["toxicity"] = pd.Series(
                        simple_toxicity_scores(df2["text"])
                    ).astype(float)
            else:
                df2["toxicity"] = 0.0

            # Filter toxic
            if filter_toxic:
                df2 = df2[df2["toxicity"] < 0.5].reset_index(drop=True)

            # Fused Sentiment (NO None values ever)
            df2["fused_score"] = (
                sentiment_weight * df2["vader"]
                + (1 - sentiment_weight) * df2["transformer"]
            ).astype(float).fillna(0.0)

            df2["fused_label"] = pd.cut(
                df2["fused_score"],
                [-1, -0.05, 0.05, 1],
                labels=["Negative", "Neutral", "Positive"]
            )

            df2["EII"] = df2["fused_score"].apply(compute_eii)

        # ---------------------------
        # Results
        # ---------------------------
        st.markdown("###  Results")
        st.dataframe(df2, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.bar_chart(df2["fused_label"].value_counts())

        if use_emotion:
            with col2:
                st.bar_chart(df2["emotion"].value_counts())

        with col3:
            st.bar_chart(df2["toxicity"].value_counts(bins=5))

        # Keywords
        words = " ".join(df2["text"]).lower()
        tokens = re.findall(r"\b[a-z']{3,}\b", words)
        top = Counter(tokens).most_common(10)

        st.markdown("### 🔠 Top Keywords")
        st.table(pd.DataFrame(top, columns=["Keyword", "Count"]))

        st.download_button(
            "⬇ Download CSV",
            df2.to_csv(index=False),
            "tweet_analysis.csv"
        )

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
