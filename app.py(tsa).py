import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from helper.fetch import fetch_tweets

def run_once():
    st.title("Twitter Sentiment Analysis — Pro")
    st.write("Fetch tweets via Twitter API v2 and analyze sentiment, emotion, toxicity, and more.")

    # Query input
    query = st.text_input("Enter your query (e.g., 'AI', 'Bitcoin')", "AI")
    max_tweets = st.slider("Number of tweets", min_value=10, max_value=500, value=100, step=10)
    lang = st.selectbox("Language", ["en", "hi", "kn"], index=0)

    if st.button("Fetch Tweets"):
        try:
            with st.spinner("Fetching tweets..."):
                df = fetch_tweets(query=query, limit=max_tweets, mode="twitter_api", lang=lang)
            
            st.success(f"Fetched {len(df)} tweets ✅")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    run_once()
