
# Twitter Sentiment Analysis WebApp — Pro (Streamlit)

An end-to-end Streamlit app with:
- Sentiment (VADER)
- Emotion (Transformers)
- Toxicity (Detoxify)
- Word clouds, hashtags, emojis
- Time-series & engagement charts
- Topic modeling (BERTopic, optional)
- Works with **snscrape** (no API keys) or **Tweepy**

## Quickstart
1. Create venv
2. `pip install -r requirements.txt`
3. (Optional) copy `.env.example` to `.env` and add Twitter keys
4. `streamlit run app.py`

### Notes
- Heavy features (Emotion/Toxicity/BERTopic) download models on first run.
- Use sidebar to toggle heavy features.
