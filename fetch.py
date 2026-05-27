# fetch.py (root) - lightweight wrapper
from helper.fetch import fetch_tweets as _fetch
import pandas as pd

def fetch_tweets(query, max_results=20, lang="en"):
    """
    Backwards-compatible wrapper used in some places.
    Returns a list of tweet texts (like your older code expected).
    Prefer using helper.fetch.fetch_tweets() which returns DataFrame.
    """
    # call helper.fetch (returns DataFrame)
    df = _fetch(query=query, limit=max_results, mode="twitter_api", lang=lang)
    if isinstance(df, pd.DataFrame):
        return df["text"].astype(str).tolist()
    # if anything else, attempt to convert
    try:
        return [str(r.get("text", "")) for r in df]
    except Exception:
        return []
