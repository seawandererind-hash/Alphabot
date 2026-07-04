"""Agents 8-10 — News, Social Sentiment, NLP Tone.
These need live API keys (NewsAPI / Reddit / Claude) to be meaningful.
Until keys are wired, they return a neutral HOLD so they don't distort voting.
Interface is identical, so upgrading is a drop-in later."""
from .base import Vote


def news(df, ctx=None):
    return Vote("News & Calendar", "HOLD", 55, "standby — add NewsAPI key")


def sentiment(df, ctx=None):
    return Vote("Social Sentiment", "HOLD", 55, "standby — add Reddit key")


def nlp_tone(df, ctx=None):
    return Vote("NLP Tone", "HOLD", 55, "standby — add Claude key")
