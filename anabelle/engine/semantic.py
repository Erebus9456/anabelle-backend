"""Keyword-based emotion short-circuit from SenseVoice transcript text."""

from __future__ import annotations

import re
from re import Pattern

# (pattern, avatar_emotion) — checked in order; first match wins.
SEMANTIC_RULES: tuple[tuple[Pattern[str], str], ...] = (
    # ANGRY - English
    (re.compile(r"\b(so angry|i'?m angry|very angry|furious|pissed off|hate this|mad as hell|irritated|annoyed|frustrated)\b", re.I), "ANGRY"),
    (re.compile(r"\b(disgusting|gross|revolting|that'?s nasty|terrible|horrible|awful)\b", re.I), "ANGRY"),
    (re.compile(r"\b(shut up|stop it|leave me alone|go away|get out)\b", re.I), "ANGRY"),
    (re.compile(r"\b(stupid|idiotic|ridiculous|unbelievable|what the hell)\b", re.I), "ANGRY"),
    # ANGRY - Japanese
    (re.compile(r"(怒|腹立|ムカ|嫌だ|許せない|腹が立つ|信じられない|最低|クソ)"), "ANGRY"),
    # HAPPY - English
    (re.compile(r"\b(that'?s great|so happy|i'?m happy|love this|wonderful|fantastic|awesome|excellent|perfect)\b", re.I), "HAPPY"),
    (re.compile(r"\b(yay|hooray|yes|finally|brilliant|love it|so good)\b", re.I), "HAPPY"),
    (re.compile(r"\b(thank you|thanks|grateful|appreciate)\b", re.I), "HAPPY"),
    (re.compile(r"\b(feel good|feeling great|in a good mood)\b", re.I), "HAPPY"),
    # HAPPY - Japanese
    (re.compile(r"(嬉し|楽し|最高|素晴ら|幸せ|ありがとう|よかった|楽しい)"), "HAPPY"),
    # SAD - English
    (re.compile(r"\b(so sad|i'?m sad|depressed|heartbroken|miserable|so upset|crying|tears)\b", re.I), "SAD"),
    (re.compile(r"\b(scared|afraid|terrified|frightened|so fearful|worried|anxious)\b", re.I), "SAD"),
    (re.compile(r"\b(miss you|lonely|alone|empty)\b", re.I), "SAD"),
    (re.compile(r"\b(sorry|apologize|my fault|regret)\b", re.I), "SAD"),
    # SAD - Japanese
    (re.compile(r"(悲し|寂し|辛い|泣|残念|怖い|不安|ごめん|すみません)"), "SAD"),
    # EXCITED - English
    (re.compile(r"\b(wow|amazing|oh my|incredible|can'?t believe|unbelievable|spectacular)\b", re.I), "EXCITED"),
    (re.compile(r"\b(omg|oh my god|holy cow|no way|seriously)\b", re.I), "EXCITED"),
    (re.compile(r"\b(let'?s go|come on|yes|alright|finally)\b", re.I), "EXCITED"),
    (re.compile(r"\b(can'?t wait|looking forward|so excited)\b", re.I), "EXCITED"),
    # EXCITED - Japanese
    (re.compile(r"(すごい|びっくり|わあ|驚|やった|楽しみ|最高)"), "EXCITED"),
)

TAG_PATTERN = re.compile(r"<\|[^|]+\|>")


def strip_sensevoice_tags(text: str) -> str:
    return TAG_PATTERN.sub("", text).strip()


def match_semantic_emotion(text: str) -> str | None:
    """Return avatar emotion when transcript clearly states affect."""
    clean = strip_sensevoice_tags(text)
    if not clean:
        return None
    for pattern, emotion in SEMANTIC_RULES:
        if pattern.search(clean):
            return emotion
    return None
