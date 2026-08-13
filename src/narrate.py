"""
Turns an already-computed detection result into one plain-English sentence.
The LLM never decides whether something is defective -- src/detect.py has
already decided that deterministically. This step only phrases the finding,
and its phrasing is checked for basic consistency with the real label before
being trusted, falling back to a template sentence if it contradicts itself.
"""
from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "qwen/qwen3.6-35b-a3b"

_DEFECT_WORDS = ["defect", "anomaly", "damage", "flaw", "scratch", "crack", "irregular", "unusual"]
_OK_WORDS = ["no defect", "no anomaly", "normal", "no issue", "looks fine", "passes inspection", " ok"]


def narrate(category: str, result: dict) -> str:
    """
    Phrases the result as prose. The LLM call is optional polish, not a
    dependency of the core feature -- if it's unavailable or misbehaves for
    any reason (no API key configured yet, provider outage, inconsistent
    phrasing), this always still returns a correct, useful sentence built
    directly from the real computed result.
    """
    prompt = (
        f"You are a factory quality-control assistant. A vision inspection system just checked a photo "
        f"of a {category} for defects. Its finding:\n"
        f"- Label: {result['label']}\n"
        f"- Anomaly score: {result['score']} (threshold for this category: {result['threshold']})\n"
        f"- Most unusual region of the photo: {result['region']}\n\n"
        f"Write ONE short, factual sentence reporting this finding to a QC operator. "
        f"Do not invent details not given above. If the label is 'OK', do not claim any defect was found."
    )
    try:
        response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        text = response.choices[0].message.content.strip()
    except Exception:
        return _fallback_sentence(category, result)

    if not _is_consistent(text, result["label"]):
        text = _fallback_sentence(category, result)
    return text


def _is_consistent(text: str, label: str) -> bool:
    lowered = text.lower()
    mentions_defect = any(w in lowered for w in _DEFECT_WORDS)
    mentions_ok = any(w in lowered for w in _OK_WORDS)
    if label == "OK" and mentions_defect and not mentions_ok:
        return False
    if label == "Defective" and not mentions_defect:
        return False
    return True


def _fallback_sentence(category: str, result: dict) -> str:
    if result["label"] == "OK":
        return f"No defects detected in this {category} -- its appearance matches the normal reference pattern."
    return (
        f"A likely defect was detected in this {category}, localized to the {result['region']} region "
        f"of the photo (anomaly score {result['score']}, threshold {result['threshold']})."
    )
