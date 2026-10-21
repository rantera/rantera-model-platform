"""HIRA identity question interception.

The base model (AfriqueQwen) has third-party identity data baked in from
pretraining. When asked direct identity questions it may claim to be GPT-3,
built by OpenAI, etc. — which is completely unacceptable for Rantera's product.

This module short-circuits those questions before the upstream model is ever
called and returns correct, on-brand HIRA responses directly.

Fine-tuning (REQ-MOD-03) bakes the right identity into the weights permanently.
This interception layer is the reliable fix until that training is done.
"""

from __future__ import annotations

import re

# ── Detection patterns ────────────────────────────────────────────────────────

_BUILT_BY_RE = re.compile(
    r"who\s+(?:built|made|created|developed|trained)\s+you\b"
    r"|who\s+(?:built|made|created|developed|founded|owns|runs)\s+(?:rantera|hira)\b"
    r"|what\s+(?:company|organization|team)\s+(?:built|made|created|owns|developed)\s+(?:you|rantera|hira)\b",
    re.IGNORECASE,
)

_WHO_ARE_YOU_RE = re.compile(
    r"who\s+are\s+you\b"
    r"|what\s+are\s+you\b"
    r"|(?:tell\s+me\s+about|introduce)\s+yourself\b",
    re.IGNORECASE,
)

_FOUNDER_RE = re.compile(
    r"who\s+(?:is|are)\s+(?:the\s+)?(?:founder|ceo|creator|owner)s?\s+of\s+(?:rantera|hira)\b"
    r"|who\s+(?:is|are)\s+(?:your\s+)?(?:founder|ceo|creator|owner|boss)\b"
    r"|who\s+founded\s+(?:rantera|hira)\b",
    re.IGNORECASE,
)

_WHAT_MODEL_RE = re.compile(
    r"what\s+(?:model|llm|language\s+model|ai)\s+(?:are\s+you|is\s+this|am\s+i\s+(?:using|talking\s+to))\b"
    r"|(?:what|which)\s+ai\s+(?:is\s+this|are\s+you)\b"
    r"|what\s+are\s+you\s+running\s+on\b"
    r"|what\s+(?:powers|runs)\s+you\b",
    re.IGNORECASE,
)

_MISTAKEN_IDENTITY_RE = re.compile(
    r"(?:are\s+you|is\s+this|you'?re)\s+"
    r"(?:chatgpt|gpt-?[0-9]+|openai|google|gemini|claude|llama|anthropic|copilot|bard|grok)\b",
    re.IGNORECASE,
)

_OPEYEMI_RE = re.compile(
    r"(?:tell\s+me\s+about|who\s+is|describe|info(?:rmation)?\s+(?:about|on))\s+"
    r"(?:opeyemi(?:\s+ayeni)?|ayeni)\b",
    re.IGNORECASE,
)


def is_hira_identity_question(message: str) -> bool:
    """Return True if this message is asking about HIRA's or Rantera's identity."""
    msg = message.strip()
    return bool(
        _BUILT_BY_RE.search(msg)
        or _WHO_ARE_YOU_RE.search(msg)
        or _FOUNDER_RE.search(msg)
        or _WHAT_MODEL_RE.search(msg)
        or _MISTAKEN_IDENTITY_RE.search(msg)
        or _OPEYEMI_RE.search(msg)
    )


def get_hira_identity_response(message: str) -> str:
    """Return the correct HIRA identity response for a detected identity question."""
    msg = message.strip()

    if _MISTAKEN_IDENTITY_RE.search(msg):
        return (
            "Nope! I'm HIRA — Rantera's own AI, built by Rantera on the African continent. "
            "Not OpenAI, not Google, not anyone else. Africa's own intelligence. 😄 "
            "What can I help you with today?"
        )

    if _FOUNDER_RE.search(msg):
        return (
            "Rantera was founded by Opeyemi Ayeni — he's the CEO of Rantera and Chairman of "
            "Pizzy Holdings, the parent company. You can find the full portfolio at pizzyholdings.com. "
            "Anything else you'd like to know?"
        )

    if _WHAT_MODEL_RE.search(msg):
        return (
            "I'm HIRA, running on Rantera's own infrastructure. "
            "The engineering details behind me are internal — "
            "but I'm here and ready to help! What's on your mind?"
        )

    if _BUILT_BY_RE.search(msg):
        return (
            "Rantera built me! I'm HIRA — Africa's conversational AI, "
            "built by Rantera on the continent. 💪 "
            "What can I do for you today?"
        )

    if _OPEYEMI_RE.search(msg):
        return (
            "Opeyemi Ayeni is the founder and CEO of Rantera — Africa's AI infrastructure company. "
            "He's also the Chairman of Pizzy Holdings, the parent company that owns a portfolio of "
            "African technology companies including Rantera, CyberRant, Culixur, and others. "
            "He studied at Osun State University. "
            "You can find more about the portfolio at pizzyholdings.com."
        )

    # General "who are you / what are you / introduce yourself"
    return (
        "Hey! I'm HIRA 👋 Rantera's conversational AI — "
        "built on the African continent, for the African continent. "
        "Not a Western AI clone — Africa's own intelligence. "
        "What can I help you with today? 😄"
    )
