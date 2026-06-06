"""Claude API integration for recommendation explanations.

Set the ANTHROPIC_API_KEY environment variable before use.
If the key is absent the module degrades gracefully — callers receive
empty strings and the UI simply hides the explanation section.
"""

import json
import os
import re
from typing import Optional

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# claude-haiku-4-5: fast and cheap — ideal for short structured outputs
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

_cached_client: Optional["_anthropic.Anthropic"] = None


def is_available() -> bool:
    """Return True if the Anthropic SDK is installed and an API key is set."""
    return HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


def _get_client() -> Optional["_anthropic.Anthropic"]:
    global _cached_client
    if not HAS_ANTHROPIC:
        return None
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    if _cached_client is None:
        _cached_client = _anthropic.Anthropic(api_key=key)
    return _cached_client


def explain_recommendations(
    title: str,
    area: Optional[str],
    abstract: Optional[str],
    journals: list[dict],
) -> list[str]:
    """Generate a one-sentence explanation for each recommended journal.

    Each explanation is specific to the user's article — not a generic
    journal description. Explanations are returned in the same order as
    the input journals list.

    Returns a list of strings (empty string per journal on failure or
    missing API key).

    Args:
        title:    User's article title.
        area:     User's research area / topic (may be None).
        abstract: User's abstract snippet (may be None).
        journals: List of dicts, each with keys: title, subjects,
                  sjr_quartile, apc_usd, is_core, cluster_label.
    """
    client = _get_client()
    if client is None or not journals:
        return [""] * len(journals)

    # Build a compact journal summary for the prompt
    lines = []
    for i, j in enumerate(journals, 1):
        apc_val = float(j.get("apc_usd") or 0)
        apc     = "Free" if apc_val == 0 else f"APC ${apc_val:,.0f}"
        quartile = j.get("sjr_quartile") or "Unranked"
        core    = " | Scopus/WoS" if j.get("is_core") else ""
        # Use subjects if available (direct mode), else cluster_label (API mode)
        scope_raw = j.get("subjects") or j.get("cluster_label") or ""
        # Keep only the first subject line (before the first pipe or semicolon)
        scope = re.split(r"[|;]", str(scope_raw))[0].strip()[:100]
        lines.append(
            f"{i}. \"{j['title']}\" | {scope} | {quartile}{core} | {apc}"
        )

    abstract_snippet = (abstract or "").strip()[:500]

    prompt = f"""You are an expert academic publishing advisor helping a researcher choose the right open-access journal.

ARTICLE TO PUBLISH:
Title: {title}
Research area: {area or "Not specified"}
Abstract: {abstract_snippet or "Not provided"}

TOP RECOMMENDED JOURNALS:
{chr(10).join(lines)}

TASK: For each journal write exactly ONE sentence (20–30 words) explaining specifically why it is a strong match for this particular article. Your explanation must:
- Reference the article's topic or methodology directly
- Mention one concrete journal characteristic (scope, indexing, cost, or prestige) that is relevant
- NOT be a generic statement that could apply to any article

Respond ONLY with a JSON array of {len(journals)} strings in the same order as the journals above.
["Explanation for journal 1.", "Explanation for journal 2.", ...]"""

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Extract JSON array — Claude sometimes wraps it in markdown
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result = [str(x).strip() for x in result]
            # Pad or trim to match journal count
            while len(result) < len(journals):
                result.append("")
            return result[:len(journals)]

    except Exception as exc:
        print(f"LLM explanation error: {exc}")

    return [""] * len(journals)
