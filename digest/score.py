"""Score paper relevance using Claude Haiku."""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from config import HAIKU_MODEL, SCORE_THRESHOLD

SYSTEM_PROMPT = """\
You are assisting a computational/theoretical chemistry researcher. Your job is
to score how relevant an arXiv paper is to their research interests.

The researcher is primarily interested in theoretical, computational, and
mechanistic studies involving spin physics in molecular systems, especially when
connected to electronic structure, photophysics, or quantum information.

MAIN RESEARCH INTERESTS

MOLECULAR QUBITS AND SPIN PROPERTIES:
- molecular qubits
- spin Hamiltonians
- spin decoherence
- ODMR
- diradicals and open-shell molecular systems

SPIN-DEPENDENT ELECTRONIC STRUCTURE AND EXCITED-STATE PROCESSES:
- spin–orbit coupling
- intersystem crossing
- spin dynamics
- singlet–triplet mixing
- excited-state spin phenomena in molecules

RADICAL PAIRS, MAGNETIC FIELD EFFECTS, AND CHIRAL/SPIN CHEMISTRY:
- radical pair mechanism
- magnetic field effects
- chiral symmetry in spin-dependent processes
- spin-selective photochemistry and related mechanisms

AUTHORS OF SPECIAL INTEREST:
Papers by or closely connected to the work of these researchers should receive a
significant relevance boost:
- Michael Wasielewski
- Francesco Di Maiolo
"""

USER_TEMPLATE = """\
Paper title: {title}
Abstract: {abstract}

Return a JSON object with:
- "score": integer 0–10
- "topics": list of matched topic keywords (e.g. ["MACE", "DFT", "many-body potential"])
- "reason": one sentence explaining the score
- "novel_method": true if the paper introduces a new design principle for qubits, long decoherence times,
new reactivity based on magnetic field effects, or different qubit addressability and read-out capabilities.

Return only valid JSON. No markdown.\
"""


def _score_paper(client: anthropic.Anthropic, paper: dict) -> dict:
    """Call Claude Haiku to score a single paper. Returns paper with score fields added."""
    user_msg = USER_TEMPLATE.format(
        title=paper["title"],
        abstract=paper["abstract"] or "(no abstract)",
    )
    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
    except Exception as e:
        print(f"  [score] Error scoring '{paper['title'][:60]}': {e}")
        result = {"score": 0, "topics": [], "reason": "scoring error", "dynamic_boundary": False}

    base_score = int(result.get("score", 0))
    novel_method = bool(result.get("novel_method", False))
    topics = result.get("topics", [])

    # Apply bonuses
    bonus = 0
    if novel_method:
        bonus += 2
    if len(topics) >= 2:
        bonus += 2
    final_score = min(10, base_score + bonus)

    return {
        **paper,
        "score": final_score,
        "base_score": base_score,
        "topics": topics,
        "reason": result.get("reason", ""),
        "novel_method": novel_method,
    }


def score_papers(papers: list[dict]) -> list[dict]:
    """Score all papers in parallel batches of 10. Returns only papers >= threshold."""
    if not papers:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=api_key)

    BATCH_SIZE = 5
    scored: list[dict] = []

    for batch_start in range(0, len(papers), BATCH_SIZE):
        batch = papers[batch_start: batch_start + BATCH_SIZE]
        print(f"  Scoring papers {batch_start + 1}–{batch_start + len(batch)} of {len(papers)}...")
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {executor.submit(_score_paper, client, p): p for p in batch}
            for future in as_completed(futures):
                scored.append(future.result())
        # Pause between batches to stay under API rate limit (50 req/min)
        if batch_start + BATCH_SIZE < len(papers):
            time.sleep(8)

    # Print all scores for transparency
    scored.sort(key=lambda p: p["score"], reverse=True)
    for p in scored:
        status = "✓" if p["score"] >= SCORE_THRESHOLD else "✗"
        print(f"  {status} [{p['score']:2d}] {p['title'][:70]}")
        print(f"         reason: {p.get('reason', '')}")

    # Filter
    relevant = [p for p in scored if p["score"] >= SCORE_THRESHOLD]
    print(f"  {len(relevant)} papers scored >= {SCORE_THRESHOLD} (out of {len(scored)} scored)")
    return relevant
