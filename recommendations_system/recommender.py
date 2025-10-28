from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv; load_dotenv()

import google.genai as genai

from .rag_index import ensure_built_index


DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class UserContext:
    age_years: Optional[float]
    sex: Optional[str]
    bmi: Optional[float]
    weight_kg: Optional[float]
    height_cm: Optional[float]
    bio_age_years: Optional[float]
    bai_z: Optional[float]


@dataclass
class ConditionRisk:
    name: str  # e.g., "diabetes", "hypertension"
    probability: float  # 0..1
    top_risk_factors: List[str]  # human-readable features increasing risk


@dataclass
class Recommendation:
    title: str
    description: str
    actions: List[str]
    citations: List[Dict[str, Any]]  # {source_path, page_start, page_end}


def _client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def _generate_call(client: genai.Client, model: str, prompt: str) -> Any:
    """Compatibility wrapper for generate_content parameter naming across SDK versions."""
    try:
        return client.models.generate_content(model=model, contents=prompt)
    except TypeError:
        try:
            return client.models.generate_content(model=model, input=prompt)
        except TypeError:
            return client.models.generate_content(model=model, content=prompt)


def _format_context_prompt(
    user: UserContext,
    risks: List[ConditionRisk],
    guideline_snippets: List[Dict[str, Any]],
) -> str:
    def _fmt_num(x: Optional[float]) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return ""
        return f"{x:.2f}"

    risk_lines = []
    for r in risks:
        lines = [f"- Condition: {r.name}", f"  Probability: {r.probability:.3f}"]
        if r.top_risk_factors:
            lines.append("  Key risk factors:")
            for f in r.top_risk_factors:
                lines.append(f"    - {f}")
        risk_lines.append("\n".join(lines))

    cites = []
    for i, s in enumerate(guideline_snippets, start=1):
        cites.append(
            f"[Doc {i}] {Path(s['source_path']).name} pages {s['page_start']}-{s['page_end']} (score={s['score']:.3f})\n---\n{s['text'][:3000]}"
        )

    p_cites = "\n\n".join(cites)
    p_risk_lines = "\n".join(risk_lines)
    prompt = f"""
You are a clinical recommendation assistant. Using only the provided Saudi Ministry of Health and WHO guideline excerpts, generate concise, patient-friendly recommendations. Tailor to the person's age, sex, BMI, biological age, and risks. Only recommend items supported in the snippets. Prefer lifestyle interventions first. Provide actionable steps and cite source snippets.

Patient profile:
- Age: {_fmt_num(user.age_years)}
- Sex: {user.sex or ''}
- BMI: {_fmt_num(user.bmi)}
- Weight (kg): {_fmt_num(user.weight_kg)}
- Height (cm): {_fmt_num(user.height_cm)}
- Biological age: {_fmt_num(user.bio_age_years)}
- BAI (z-score): {_fmt_num(user.bai_z)}

Risks and factors:
{p_risk_lines}

Guideline evidence (verbatim excerpts; do not hallucinate beyond these):
{p_cites}

Output JSON with this schema:
{{
  "recommendations": [
    {{
      "title": "string",
      "description": "1-2 sentences referencing guideline-backed rationale",
      "actions": ["action 1", "action 2", ...],
      "citations": [{{"source": "filename.pdf", "page_start": int, "page_end": int}}]
    }},
    ...
  ]
}}
Ensure the JSON is valid and parsable. Be specific (targets, ranges, frequencies). If data is insufficient, say so in description but still ground in provided text.
"""
    return prompt


def _extract_text_from_generation(resp: Any) -> str:
    # Try common shapes from google-genai
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        cand = resp.candidates[0]
        # Some SDKs keep parts under content.parts
        if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
            part = cand.content.parts[0]
            if hasattr(part, "text") and part.text:
                return part.text
        # Or direct text on candidate
        if hasattr(cand, "text") and cand.text:
            return cand.text
    # Dict fallbacks
    if isinstance(resp, dict):
        if "text" in resp and resp["text"]:
            return str(resp["text"])  # type: ignore[index]
        cands = resp.get("candidates") or []
        if cands:
            c0 = cands[0]
            if isinstance(c0, dict):
                parts = (((c0.get("content") or {}).get("parts")) or [])
                if parts:
                    p0 = parts[0]
                    if isinstance(p0, dict) and p0.get("text"):
                        return str(p0.get("text"))
                if c0.get("text"):
                    return str(c0.get("text"))
    return ""


def generate_recommendations(
    user: UserContext,
    risks: List[ConditionRisk],
    top_k: int = 6,
    model: str = DEFAULT_MODEL,
) -> List[Recommendation]:
    idx = ensure_built_index()

    # Build a focused query combining conditions and risk factors
    conditions = ", ".join(sorted({r.name for r in risks})) or "cardiometabolic prevention"
    factors = "; ".join([f for r in risks for f in r.top_risk_factors][:6])
    query = f"Guideline-based lifestyle and clinical recommendations for {conditions}. Risk factors: {factors}. Include thresholds for BMI/weight/blood pressure/glucose if present."

    evidence = idx.search(query, top_k=top_k)
    prompt = _format_context_prompt(user, risks, evidence)

    client = _client()
    resp = _generate_call(client, model, prompt)
    text = _extract_text_from_generation(resp)
    if not text:
        return []

    import json
    parsed: Optional[dict] = None
    # Try direct JSON
    try:
        parsed = json.loads(text.strip())
    except Exception:
        # Attempt to extract fenced JSON
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p_stripped = p.strip()
                # Remove json/JSON prefix if present
                if p_stripped.startswith("json"):
                    p_stripped = p_stripped[4:].strip()
                elif p_stripped.startswith("JSON"):
                    p_stripped = p_stripped[4:].strip()
                if p_stripped.startswith("{") and p_stripped.endswith("}"):
                    try:
                        parsed = json.loads(p_stripped)
                        break
                    except Exception:
                        continue
    if not parsed or "recommendations" not in parsed:
        return []

    out: List[Recommendation] = []
    for rec in parsed.get("recommendations", []):
        title = str(rec.get("title", "Recommendation"))
        desc = str(rec.get("description", ""))
        actions = [str(a) for a in rec.get("actions", [])]
        citations_in = rec.get("citations", [])
        citations: List[Dict[str, Any]] = []
        for c in citations_in:
            try:
                citations.append(
                    {
                        "source": str(c.get("source", "")),
                        "page_start": int(c.get("page_start", 0)),
                        "page_end": int(c.get("page_end", 0)),
                    }
                )
            except Exception:
                continue
        out.append(Recommendation(title=title, description=desc, actions=actions, citations=citations))
    return out


__all__ = [
    "UserContext",
    "ConditionRisk",
    "Recommendation",
    "generate_recommendations",
]


