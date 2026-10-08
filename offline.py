"""Offline stand-ins for the two things that normally need Ollama.

* ``HashingEmbeddings`` - a real (if crude) bag-of-words embedding model: tokens are
  feature-hashed into a fixed-size vector. FAISS retrieval on top of it is genuine.
* ``ScriptedLLM`` - NOT a language model. It looks at which agent's prompt it received,
  and builds a deterministic JSON reply from the context in that prompt (the retrieved
  passages, the research notes, ...). It exists so the whole pipeline - prompts, LCEL
  chains, JSON parsing, RAG retrieval, the critic -> writer revision loop and the
  Streamlit UI - can be exercised and demonstrated without a GPU or a model download.

Anything produced this way is *scripted output*, not model output. Set
``MULTI_AGENT_OFFLINE=1`` to use it from the CLI demos and the Streamlit app.
"""
import json
import re
import zlib
from typing import Any, List, Optional

import numpy as np
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Feature-hashing bag-of-words embeddings (no model download, deterministic)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _WORD.findall(text.lower()):
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]  # crude plural folding
            h = zlib.crc32(token.encode())
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            vec[(h >> 8) % self.dim] += 0.5
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def _json_objects(text: str) -> List[dict]:
    """All top-level JSON objects embedded in a block of text."""
    decoder, found, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("{", i)
        if i == -1:
            return found
        try:
            obj, end = decoder.raw_decode(text, i)
            found.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1


def _sentences(text: str, limit: int) -> List[str]:
    """Split into sentences; wrapped lines are re-joined, bullets/numbered items stay separate."""
    items: List[str] = []
    fresh = True  # a blank line starts a new paragraph
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            fresh = True
            continue
        if fresh or re.match(r"([-•*]|\d+\.)\s", line) or not items:
            fresh = False
            items.append(re.sub(r"^([-•*]|\d+\.)\s+", "", line))
        else:
            items[-1] += " " + line
    out: List[str] = []
    for item in items:
        out.extend(re.split(r"(?<=[.!?])\s+", item))
    out = [p.strip() for p in out if len(p.strip()) > 25]
    return [p if p[-1] in ".!?" else p.rstrip(":;,") + "." for p in out][:limit]


def _field(prompt: str, label: str, until: Optional[str] = None) -> str:
    start = prompt.find(label)
    if start == -1:
        return ""
    start += len(label)
    end = prompt.find(until, start) if until else -1
    return prompt[start:end if end != -1 else None].strip()


class ScriptedLLM(LLM):
    """Deterministic stand-in for an LLM. See the module docstring - it is not a model."""

    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        if "knowledgeable assistant" in prompt:
            reply = self._rag(prompt)
        elif "research specialist" in prompt:
            reply = self._research(prompt)
        elif "professional writer" in prompt:
            reply = self._write(prompt)
        elif "professional critic" in prompt:
            reply = self._critique(prompt)
        else:
            reply = {"content": "Scripted offline model: unrecognised prompt."}
        return json.dumps(reply)

    # -- one method per agent role ------------------------------------------------
    def _rag(self, prompt: str) -> dict:
        question = _field(prompt, "Question:", "Retrieved Context:")
        context = _field(prompt, "Retrieved Context:", "Please provide")
        facts = _sentences(context, 20)
        return {
            "answer": " ".join(facts) if facts else "No relevant passage was retrieved.",
            "sources_used": ["retrieved passages"] if facts else [],
            "confidence": "medium" if facts else "low",
            "additional_info_needed": "" if facts else f"Documents about: {question}",
        }

    def _research(self, prompt: str) -> dict:
        topic = _field(prompt, "Topic:", "Context:")
        context = _field(prompt, "Context:", "Please provide")
        answers = [o["answer"] for o in _json_objects(context) if "answer" in o]
        facts = _sentences(" ".join(answers) or context, 24)
        return {
            "summary": facts[0] if facts else f"No supporting material was supplied for {topic}.",
            "key_concepts": [topic],
            "main_points": facts[1:5] or facts,
            "perspectives": ["Opportunities", "Risks and open questions"],
            "challenges": [f for f in facts if re.search(r"challeng|risk|privacy|cost|bias|divide", f, re.I)][:3],
            "sources_consulted": "RAG-retrieved passages" if answers else "none",
        }

    def _write(self, prompt: str) -> dict:
        topic = _field(prompt, "Topic:", "Research Data:")
        data = _field(prompt, "Research Data:", "Please write")
        revised = "Reviewer feedback:" in data
        notes = [o for o in _json_objects(data) if "main_points" in o]
        points = notes[0]["main_points"] if notes else []
        challenges = notes[0].get("challenges", []) if notes else []
        article = {
            "title": f"{topic}: What the Evidence Says",
            "introduction": notes[0]["summary"] if notes else f"This article looks at {topic}.",
            "body": " ".join(points) or "No research notes were available.",
            "conclusion": "Taken together, these points suggest a measured, evidence-led approach.",
        }
        if revised:  # the revision round answers the critic's feedback
            article["body"] += " Open challenges: " + ("; ".join(c.rstrip(".") for c in challenges) or "none identified") + "."
            article["conclusion"] += " Concrete next step: pilot with a small group and measure outcomes."
            article["revision_note"] = "Expanded body with challenges and added a concrete next step."
        article["word_count"] = str(len(" ".join(str(v) for v in article.values()).split()))
        return article

    def _critique(self, prompt: str) -> dict:
        content = _field(prompt, "Content to Review:", "Please provide")
        revised = '"revision_note"' in content
        return {
            "overall_assessment": "Solid and well grounded." if revised else "Readable but thin.",
            "strengths": ["Grounded in retrieved material", "Clear structure"],
            "improvements": [] if revised else ["Discuss challenges", "End with a concrete next step"],
            "suggestions": [] if revised else ["Add the open challenges", "Propose a next step"],
            "score": 8.5 if revised else 6.5,
            "final_verdict": "approve" if revised else "revise",
        }


def is_offline() -> bool:
    import os
    return os.getenv("MULTI_AGENT_OFFLINE", "").lower() in {"1", "true", "yes"}
