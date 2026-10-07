import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from offline import HashingEmbeddings, ScriptedLLM  # noqa: E402
from multi_agent_system import MultiAgentSystem  # noqa: E402

DOCS = [
    "Benefits of AI in Education\n\nAI tutors adapt lessons to each student and give real-time feedback.",
    "Challenges of AI in Education\n\n1. Privacy and Data Security: Student data protection is paramount\n"
    "2. Digital Divide: Ensuring equitable access to AI-powered tools",
    "Volcano Geology\n\nMagma chambers, tectonic plates and eruption cycles shape volcanic islands.",
]
SOURCES = [{"source": "benefits"}, {"source": "challenges"}, {"source": "volcano"}]


@pytest.fixture
def system():
    s = MultiAgentSystem(llm=ScriptedLLM(), embeddings=HashingEmbeddings())
    s.add_documents_to_rag(DOCS, SOURCES)
    return s
