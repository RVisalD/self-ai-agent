"""LangChain tool exposing the existing Paragon retrieval pipeline."""
from langchain.tools import tool


@tool
def search_paragon(question: str) -> str:
    """Search internal Paragon International University documentation.

    MUST use for Paragon University facts, including courses, study plans,
    semesters, admissions, tuition, leadership, and follow-up questions.
    Rewrite follow-ups to a standalone question using prior conversation.
    Returns source excerpts. Only answer Paragon facts supported by these excerpts.
    """
    from .paragon_search import format_context, search

    return format_context(search(question))
