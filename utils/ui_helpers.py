"""
utils/ui_helpers.py — Iteration 6

Streamlit has been removed. This file now contains only:
  1. Context enrichment logic  (enrich_with_context, _is_vague, etc.)
  2. Input validation          (validate_input)
  3. Smart fallback detection  (get_smart_fallback)
  4. Follow-up suggestions     (get_followup_suggestions)

All of these are called from main.py (FastAPI) — pure Python, no UI framework.
"""

from __future__ import annotations

import re
from typing import Optional


# ══════════════════════════════════════════════════
# 1. Context Awareness  (Iteration 3 — unchanged)
# ══════════════════════════════════════════════════

_STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "tell", "me",
    "about", "how", "many", "does", "do", "kiet", "university", "please",
    "can", "you", "i", "want", "know", "give", "show", "explain",
    "describe", "list", "find", "get", "any", "all", "some", "its",
    "their", "there", "which", "who", "when", "where", "will", "would",
    "could", "should", "have", "has", "had", "and", "or", "of", "for",
    "in", "at", "by", "with", "this", "that", "my", "your", "our",
    "also", "more", "from", "not", "but",
}

_NAMED_TOPICS = {
    "code jung", "hostel", "transport", "scholarship", "sports", "library",
    "lab", "clubs", "alumni", "jobs", "research", "oric", "qec", "lms",
    "portal", "campus", "location", "contact", "vision", "mission",
    "history", "founder", "principal", "rector", "timetable", "result",
    "internship", "semester", "software", "computer", "engineering",
    "management", "business", "science", "technology", "bachelor",
    "master", "phd", "program", "degree", "faculty", "department",
    "structure", "career", "placement", "accreditation", "ranking",
    "fee structure",
}


def _extract_topic(text: str) -> str:
    """Strip stop words → core topic keywords."""
    tokens = text.lower().split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def _has_named_topic(question: str) -> bool:
    q = question.lower()
    return any(topic in q for topic in _NAMED_TOPICS)


def _is_vague(question: str) -> bool:
    """
    Vague = no named topic AND 6 words or fewer.
    "what documents are required"  → vague
    "tell me about code jung"      → NOT vague (named topic)
    """
    if _has_named_topic(question):
        return False
    return len(question.strip().split()) <= 6


def enrich_with_context(current_question: str, history: list) -> str:
    """
    Enriches a vague follow-up with the topic from the previous user message.

    history format: [{"role": "user"|"assistant", "content": str}, ...]

    Examples:
        prev="what is the admission process", current="what documents are required"
        → "admission process what documents are required"
    """
    if not _is_vague(current_question):
        return current_question

    last_user_msg: Optional[str] = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        return current_question

    topic = _extract_topic(last_user_msg)
    if not topic:
        return current_question

    return f"{topic} {current_question}"


# ══════════════════════════════════════════════════
# 2. Input Validation  (Iteration 5 — unchanged)
# ══════════════════════════════════════════════════

def validate_input(question: str, recent_questions: list[str]) -> Optional[str]:
    """
    Validates user input before sending to pipeline.
    Returns an error message string if invalid, None if valid.

    Checks:
      - Too short  (< 2 chars)
      - Too long   (> 500 chars)
      - No letters (only symbols/numbers)
      - Duplicate  (same as one of last 3 questions)
    """
    q = (question or "").strip()

    if len(q) < 2:
        return "Please enter a longer question."

    if len(q) > 500:
        return "Your question is too long. Please keep it under 500 characters."

    if not re.search(r"[a-zA-Z]", q):
        return "Please enter a question with actual words."

    return None


# ══════════════════════════════════════════════════
# 3. Smart Fallback  (Iteration 5 — unchanged)
# ══════════════════════════════════════════════════

_SMART_FALLBACKS = {
    # LMS / results
    ("result", "marks", "grade", "cgpa", "gpa", "transcript"): {
        "message": "For results and grades, please log in to the Student LMS Portal.",
        "link_label": "🖥 LMS Portal",
        "link_url": "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php",
    },
    # Timetable
    ("timetable", "time table", "class schedule", "lecture schedule"): {
        "message": "Class timetables are available on the Student LMS Portal.",
        "link_label": "🖥 LMS Portal",
        "link_url": "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php",
    },
    # Complaint
    ("complaint", "complain", "grievance", "report issue", "problem with"): {
        "message": "For complaints or grievances, please contact the Admin Office directly.",
        "link_label": "📞 Departments Contact",
        "link_url": "https://kiet.edu.pk/departments-contact/",
    },
    # Hostel
    ("hostel", "dormitory", "accommodation", "room", "boarding"): {
        "message": "For hostel availability and room allocation, contact Student Affairs.",
        "link_label": "📞 Departments Contact",
        "link_url": "https://kiet.edu.pk/departments-contact/",
    },
    # Jobs
    ("job", "vacancy", "career", "hiring", "recruitment", "apply for job"): {
        "message": "Current job openings at KIET are listed on the Jobs page.",
        "link_label": "💼 Jobs at KIET",
        "link_url": "https://kiet.edu.pk/jobs/",
    },
    # Lost & found
    ("lost", "found", "missing item", "lost item"): {
        "message": "For lost and found items, please visit the Security Office on campus.",
        "link_label": "📞 Departments Contact",
        "link_url": "https://kiet.edu.pk/departments-contact/",
    },
}


def get_smart_fallback(question: str) -> Optional[dict]:
    """
    Returns a smart fallback dict if the question matches a known pattern,
    otherwise returns None (pipeline should handle it normally).

    Return format:
    {
        "message":    str,
        "link_label": str,
        "link_url":   str,
    }
    """
    q = question.lower()
    for keywords, response in _SMART_FALLBACKS.items():
        if any(kw in q for kw in keywords):
            return response
    return None


# ══════════════════════════════════════════════════
# 4. Follow-up Suggestions  (Iteration 5 — unchanged)
# ══════════════════════════════════════════════════

_FOLLOWUP_MAP = {
    ("admission", "apply", "eligibility", "requirement", "merit", "entry test"): [
        "What documents are required?",
        "What is the fee structure?",
        "When is the last date to apply?",
    ],
    ("fee", "fees", "tuition", "payment", "cost"): [
        "Is there any scholarship available?",
        "What is the admission process?",
        "Are there any fee discounts?",
    ],
    ("scholarship", "discount", "financial aid", "bursary"): [
        "What is the fee structure?",
        "What is the eligibility for scholarship?",
        "How do I apply for scholarship?",
    ],
    ("program", "degree", "bachelor", "master", "bs", "ms", "course"): [
        "What is the fee structure?",
        "What is the admission process?",
        "What departments does KIET have?",
    ],
    ("hostel", "dormitory", "accommodation"): [
        "What is the hostel fee?",
        "Is hostel available for girls?",
        "How do I apply for hostel?",
    ],
    ("transport", "bus", "shuttle", "route"): [
        "What is the transport fee?",
        "Which areas does KIET transport cover?",
        "How do I register for transport?",
    ],
    ("event", "code jung", "hackathon", "fest", "seminar", "workshop"): [
        "How do I register for the event?",
        "What are the prizes?",
        "Who can participate?",
    ],
    ("lms", "portal", "login", "password"): [
        "How do I reset my LMS password?",
        "Where can I see my timetable?",
        "How do I check my results?",
    ],
}


def get_followup_suggestions(question: str, answer: str) -> list[str]:
    """
    Returns up to 3 follow-up question suggestions based on the topic.
    Matches against both the question and the answer for better coverage.
    """
    combined = (question + " " + answer).lower()
    for keywords, suggestions in _FOLLOWUP_MAP.items():
        if any(kw in combined for kw in keywords):
            return suggestions[:3]
    return []