"""LangGraph 실습에서 공유하는 State 정의."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    user_input: str
    input_type: str
    analysis: dict
    final_answer: str
    error: str | None
