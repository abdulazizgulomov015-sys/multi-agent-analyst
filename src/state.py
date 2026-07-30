from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    question: str
    plan: str
    next: str
    documents: List[str]
    sql_result: Optional[str]
    code_result: Optional[str]
    answer: str
    steps: List[str]
    revisions: int