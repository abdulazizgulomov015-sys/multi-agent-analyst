from tavily import TavilyClient
from src.config import TAVILY_API_KEY
from src.state import AgentState

def web_node(state: AgentState) -> AgentState:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(state["question"], max_results=3)
    results = [r["content"] for r in response["results"]]
    state["documents"] = state.get("documents", []) + results
    state["steps"] = state.get("steps", []) + ["web"]
    return state

if __name__ == "__main__":
    test_state: AgentState = {
        "question": "What is the latest version of Python?",
        "plan": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }
    result = web_node(test_state)
    print("Web results:")
    for d in result["documents"]:
        print("-", d[:150])
    print("\nSteps so far:", result["steps"])