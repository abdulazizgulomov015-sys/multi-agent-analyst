from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GOOGLE_API_KEY, MODEL_NAME
from src.state import AgentState

def extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content

def critic_node(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

    evidence = f"""Documents: {state.get('documents', [])}
SQL result: {state.get('sql_result')}
Code result: {state.get('code_result')}"""

    prompt = f"""You are a strict critic reviewing an answer before it ships.

Question: {state['question']}
Proposed answer: {state.get('answer', '(no answer written yet)')}

Evidence gathered:
{evidence}

Is the answer well-supported by the evidence, specific, and directly responsive to the question?
Reply with ONLY one word: PASS or RETRY."""

    response = llm.invoke(prompt)
    verdict = extract_text(response.content).strip().upper()

    if "PASS" in verdict:
        state["next"] = "finish"
    else:
        state["revisions"] = state.get("revisions", 0) + 1
        if state["revisions"] >= 3:
            state["next"] = "finish"
        else:
            state["next"] = "retriever"

    state["steps"] = state.get("steps", []) + ["critic"]
    return state

if __name__ == "__main__":
    test_state: AgentState = {
        "question": "What is the total sales amount in the North region?",
        "plan": "",
        "next": "",
        "documents": [],
        "sql_result": "Query: SELECT SUM(amount) FROM sales WHERE region='North'\nResult: [(2190.75,)]",
        "code_result": None,
        "answer": "The total sales in the North region is $2190.75.",
        "steps": [],
        "revisions": 0,
    }
    result = critic_node(test_state)
    print("Critic verdict → next:", result["next"])
    print("Revisions:", result["revisions"])