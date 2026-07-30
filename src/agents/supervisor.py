from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GOOGLE_API_KEY, MODEL_NAME
from src.state import AgentState

VALID_ROUTES = ["retriever", "web", "sql", "code", "finish"]

def extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content

def supervisor_node(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

    prompt = f"""You are a supervisor routing a question to the right specialist agent.

Question: {state['question']}
Steps already taken: {state.get('steps', [])}
Documents retrieved so far: {len(state.get('documents', []))}
SQL result so far: {state.get('sql_result')}
Code result so far: {state.get('code_result')}

Agents available:
- retriever: searches internal documents/knowledge base
- web: searches the live internet for current info
- sql: queries a sales database (columns: product, region, amount)
- code: writes and runs Python for calculations
- finish: enough information has been gathered, ready to write final answer

Reply with ONLY one word: retriever, web, sql, code, or finish."""

    response = llm.invoke(prompt)
    choice = extract_text(response.content).strip().lower()

    if choice not in VALID_ROUTES:
        choice = "finish"

    state["next"] = choice
    state["plan"] = state.get("plan", "") + f"\nSupervisor chose: {choice}"
    return state

if __name__ == "__main__":
    test_state: AgentState = {
        "question": "What is the total sales amount in the North region?",
        "plan": "",
        "next": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }
    result = supervisor_node(test_state)
    print("Supervisor decided:", result["next"])
    print("Plan log:", result["plan"])