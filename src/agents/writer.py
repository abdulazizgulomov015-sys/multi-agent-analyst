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

def writer_node(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

    evidence = f"""Documents: {state.get('documents', [])}
SQL result: {state.get('sql_result')}
Code result: {state.get('code_result')}"""

    prompt = f"""Answer the question using ONLY the evidence below. Be specific and direct.

Question: {state['question']}

Evidence:
{evidence}

Answer:"""

    response = llm.invoke(prompt)
    state["answer"] = extract_text(response.content).strip()
    state["steps"] = state.get("steps", []) + ["writer"]
    return state