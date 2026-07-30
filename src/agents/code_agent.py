import io
import contextlib
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

def generate_code(question: str) -> str:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    prompt = f"""Write Python code that answers this question by printing the result.
Use only the standard library. Return ONLY the raw code, no explanation, no markdown fences.

Question: {question}
Code:"""
    response = llm.invoke(prompt)
    code = extract_text(response.content).strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code

def run_code_safely(code: str) -> str:
    output = io.StringIO()
    safe_builtins = {
        "print": print, "range": range, "len": len, "sum": sum,
        "min": min, "max": max, "sorted": sorted, "abs": abs,
        "round": round, "enumerate": enumerate, "str": str,
        "int": int, "float": float, "list": list, "dict": dict,
    }
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": safe_builtins}, {})
        return output.getvalue().strip()
    except Exception as e:
        return f"Code execution failed: {e}"

def code_node(state: AgentState) -> AgentState:
    code = generate_code(state["question"])
    result = run_code_safely(code)
    state["code_result"] = f"Code:\n{code}\nOutput:\n{result}"
    state["steps"] = state.get("steps", []) + ["code"]
    return state

if __name__ == "__main__":
    test_state: AgentState = {
        "question": "Calculate the sum of squares from 1 to 10",
        "plan": "",
        "next": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }
    result = code_node(test_state)
    print(result["code_result"])
    print("\nSteps so far:", result["steps"])