import sqlite3
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GOOGLE_API_KEY, MODEL_NAME
from src.state import AgentState

SCHEMA = """
Table: sales
Columns: id (INTEGER), product (TEXT), region (TEXT), amount (REAL)
"""

def generate_sql(question: str) -> str:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    prompt = f"""Given this SQLite schema:
{SCHEMA}

Write ONE SQLite query to answer this question. Return ONLY the raw SQL, no explanation, no markdown fences.

Question: {question}
SQL:"""
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    sql = content.strip()
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE).strip()
    return sql

def sql_node(state: AgentState) -> AgentState:
    sql = generate_sql(state["question"])
    conn = sqlite3.connect("data/sample.db")
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        state["sql_result"] = f"Query: {sql}\nResult: {rows}"
    except Exception as e:
        state["sql_result"] = f"Query failed: {sql}\nError: {e}"
    finally:
        conn.close()
    state["steps"] = state.get("steps", []) + ["sql"]
    return state

if __name__ == "__main__":
    test_state: AgentState = {
        "question": "What is the total sales amount by region?",
        "plan": "",
        "next": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }
    result = sql_node(test_state)
    print(result["sql_result"])
    print("\nSteps so far:", result["steps"])