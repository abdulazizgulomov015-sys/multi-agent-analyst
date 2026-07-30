import gradio as gr
from src.graph import build_graph
from src.state import AgentState

app = build_graph()

def ask(question, history):
    initial_state: AgentState = {
        "question": question,
        "plan": "",
        "next": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }
    result = app.invoke(initial_state)
    answer = result["answer"]
    steps = " → ".join(result["steps"])
    return f"{answer}\n\n_Route: {steps}_"

demo = gr.ChatInterface(
    fn=ask,
    title="Multi-Agent AI Analyst",
    description="Ask about sales data, general knowledge, live web info, or calculations.",
    examples=[
        "What is the total sales amount in the North region?",
        "What is coffee?",
        "Calculate the sum of squares from 1 to 10",
    ],
)

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))