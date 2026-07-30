from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.retriever import retriever_node
from src.agents.web import web_node
from src.agents.sql_agent import sql_node
from src.agents.code_agent import code_node
from src.agents.supervisor import supervisor_node
from src.agents.writer import writer_node
from src.agents.critic import critic_node
from src.memory import save_interaction

def route_after_supervisor(state: AgentState) -> str:
    return state["next"]

def route_after_critic(state: AgentState) -> str:
    return state["next"]

def memory_save_node(state: AgentState) -> AgentState:
    save_interaction(state["question"], state["answer"])
    return state

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("web", web_node)
    graph.add_node("sql", sql_node)
    graph.add_node("code", code_node)
    graph.add_node("writer", writer_node)
    graph.add_node("memory_save", memory_save_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retriever": "retriever",
            "web": "web",
            "sql": "sql",
            "code": "code",
            "finish": "writer",
        },
    )

    graph.add_edge("retriever", "supervisor")
    graph.add_edge("web", "supervisor")
    graph.add_edge("sql", "supervisor")
    graph.add_edge("code", "supervisor")

    graph.add_edge("writer", "memory_save")
    graph.add_edge("memory_save", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finish": END,
            "retriever": "retriever",
        },
    )

    return graph.compile()

if __name__ == "__main__":
    from langfuse.langchain import CallbackHandler
    from langfuse import get_client
    langfuse_handler = CallbackHandler()

    app = build_graph()
    result = app.invoke({
        "question": "What is the total sales amount in the North region?",
        "plan": "",
        "next": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }, config={"callbacks": [langfuse_handler]})
    print("ANSWER:", result["answer"])
    print("\nSTEPS:", result["steps"])
    print("\nREVISIONS:", result["revisions"])
    get_client().flush()