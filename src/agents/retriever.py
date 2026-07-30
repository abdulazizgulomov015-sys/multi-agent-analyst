from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import GOOGLE_API_KEY
from src.state import AgentState

def get_vectorstore(collection: str = "docs"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=768,
    )
    client = QdrantClient(path="./data/qdrant_db")
    return QdrantVectorStore(
        client=client, collection_name=collection, embedding=embeddings
    )

def retriever_node(state: AgentState) -> AgentState:
    store = get_vectorstore()
    results = store.similarity_search(state["question"], k=3)
    state["documents"] = [r.page_content for r in results]
    state["steps"] = state.get("steps", []) + ["retriever"]
    return state

if __name__ == "__main__":
    test_state: AgentState = {
        "question": "What is coffee?",
        "plan": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }
    result = retriever_node(test_state)
    print("Retrieved documents:")
    for d in result["documents"]:
        print("-", d[:150])
    print("\nSteps so far:", result["steps"])