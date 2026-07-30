from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.config import GOOGLE_API_KEY

def build_vectorstore(doc_paths: list[str], collection: str = "docs"):
    docs = []
    for p in doc_paths:
        docs.extend(TextLoader(p).load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    ).split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=768,
    )

    client = QdrantClient(path="./data/qdrant_db")

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    store = QdrantVectorStore(
        client=client, collection_name=collection, embedding=embeddings
    )
    store.add_documents(chunks)
    return store

if __name__ == "__main__":
    store = build_vectorstore(["data/sample.txt"])
    results = store.similarity_search("test query", k=3)
    for r in results:
        print(r.page_content[:150], "\n---")