import json
import os

MEMORY_PATH = "data/memory.json"

def load_memory() -> list[dict]:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_interaction(question: str, answer: str):
    history = load_memory()
    history.append({"question": question, "answer": answer})
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def get_relevant_history(question: str, limit: int = 3) -> list[dict]:
    history = load_memory()
    question_words = set(question.lower().split())
    scored = []
    for item in history:
        overlap = len(question_words & set(item["question"].lower().split()))
        if overlap > 0:
            scored.append((overlap, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]

if __name__ == "__main__":
    save_interaction("What is the total sales in the North region?", "2190.75")
    save_interaction("What is coffee?", "A brewed drink from roasted beans.")
    relevant = get_relevant_history("How much did North sell?")
    print("Relevant history found:")
    for r in relevant:
        print("-", r)