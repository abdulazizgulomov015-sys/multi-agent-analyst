from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GOOGLE_API_KEY, MODEL_NAME

def extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content

def llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

def score_faithfulness(answer: str, evidence: str) -> float:
    """RAGAS-style: is every claim in the answer backed by the evidence? 0.0-1.0"""
    prompt = f"""Evidence:
{evidence}

Answer:
{answer}

Score from 0.0 to 1.0 how well the answer is supported ONLY by the evidence above
(1.0 = fully grounded, 0.0 = fabricated/unsupported).
Reply with ONLY the number."""
    response = llm().invoke(prompt)
    try:
        return float(extract_text(response.content).strip())
    except ValueError:
        return 0.0

def score_answer_relevancy(question: str, answer: str) -> float:
    """RAGAS-style: does the answer actually address the question? 0.0-1.0"""
    prompt = f"""Question: {question}
Answer: {answer}

Score from 0.0 to 1.0 how directly and completely the answer addresses the question.
Reply with ONLY the number."""
    response = llm().invoke(prompt)
    try:
        return float(extract_text(response.content).strip())
    except ValueError:
        return 0.0

def llm_judge(question: str, answer: str) -> dict:
    """Overall LLM-as-judge score with reasoning, 1-5 scale."""
    prompt = f"""You are an expert judge evaluating an AI system's answer.

Question: {question}
Answer: {answer}

Rate the answer 1-5 (5=excellent) and give a one-sentence reason.
Reply in EXACTLY this format:
Score: <number>
Reason: <one sentence>"""
    response = llm().invoke(prompt)
    text = extract_text(response.content).strip()
    score_line = [l for l in text.split("\n") if l.startswith("Score:")]
    reason_line = [l for l in text.split("\n") if l.startswith("Reason:")]
    score = float(score_line[0].replace("Score:", "").strip()) if score_line else 0.0
    reason = reason_line[0].replace("Reason:", "").strip() if reason_line else text
    return {"score": score, "reason": reason}

if __name__ == "__main__":
    question = "What is the total sales amount in the North region?"
    answer = "The total sales amount in the North region is 2190.75."
    evidence = "Query: SELECT SUM(amount) FROM sales WHERE region='North'\nResult: [(2190.75,)]"

    faith = score_faithfulness(answer, evidence)
    relevancy = score_answer_relevancy(question, answer)
    judge = llm_judge(question, answer)

    print(f"Faithfulness: {faith}")
    print(f"Answer Relevancy: {relevancy}")
    print(f"LLM Judge: {judge['score']}/5 — {judge['reason']}")