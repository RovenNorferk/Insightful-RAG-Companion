import os
import re
import pickle
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def extract_math_expression(query: str) -> str | None:
    """Return the first clean arithmetic expression, or None."""
    matches = re.findall(r"[\d\.\+\-\*/\(\)\s]+", query)
    for m in matches:
        cleaned = m.strip()
        if cleaned and re.fullmatch(r"[0-9\.\+\-\*/\(\)\s]+", cleaned):
            return cleaned
    return None


def route_query(query: str) -> str:
    """calculator | dictionary | rag_llm"""
    expr = extract_math_expression(query)
    if expr and len(expr.strip()) >= 3:
        return "calculator"
    if re.search(r"\bdefine\b", query, re.IGNORECASE):
        return "dictionary"
    return "rag_llm"


def contains_number_theory_query(query: str) -> bool:
    return bool(re.search(r"\b(?:\d+[a-zA-Z]*|[a-zA-Z]*\d+)\b", query))


def use_calculator(query: str) -> str:
    expr = extract_math_expression(query)
    if not expr:
        return "No valid expression found."
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error evaluating `{expr}`: {e}"


def define_term(term: str) -> str:
    return f"Definition of '{term}' (stub – replace with a real dictionary API later)."


qa_model_name = "valhalla/t5-base-qa-qg-hl"
qa_pipeline = pipeline(
    "text2text-generation",
    model=qa_model_name,
    tokenizer=qa_model_name,
    device=-1,          
)


def answer_with_qa_pipeline(query: str, retriever) -> tuple[str, list[Document]]:
    """Standard RAG → LLM answer."""
    docs = retriever.invoke(query)                    
    context = " ".join([doc.page_content for doc in docs])

    prompt = f"question: {query}  context: {context}"
    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result, docs


def number_with_theory(query: str, retriever) -> tuple[str, list[Document]]:
    """Special handling for queries that mix numbers + theory."""
    docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in docs])

    prompt = (
        f"Extract detailed technical information for: {query}\n\n"
        f"Context (use only if relevant):\n{context}"
    )
    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result, docs