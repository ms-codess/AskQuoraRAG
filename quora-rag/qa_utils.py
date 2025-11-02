import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


def load_retriever(chroma_dir: str, k: int = 4, search_type: str = "similarity"):
    load_dotenv()
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    db = Chroma(persist_directory=str(Path(chroma_dir)), embedding_function=embeddings)
    if search_type == "mmr":
        return db.as_retriever(search_type="mmr", search_kwargs={"k": k})
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})


def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        parts.append(d.page_content)
    return "\n\n---\n\n".join(parts)


def answer_question(
    question: str,
    retriever,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    load_dotenv()
    model = ChatOpenAI(model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering based strictly on the provided context. If the answer is not in the context, say you don't know.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    docs = retriever.get_relevant_documents(question)
    context = _format_docs(docs)
    messages = prompt.format_messages(question=question, context=context)
    resp = model.invoke(messages)

    sources: List[Dict[str, Any]] = []
    for d in docs:
        sources.append({
            "content": d.page_content,
            "metadata": d.metadata,
        })

    return {"answer": resp.content, "sources": sources}

