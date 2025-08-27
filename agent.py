from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict


# ---- Graph Definition ----
class State(TypedDict):
    question: str
    answer: str
    api_key: str
    model: str
    provider: str


def call_llm(state: State):
    """Call the selected LLM (Groq or OpenAI)."""
    provider = state["provider"]
    model = state["model"]
    api_key = state["api_key"]

    if provider == "Groq":
        llm = ChatGroq(model=model, api_key=api_key)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    resp = llm.invoke(state["question"])
    return {"answer": resp.content}


# ---- LangGraph ----
graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

# Export compiled graph
app = graph.compile()
