from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()


# ---- Graph Definition ----
class State(TypedDict):
    question: str
    answer: str
    api_key: str
    model: str
    provider: str

def preprocess(state: State):
    """Preprocess input to set default values."""
    if "api_key" not in state or not state["api_key"]:
        state["api_key"]= os.getenv("GROQ_API_KEY")  # Default to Groq API key from environment
    if "model" not in state or not state["model"]:
        state["model"] = "llama3-70b-8192"  # Default Groq model
    if "provider" not in state or not state["provider"]:
        state["provider"] = "Groq"  # Default provider
    
    state["question"] = state["question"].strip()
    return state

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
graph.add_node("preprocess",preprocess)
graph.add_node("llm", call_llm)

graph.set_entry_point("preprocess")
graph.add_edge("preprocess", "llm")
graph.add_edge("llm", END)

# Export compiled graph
app = graph.compile()
