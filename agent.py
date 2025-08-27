from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Tavily search tool
search_tool = TavilySearchResults(k=3)

# ---- Graph Definition ----
class State(TypedDict):
    question: str
    answer: str
    api_key: str
    model: str
    provider: str
    action: str  # NEW: to store decision ("search" or "direct")


# ---- Nodes ----
def validate_input(state: State):
    """Check if the user provided a valid question."""
    if not state["question"] or len(state["question"].strip()) == 0:
        raise ValueError("Question cannot be empty.")
    return state


def preprocess(state: State):
    """Preprocess input to set default values."""
    if "api_key" not in state or not state["api_key"]:
        state["api_key"] = os.getenv("GROQ_API_KEY")
    if "model" not in state or not state["model"]:
        state["model"] = "llama3-70b-8192"
    if "provider" not in state or not state["provider"]:
        state["provider"] = "Groq"
    
    state["question"] = state["question"].strip()
    return state


def decide_action(state: State):
    """Simple decision node: check if query looks like it needs search."""
    q = state["question"].lower()
    if any(word in q for word in ["latest", "current", "today", "news"]):
        state["action"] = "search"
    else:
        state["action"] = "direct"
    return state


def tool_search(state: State):
    """Real Tavily search tool."""
    q = state["question"]
    results = search_tool.invoke(q)  # list of search results
    state["question"] = q + "\nContext: " + str(results)
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


def postprocess(state: State):
    """Format or clean up the LLM response."""
    answer = state["answer"].strip()
    state["answer"] = f"🤖 Answer: {answer}"
    return state


def log_state(state: State):
    """Log state for debugging / analytics."""
    print(
        f"LOG | Action: {state.get('action')} "
        f"| Provider: {state['provider']} "
        f"| Model: {state['model']} "
        f"| Q: {state['question']}"
    )
    return state


# ---- LangGraph ----
graph = StateGraph(State)

graph.add_node("validate", validate_input)
graph.add_node("preprocess", preprocess)
graph.add_node("decide", decide_action)
graph.add_node("search", tool_search)
graph.add_node("llm", call_llm)
graph.add_node("postprocess", postprocess)
graph.add_node("log", log_state)

# Entry
graph.set_entry_point("validate")

# Flow
graph.add_edge("validate", "preprocess")
graph.add_edge("preprocess", "decide")

# Branching
graph.add_conditional_edges(
    "decide",
    lambda state: state["action"],
    {
        "search": "search",
        "direct": "llm"
    }
)

# Continue
graph.add_edge("search", "llm")
graph.add_edge("llm", "postprocess")
graph.add_edge("postprocess", "log")
graph.add_edge("log", END)

# Compile
app = graph.compile()
