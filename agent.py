# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# from langchain_community.tools import TavilySearchResults
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, List, Dict
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize Tavily search tool
# search_tool = TavilySearchResults(k=3)

# # ---- Global memory ----
# GLOBAL_HISTORY: List[Dict[str, str]] = []   # persists across calls


# # ---- Graph Definition ----
# class State(TypedDict):
#     question: str
#     answer: str
#     api_key: str
#     model: str
#     provider: str
#     action: str
#     history: List[Dict[str, str]]   


# # ---- Nodes ----
# def validate_input(state: State):
#     """Check if the user provided a valid question."""
#     if not state["question"] or len(state["question"].strip()) == 0:
#         raise ValueError("Question cannot be empty.")
#     return state

# def analyze_sentiment(state: State):
#     """Analyze sentiment of user input."""
#     import requests

#     api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
#     headers = {"Authorization": f"Bearer {state['']}"}
#     payload = {"inputs": state["question"]}
    
#     response = requests.post(api_url, headers=headers, json=payload)
#     sentiment = response.json()[0]["label"]  # e.g., "POSITIVE" or "NEGATIVE"
#     state["sentiment"] = sentiment
#     return state

# def preprocess(state: State):
#     """Preprocess input and inject global history."""
#     if "api_key" not in state or not state["api_key"]:
#         state["api_key"] = os.getenv("GROQ_API_KEY")
#     if "model" not in state or not state["model"]:
#         state["model"] = "llama3-70b-8192"
#     if "provider" not in state or not state["provider"]:
#         state["provider"] = "Groq"

#     # Always inject global history
#     state["history"] = GLOBAL_HISTORY

#     state["question"] = state["question"].strip()
#     return state


# def decide_action(state: State):
#     """Simple decision node: check if query looks like it needs search."""
#     q = state["question"].lower()
#     if any(word in q for word in ["latest", "current", "today", "news"]):
#         state["action"] = "search"
#     else:
#         state["action"] = "direct"
#     return state


# def tool_search(state: State):
#     """Use Tavily search tool."""
#     q = state["question"]
#     results = search_tool.invoke(q)  # list of search results
#     state["question"] = q + "\nContext: " + str(results)
#     return state


# def call_llm(state: State):
#     """Call the selected LLM (Groq or OpenAI) with conversation history."""
#     provider = state["provider"]
#     model = state["model"]
#     api_key = state["api_key"]

#     if provider == "Groq":
#         llm = ChatGroq(model=model, api_key=api_key)
#     elif provider == "OpenAI":
#         llm = ChatOpenAI(model=model, api_key=api_key)
#     else:
#         raise ValueError(f"Unsupported provider: {provider}")

#     # Keep only last 6 messages in context for efficiency
#     trimmed_history = state["history"][-6:]

#     # Build messages with history + current question
#     messages = trimmed_history + [{"role": "user", "content": state["question"]}]
#     resp = llm.invoke(messages)

#     # Store answer
#     state["answer"] = resp.content

#     # Update global history (persists across calls)
#     GLOBAL_HISTORY.extend([
#         {"role": "user", "content": state["question"]},
#         {"role": "assistant", "content": resp.content}
#     ])

#     return state


# def postprocess(state: State):
#     """Format or clean up the LLM response."""
#     answer = state["answer"].strip()
#     state["answer"] = f"🤖 Answer: {answer}"
#     return state


# def log_state(state: State):
#     """Log state for debugging / analytics."""
#     print(
#         f"LOG | Action: {state.get('action')} "
#         f"| Provider: {state['provider']} "
#         f"| Model: {state['model']} "
#         f"| History turns: {len(GLOBAL_HISTORY)//2} "
#         f"| Last Q: {state['question']}"
#     )
#     return state


# # ---- LangGraph ----
# graph = StateGraph(State)

# graph.add_node("validate", validate_input)
# graph.add_node("preprocess", preprocess)
# graph.add_node("decide", decide_action)
# graph.add_node("search", tool_search)
# graph.add_node("llm", call_llm)
# graph.add_node("postprocess", postprocess)
# graph.add_node("log", log_state)
# graph.add_node("sentiment", analyze_sentiment)

# # Entry
# graph.set_entry_point("validate")

# # Flow
# graph.add_edge("validate", "preprocess")
# graph.add_edge("preprocess", "decide")

# # Branching
# graph.add_conditional_edges(
#     "decide",
#     lambda state: state["action"],
#     {
#         "search": "search",
#         "direct": "llm"
#     }
# )

# # Continue
# graph.add_edge("search", "llm")
# graph.add_edge("llm", "postprocess")
# graph.add_edge("postprocess", "log")
# graph.add_edge("log", END)

# # Compile
# app = graph.compile()
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # 👈 add this
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import os
from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

# Initialize Tavily search tool
search_tool = TavilySearchResults(k=3)

# ---- Global memory ----
GLOBAL_HISTORY: List[Dict[str, str]] = []   # persists across calls

# ---- Graph Definition ----
class State(TypedDict):
    question: str
    answer: str
    api_key: str
    model: str
    provider: str
    action: str
    history: List[Dict[str, str]]
    sentiment: str  # new field

# ---- Nodes ----
def validate_input(state: State):
    if not state["question"] or len(state["question"].strip()) == 0:
        raise ValueError("Question cannot be empty.")
    return state

def analyze_sentiment(state: State):
    """Analyze sentiment of user input."""
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
    payload = {"inputs": state["question"]}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        sentiment = response.json()[0]["label"]  # "POSITIVE" or "NEGATIVE"
    except Exception:
        sentiment = "NEUTRAL"
    state["sentiment"] = sentiment
    return state

def preprocess(state: State):
    if "api_key" not in state or not state["api_key"]:
        state["api_key"] = os.getenv("GROQ_API_KEY")
    if "model" not in state or not state["model"]:
        state["model"] = "llama3-70b-8192"
    if "provider" not in state or not state["provider"]:
        state["provider"] = "Groq"

    # Use session state for history
    if "history" not in st.session_state:
        st.session_state.history = []
    state["history"] = st.session_state.history
    state["question"] = state["question"].strip()
    return state

def decide_action(state: State):
    q = state["question"].lower()
    if any(word in q for word in ["latest", "current", "today", "news"]):
        state["action"] = "search"
    else:
        state["action"] = "direct"
    return state

def tool_search(state: State):
    q = state["question"]
    results = search_tool.invoke(q)
    state["question"] = q + "\nContext: " + str(results)
    return state

def call_llm(state: State):
    provider = state["provider"]
    model = state["model"]
    api_key = state["api_key"]

    if provider == "Groq":
        llm = ChatGroq(model=model, api_key=api_key)

    elif provider == "OpenAI":
        llm = ChatOpenAI(model=model, api_key=api_key)

    elif provider == "Gemini":
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,   # 👈 Gemini key
            temperature=0.7
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Keep only last 6 turns
    trimmed_history = state["history"][-6:]

    prompt = state["question"]
    if state.get("sentiment") == "NEGATIVE":
        prompt = f"Respond empathetically: {prompt}"
    elif state.get("sentiment") == "POSITIVE":
        prompt = f"Respond cheerfully: {prompt}"

    messages = trimmed_history + [{"role": "user", "content": prompt}]
    resp = llm.invoke(messages)

    # Some models (Gemini, Groq, OpenAI) all return .content
    state["answer"] = resp.content

    # Update session history
    st.session_state.history.extend([
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": resp.content}
    ])
    state["history"] = st.session_state.history
    return state

def postprocess(state: State):
    answer = state["answer"].strip()
    state["answer"] = f"🤖 Answer: {answer}"
    return state

def log_state(state: State):
    print(
        f"LOG | Action: {state.get('action')} "
        f"| Provider: {state['provider']} "
        f"| Model: {state['model']} "
        f"| Sentiment: {state.get('sentiment')} "
        f"| History turns: {len(GLOBAL_HISTORY)//2} "
        f"| Last Q: {state['question']}"
    )
    return state

# ---- LangGraph ----
graph = StateGraph(State)

graph.add_node("validate", validate_input)
graph.add_node("sentiment", analyze_sentiment)
graph.add_node("preprocess", preprocess)
graph.add_node("decide", decide_action)
graph.add_node("search", tool_search)
graph.add_node("llm", call_llm)
graph.add_node("postprocess", postprocess)
graph.add_node("log", log_state)

# Entry
graph.set_entry_point("validate")

# Flow
graph.add_edge("validate", "sentiment")
graph.add_edge("sentiment", "preprocess")
graph.add_edge("preprocess", "decide")

graph.add_conditional_edges(
    "decide",
    lambda state: state["action"],
    {
        "search": "search",
        "direct": "llm"
    }
)

graph.add_edge("search", "llm")
graph.add_edge("llm", "postprocess")
graph.add_edge("postprocess", "log")
graph.add_edge("log", END)

# Compile
app = graph.compile()
