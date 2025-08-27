# app.py
import streamlit as st
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

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)
app = graph.compile()

# ---- Streamlit UI ----
st.set_page_config(page_title="LangGraph Agent", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")

    llm_provider = st.selectbox("Select LLM Provider", ["Groq", "OpenAI"])
    if llm_provider == "Groq":
        model = st.selectbox(
            "Select Model",
            [
                "llama3-8b-8192",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma-7b-it",
                "llama3-groq-8b-8192-tool-use-preview",
                "llama3-groq-70b-8192-tool-use-preview"
            ]
        )
    else:
        model = st.selectbox("Select Model", ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])

    api_key = st.text_input(f"{llm_provider} API Key", type="password")

    st.warning("⚠️ Please enter your API key to proceed.")

    usecase = st.selectbox("Select Usecase", ["Basic Chatbot", "Research Assistant"])

# Main title
st.markdown(
    "<h1 style='text-align: center;'>🤖 LangGraph: Build Stateful Agentic AI Graph</h1>", 
    unsafe_allow_html=True
)

# Chat input + history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["bot"])

if query := st.chat_input("Enter your message:"):
    if not api_key:
        st.error("Please provide your API Key first!")
    else:
        try:
            response = app.invoke({
                "question": query,
                "api_key": api_key,
                "model": model,
                "provider": llm_provider,
                "answer": ""
            })
            st.session_state.messages.append({"user": query, "bot": response["answer"]})
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(response["answer"])
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
