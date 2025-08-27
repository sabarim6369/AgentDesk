import streamlit as st
from agent import app  # Import from agent.py


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


# ---- Main title ----
st.markdown(
"<h1 style='text-align: center;'>🤖 AgentDesk: Talk with Multiple AI Agents</h1>",
    unsafe_allow_html=True
)


# ---- Chat input + history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["bot"])

if query := st.chat_input("Enter your message:"):
    if not api_key:
        st.error("Please provide your API Key first!")
    else:
        # Show user message immediately
        st.session_state.messages.append({"user": query, "bot": None})
        st.chat_message("user").write(query)

        # Create placeholder for assistant
        placeholder = st.chat_message("assistant").empty()

        try:
            with st.spinner("🤖 Thinking..."):
                response = app.invoke({
                    "question": query,
                    "api_key": api_key,
                    "model": model,
                    "provider": llm_provider,
                    "answer": ""
                })
                answer = response["answer"]
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        # Update assistant message
        st.session_state.messages[-1]["bot"] = answer
        placeholder.write(answer)
