# filename: app.py

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict

# === API Key and Model Setup ===
API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)

if not API_KEY:
    st.error("Please set your Anthropic API key in Streamlit secrets!")
    st.stop()

llm = ChatAnthropic(
    api_key=API_KEY,
    model="claude-3-5-sonnet-20240620",
)

# === LangGraph Setup ===
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State) -> State:
    langchain_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        else:
            langchain_messages.append(msg)

    try:
        response_text = ""
        for chunk in llm.stream(langchain_messages):
            if hasattr(chunk, "content") and chunk.content:
                response_text += chunk.content

        final_response = AIMessage(content=response_text)
        return {"messages": [final_response]}
    
    except Exception as e:
        st.error(f"Chatbot error: {e}")
        raise

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# === Streamlit UI ===

st.set_page_config(page_title="Anthropic Chatbot 🤖", page_icon="🤖")

st.title("🤖 Anthropic Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Query the chatbot
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            response_text = ""

            for event in graph.stream({"messages": st.session_state.messages}):
                for value in event.values():
                    partial = value["messages"][-1].content
                    response_text = partial  # latest chunk

                    placeholder.markdown(response_text)

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            placeholder.error(f"Error: {e}")
