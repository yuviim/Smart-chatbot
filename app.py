import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
from typing import Annotated
from langchain_tavily import TavilySearch
import json

# 1. Human assistance tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response.get("data", "")

# 2. Load API keys from Streamlit secrets
ANTHROPIC_API_KEY = st.secrets["anthropic"]["api_key"]
TAVILY_API_KEY = st.secrets["Tavily"]["api_key"]

# 3. Initialize LLM and Tools
tavily_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=2)
tools = [tavily_tool, human_assistance]

llm = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# 4. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 5. Tool Node
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        """Initialize the BasicToolNode with a list of tools."""
        self.tools_by_name = {t.name: t for t in tools if hasattr(t, "name")}

    def __call__(self, inputs: dict):
        """Process incoming inputs and invoke appropriate tool."""
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in input")

        message = messages[-1]
        outputs = []

        if hasattr(message, "tool_calls"):
            for call in message.tool_calls:
                tool_name = call.get("name")
                args = call.get("args", {})
                tool_id = call.get("id", "tool_call")

                if tool_name in self.tools_by_name:
                    result = self.tools_by_name[tool_name].invoke(args)

                    # Format Tavily results, fallback otherwise
                    if isinstance(result, dict) and "results" in result:
                        summary = ""
                        for r in result["results"][:2]:
                            title = r.get("title", "Untitled")
                            url = r.get("url", "")
                            content = r.get("content", "")
                            summary += f"**{title}**\n{content}\n[Read more]({url})\n\n"
                        result = summary.strip()
                    else:
                        result = str(result)

                    outputs.append(
                        ToolMessage(
                            content=result,
                            name=tool_name,
                            tool_call_id=tool_id,
                        )
                    )
        return {"messages": outputs}

# 6. Chatbot node
def chatbot(state: State):
    """Process messages and invoke LLM with tools."""
    langchain_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        else:
            langchain_messages.append(msg)

    return {"messages": [llm_with_tools.invoke(langchain_messages)]}

# 7. Tool router
def route_tools(state: State):
    """Route the flow to tools or end based on message content."""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, dict) and "tool_calls" in last_msg:
            return "tools"
        elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
    return END

# 8. Build the LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", BasicToolNode(tools=tools))
graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

# 9. Streamlit UI
st.title("🤖Smart Chatbot with Human Guidance & Instant Search Results")

# Initialize session history if not already done
if "history" not in st.session_state:
    st.session_state.history = []

# Get user input
user_input = st.text_input("You:", key="input").strip()

# Add user input to history if it's not empty
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    full_event_messages = []

    with st.spinner("🤖 Thinking..."):
        for event in graph.stream({"messages": st.session_state.history}):
            for value in event.values():
                full_event_messages.extend(value["messages"])

    # Append final assistant or tool messages only
    for msg in full_event_messages:
        content = None
        role = "assistant"

        if isinstance(msg, AIMessage):
            # Only show AI messages that contain plain text (not tool call plans)
            if isinstance(msg.content, str):
                content = msg.content
        elif isinstance(msg, ToolMessage):
            content = msg.content
        elif isinstance(msg, dict):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

        # Only append if there's actual text content to show
        if content and isinstance(content, str) and content.strip():
            st.session_state.history.append({"role": role, "content": content})

# 10. Display full chat history
for message in st.session_state.history:
    if message["role"] == "user":
        st.markdown(f"**🧑 You:** {message['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {message['content']}")
