import os
import json
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, List, TypedDict

import streamlit as st
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# >>> LangGraph additions start
from langgraph.graph import StateGraph, END
# >>> LangGraph additions end

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a practical weather assistant inside a Streamlit dashboard.
Use available MCP tools whenever the user asks for live weather,
forecast, alerts, coordinates, or city comparisons.
If tool results are available, ground your answer in them.
Keep answers crisp, useful, and action-oriented.
""".strip()

# Latest Flash model from Gemini API docs
MODEL_NAME = "gemini-3.5-flash"  # you can change to another flash model if needed


# --------------------------------------------------------------------
# UTIL: API KEY
# --------------------------------------------------------------------
def ensure_api_key() -> str:
    """
    Read Gemini API key from:
    - Streamlit secrets (GEMINI_API_KEY), or
    - Environment variable GEMINI_API_KEY.
    """
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    api_key = st.sidebar.text_input(
        "Gemini API Key (optional override)",
        type="password",
        value=api_key,
        help="In Streamlit Cloud, prefer putting GEMINI_API_KEY in Secrets.",
    )

    if not api_key:
        st.warning("Set GEMINI_API_KEY in Streamlit secrets or here in the sidebar.")
        st.stop()
    return api_key


# --------------------------------------------------------------------
# MCP CLIENT AGENT
# --------------------------------------------------------------------
class MCPWeatherAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools_schema: List[types.FunctionDeclaration] = []
        self.connected = False

    async def connect_stdio_server(self, script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[script_path],
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

        tool_result = await self.session.list_tools()
        self.tools_schema = []
        for tool in tool_result.tools:
            self.tools_schema.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or f"MCP tool: {tool.name}",
                    parameters=tool.inputSchema,
                )
            )

        self.connected = True
        return tool_result.tools

    async def close(self):
        await self.exit_stack.aclose()
        self.connected = False

    async def run(self, user_prompt: str) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("MCP session is not connected")

        contents: List[types.Content] = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=SYSTEM_PROMPT + "\n\nUser request: " + user_prompt
                    )
                ],
            )
        ]
        trace: List[Dict[str, Any]] = []

        # Simple tool-calling loop
        for _ in range(6):
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    tools=[types.Tool(function_declarations=self.tools_schema)],
                ),
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts
            function_calls = [
                p.function_call for p in parts if getattr(p, "function_call", None)
            ]

            # If no more tool calls, we have final answer
            if not function_calls:
                final_text = response.text or "No response generated."
                return {"answer": final_text, "trace": trace}

            # Add model function-call message into conversation
            contents.append(candidate.content)

            # Execute each tool call with MCP
            function_response_parts = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                result = await self.session.call_tool(fc.name, args)

                text_parts = []
                for item in result.content:
                    item_text = getattr(item, "text", None)
                    if item_text:
                        text_parts.append(item_text)
                    else:
                        text_parts.append(str(item))

                tool_text = "\n".join(text_parts) if text_parts else ""
                trace.append({"tool": fc.name, "args": args, "result": tool_text})

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": tool_text},
                    )
                )

            # Feed tool results back to Gemini
            contents.append(
                types.Content(role="tool", parts=function_response_parts)
            )

        return {
            "answer": "I reached the tool-call limit for this request.",
            "trace": trace,
        }


# --------------------------------------------------------------------
# LANGGRAPH: wrap MCP agent in a graph
# --------------------------------------------------------------------
# We use a tiny state dict: just user input, answer, trace.
class AgentState(TypedDict):
    user_input: str
    answer: str
    trace: List[Dict[str, Any]]


def build_langgraph(agent: MCPWeatherAgent):
    """
    Build a very small LangGraph agent:
    - Node 'mcp_agent': calls MCPWeatherAgent.run
    - Then END.
    """
    graph = StateGraph(AgentState)

    async def mcp_node(state: AgentState) -> AgentState:
        result = await agent.run(state["user_input"])
        return {
            "user_input": state["user_input"],
            "answer": result["answer"],
            "trace": result["trace"],
        }

    graph.add_node("mcp_agent", mcp_node)
    graph.set_entry_point("mcp_agent")
    graph.add_edge("mcp_agent", END)

    compiled = graph.compile()
    return compiled


# --------------------------------------------------------------------
# STREAMLIT HELPERS
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def bootstrap_agent(api_key: str):
    loop = get_loop()
    if "agent" not in st.session_state:
        agent = MCPWeatherAgent(api_key)
        script_path = os.path.abspath("weather_mcp_server.py")
        tools = loop.run_until_complete(agent.connect_stdio_server(script_path))
        st.session_state.agent = agent
        st.session_state.mcp_tools = [
            {"name": t.name, "description": t.description} for t in tools
        ]
        # >>> LangGraph: build graph once and store
        st.session_state.graph = build_langgraph(agent)
    return st.session_state.agent, loop, st.session_state.graph


def render_sidebar(api_key: str):
    st.sidebar.title("Setup")
    st.sidebar.success("Gemini key loaded")
    st.sidebar.caption(f"Model: {MODEL_NAME}")

    if "mcp_tools" in st.session_state:
        st.sidebar.subheader("MCP Tools")
        for tool in st.session_state.mcp_tools:
            st.sidebar.markdown(
                f"- **{tool['name']}**: {tool['description']}"
            )


# --------------------------------------------------------------------
# MAIN STREAMLIT APP
# --------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Weather MCP with Gemini",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ Weather MCP Dashboard with Gemini")
    st.caption(
        "Gemini uses a local MCP weather server, and Streamlit shows both "
        "the answer and the tool performance trace. A LangGraph agent "
        "wraps the reasoning loop."
    )

    api_key = ensure_api_key()
    agent, loop, graph = bootstrap_agent(api_key)
    render_sidebar(api_key)

    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.subheader("Ask weather tasks")
        prompt = st.text_area(
            "Example: Give me today's weather in Chennai, plus a 3‑day outlook and one travel tip.",
            height=130,
            value="Give me current weather in Chennai and a short practical summary.",
        )
        run = st.button("Run weather task", use_container_width=True)

        st.markdown("**High-potential task ideas**")
        sample_prompts = [
            "Current weather in Chennai with humidity and wind.",
            "3-day forecast for Madurai with clothing advice.",
            "Compare Chennai and Coimbatore weather today.",
            "Weather in Bengaluru with a simple travel recommendation.",
        ]
        for item in sample_prompts:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Performance view")
        trace_box = st.container(border=True)
        answer_box = st.container(border=True)

    if run and prompt.strip():
        with st.spinner("LangGraph agent is reasoning with MCP tools..."):
            # >>> LangGraph: run graph instead of calling agent.run directly
            state: AgentState = {"user_input": prompt.strip(), "answer": "", "trace": []}
            # LangGraph run is sync; inside we `await` the mcp_node
            result_state = loop.run_until_complete(graph.ainvoke(state))

        with answer_box:
            st.markdown("### Final answer")
            st.write(result_state["answer"])

        with trace_box:
            st.markdown("### Tool execution trace")
            if not result_state["trace"]:
                st.info("No MCP tools were used for this request.")
            for i, step in enumerate(result_state["trace"], start=1):
                with st.expander(f"Step {i}: {step['tool']}", expanded=True):
                    st.code(json.dumps(step["args"], indent=2), language="json")
                    st.text(step["result"])

    st.divider()
    st.subheader("What this project gives you")
    st.markdown(
        """
- Gemini tool calling over MCP tools (MCPWeatherAgent).
- A LangGraph agent node that orchestrates the reasoning loop.
- A local weather MCP server using Open‑Meteo.
- Streamlit UI that shows both answer quality and tool performance.
- Clean GitHub + Streamlit Cloud ready structure (API key in Secrets).
"""
    )


if __name__ == "__main__":
    main()
