The error is clear from the screenshot: [weathermcpgeminiapp-bhtobwr8twyrcpgc5wr8qj.streamlit](https://weathermcpgeminiapp-bhtobwr8twyrcpgc5wr8qj.streamlit.app/)

```
mcp.shared.exceptions.McpError: Connection closed
```

This means the MCP server (`weather_mcp_server.py`) **started but immediately closed the connection**. This is a known issue on **Streamlit Cloud** because the MCP server runs as a subprocess via `stdio`, and Streamlit Cloud's environment sometimes kills subprocesses or has path issues.

***

## Root Cause

The `mcp.run()` call in `weather_mcp_server.py` uses `stdio` transport, which works locally but **Streamlit Cloud has issues with subprocess stdio MCP servers** because:
1. The Python subprocess path may differ.
2. Streamlit Cloud sandboxes subprocess execution.
3. The MCP server process exits before the client connects.

***

## Fix — Remove the separate MCP server, run everything in-process

The cleanest fix for Streamlit Cloud is to **call the weather functions directly** inside the agent instead of spawning a subprocess. Replace both files with these:

***

### New `main.py` (fully fixed for Streamlit Cloud)

```python
import os
import json
import asyncio
from typing import Any, Dict, List, TypedDict

import streamlit as st
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a practical weather assistant inside a Streamlit dashboard.
Use available tools whenever the user asks for live weather,
forecast, alerts, coordinates, or city comparisons.
If tool results are available, ground your answer in them.
Keep answers crisp, useful, and action-oriented.
""".strip()

MODEL_NAME = "gemini-2.0-flash"

# --------------------------------------------------------------------
# UTIL: API KEY
# --------------------------------------------------------------------
def ensure_api_key() -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    api_key = st.sidebar.text_input(
        "Gemini API Key (optional override)",
        type="password",
        value=api_key,
        help="In Streamlit Cloud, put GEMINI_API_KEY in Secrets.",
    )

    if not api_key:
        st.warning("Set GEMINI_API_KEY in Streamlit secrets or here in the sidebar.")
        st.stop()
    return api_key


# --------------------------------------------------------------------
# WEATHER TOOLS (in-process, no subprocess needed)
# --------------------------------------------------------------------
import httpx

USER_AGENT = "weather-mcp-streamlit/1.0"


async def geocode_city(city: str) -> Dict[str, Any]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Could not find coordinates for city: {city}")
    item = results[0]
    return {
        "name": item.get("name"),
        "country": item.get("country"),
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "timezone": item.get("timezone"),
    }


async def get_current_weather(city: str) -> str:
    location = await geocode_city(city)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "is_day,precipitation,rain,weather_code,cloud_cover,wind_speed_10m"
        ),
        "timezone": "auto",
    }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    current = data.get("current", {})
    return (
        f"Current weather for {location['name']}, {location['country']}: "
        f"temperature {current.get('temperature_2m')}°C, "
        f"feels like {current.get('apparent_temperature')}°C, "
        f"humidity {current.get('relative_humidity_2m')}%, "
        f"precipitation {current.get('precipitation')} mm, "
        f"cloud cover {current.get('cloud_cover')}%, "
        f"wind {current.get('wind_speed_10m')} km/h, "
        f"weather code {current.get('weather_code')}, "
        f"timezone {location.get('timezone')}."
    )


async def get_forecast(city: str, days: int = 3) -> str:
    days = max(1, min(days, 7))
    location = await geocode_city(city)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,wind_speed_10m_max"
        ),
        "forecast_days": days,
        "timezone": "auto",
    }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    daily = data.get("daily", {})
    lines = [f"Forecast for {location['name']}, {location['country']} ({days} days):"]
    for i in range(len(daily.get("time", []))):
        lines.append(
            f"- {daily['time'][i]}: "
            f"min {daily['temperature_2m_min'][i]}°C, "
            f"max {daily['temperature_2m_max'][i]}°C, "
            f"precipitation {daily['precipitation_sum'][i]} mm, "
            f"wind up to {daily['wind_speed_10m_max'][i]} km/h"
        )
    return "\n".join(lines)


async def compare_weather(city_a: str, city_b: str) -> str:
    first = await get_current_weather(city_a)
    second = await get_current_weather(city_b)
    return first + "\n" + second


# Tool registry for the agent
TOOL_REGISTRY = {
    "get_current_weather": get_current_weather,
    "get_forecast": get_forecast,
    "compare_weather": compare_weather,
}

# Gemini function declarations
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_current_weather",
        description="Get current weather for a city including temperature, humidity, wind and precipitation.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Chennai"}
            },
            "required": ["city"],
        },
    ),
    types.FunctionDeclaration(
        name="get_forecast",
        description="Get daily forecast (min/max temp, precipitation, wind) for up to 7 days.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Number of days (1-7), default 3"},
            },
            "required": ["city"],
        },
    ),
    types.FunctionDeclaration(
        name="compare_weather",
        description="Compare current weather between two cities.",
        parameters={
            "type": "object",
            "properties": {
                "city_a": {"type": "string", "description": "First city name"},
                "city_b": {"type": "string", "description": "Second city name"},
            },
            "required": ["city_a", "city_b"],
        },
    ),
]


# --------------------------------------------------------------------
# GEMINI WEATHER AGENT (in-process, no subprocess MCP)
# --------------------------------------------------------------------
class GeminiWeatherAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def run(self, user_prompt: str) -> Dict[str, Any]:
        contents: List[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part(text=SYSTEM_PROMPT + "\n\nUser request: " + user_prompt)],
            )
        ]
        trace: List[Dict[str, Any]] = []

        for _ in range(6):
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
                ),
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts
            function_calls = [
                p.function_call for p in parts if getattr(p, "function_call", None)
            ]

            if not function_calls:
                final_text = response.text or "No response generated."
                return {"answer": final_text, "trace": trace}

            contents.append(candidate.content)

            function_response_parts = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                tool_fn = TOOL_REGISTRY.get(fc.name)
                if tool_fn:
                    try:
                        tool_text = await tool_fn(**args)
                    except Exception as e:
                        tool_text = f"Error calling {fc.name}: {str(e)}"
                else:
                    tool_text = f"Unknown tool: {fc.name}"

                trace.append({"tool": fc.name, "args": args, "result": tool_text})
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": tool_text},
                    )
                )

            contents.append(
                types.Content(role="tool", parts=function_response_parts)
            )

        return {
            "answer": "I reached the tool-call limit for this request.",
            "trace": trace,
        }


# --------------------------------------------------------------------
# LANGGRAPH AGENT
# --------------------------------------------------------------------
class AgentState(TypedDict):
    user_input: str
    answer: str
    trace: List[Dict[str, Any]]


def build_langgraph(agent: GeminiWeatherAgent):
    graph = StateGraph(AgentState)

    async def mcp_node(state: AgentState) -> AgentState:
        result = await agent.run(state["user_input"])
        return {
            "user_input": state["user_input"],
            "answer": result["answer"],
            "trace": result["trace"],
        }

    graph.add_node("weather_agent", mcp_node)
    graph.set_entry_point("weather_agent")
    graph.add_edge("weather_agent", END)
    return graph.compile()


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
        agent = GeminiWeatherAgent(api_key)
        st.session_state.agent = agent
        st.session_state.graph = build_langgraph(agent)
    return st.session_state.agent, loop, st.session_state.graph


def render_sidebar():
    st.sidebar.title("Weather MCP Agent")
    st.sidebar.success("Gemini key loaded")
    st.sidebar.caption(f"Model: {MODEL_NAME}")
    st.sidebar.subheader("Available Tools")
    st.sidebar.markdown("- **get_current_weather**: Live weather for any city")
    st.sidebar.markdown("- **get_forecast**: Up to 7-day daily forecast")
    st.sidebar.markdown("- **compare_weather**: Side-by-side city comparison")
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Open-Meteo (free, no key needed)")


# --------------------------------------------------------------------
# MAIN STREAMLIT APP
# --------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Weather MCP with Gemini",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ Weather MCP Dashboard — Gemini + LangGraph")
    st.caption(
        "Gemini reasons with weather tools via function calling. "
        "LangGraph orchestrates the agent loop. Streamlit shows the full trace."
    )

    api_key = ensure_api_key()
    agent, loop, graph = bootstrap_agent(api_key)
    render_sidebar()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Ask a weather question")
        prompt = st.text_area(
            "Type your weather task below:",
            height=130,
            value="Give me current weather in Chennai and a short practical summary.",
        )
        run = st.button("Run weather task", use_container_width=True, type="primary")

        st.markdown("**Try these high-potential tasks:**")
        sample_prompts = [
            "Current weather in Chennai with humidity and wind.",
            "3-day forecast for Madurai with clothing advice.",
            "Compare Chennai and Coimbatore weather today.",
            "Weather in Bengaluru with a travel recommendation.",
            "5-day forecast for Delhi — should I carry an umbrella?",
        ]
        for item in sample_prompts:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Performance trace")
        trace_box = st.container(border=True)
        answer_box = st.container(border=True)

    if run and prompt.strip():
        with st.spinner("LangGraph agent + Gemini is working..."):
            state: AgentState = {
                "user_input": prompt.strip(),
                "answer": "",
                "trace": [],
            }
            result_state = loop.run_until_complete(graph.ainvoke(state))

        with answer_box:
            st.markdown("### Final Answer")
            st.write(result_state["answer"])

        with trace_box:
            st.markdown("### Tool Execution Trace")
            if not result_state["trace"]:
                st.info("No tools were called for this request.")
            for i, step in enumerate(result_state["trace"], start=1):
                with st.expander(f"Step {i} — {step['tool']}", expanded=True):
                    st.code(json.dumps(step["args"], indent=2), language="json")
                    st.text_area(
                        "Tool result",
                        value=step["result"],
                        height=120,
