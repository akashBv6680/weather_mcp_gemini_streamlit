import os
import json
import asyncio
from typing import Any, Dict, List, TypedDict

import httpx
import streamlit as st
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

# --- TTS imports (from your RAG app) ---
import io
import base64
try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are a practical weather assistant in a Streamlit dashboard.
Use the available weather tools whenever the user asks for
live weather, forecast, rain chances, wind, or city comparisons.
Always:
- Call tools first when weather data is needed.
- Ground your final answer in the tool results.
- Give concise, action-oriented explanations for a normal user.
""".strip()

MODEL_NAME = "gemini-2.5-flash"
USER_AGENT = "weather-mcp-streamlit/1.0"


# ------------------------------------------------------------
# GEMINI API KEY HANDLING
# ------------------------------------------------------------

def ensure_api_key() -> str:
    # Prefer Streamlit secrets in Cloud
    api_key = st.secrets.get("GEMINI_API_KEY", None)

    # Fallback to environment variable (local dev)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    # Allow override from sidebar
    api_key = st.sidebar.text_input(
        "Gemini API Key (optional override)",
        type="password",
        value=api_key,
        help="In Streamlit Cloud, add GEMINI_API_KEY in Secrets.",
    )

    if not api_key:
        st.warning("Please set GEMINI_API_KEY in Secrets or here in the sidebar.")
        st.stop()

    return api_key


# ------------------------------------------------------------
# WEATHER FUNCTIONS (IN-PROCESS, NO MCP SERVER)
# ------------------------------------------------------------

async def geocode_city(city: str) -> Dict[str, Any]:
    """Convert city name to coordinates via Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}")

    item = results[0]
    return {
        "name": item.get("name"),
        "country": item.get("country"),
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "timezone": item.get("timezone"),
    }


async def get_current_weather(city: str) -> str:
    loc = await geocode_city(city)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,weather_code,cloud_cover,wind_speed_10m"
        ),
        "timezone": "auto",
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    c = data.get("current", {})
    return (
        f"Current weather for {loc['name']}, {loc['country']}: "
        f"temperature {c.get('temperature_2m')}C, "
        f"feels like {c.get('apparent_temperature')}C, "
        f"humidity {c.get('relative_humidity_2m')}%, "
        f"precipitation {c.get('precipitation')} mm, "
        f"cloud cover {c.get('cloud_cover')}%, "
        f"wind {c.get('wind_speed_10m')} km/h, "
        f"weather code {c.get('weather_code')}, "
        f"timezone {loc.get('timezone')}."
    )


async def get_forecast(city: str, days: int = 3) -> str:
    days = max(1, min(days, 7))
    loc = await geocode_city(city)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,wind_speed_10m_max"
        ),
        "forecast_days": days,
        "timezone": "auto",
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    daily = data.get("daily", {})
    lines = [f"Forecast for {loc['name']}, {loc['country']} ({days} days):"]
    for i in range(len(daily.get("time", []))):
        lines.append(
            f"- {daily['time'][i]}: "
            f"min {daily['temperature_2m_min'][i]}C, "
            f"max {daily['temperature_2m_max'][i]}C, "
            f"rain {daily['precipitation_sum'][i]} mm, "
            f"wind {daily['wind_speed_10m_max'][i]} km/h"
        )

    return "\n".join(lines)


async def compare_weather(city_a: str, city_b: str) -> str:
    a = await get_current_weather(city_a)
    b = await get_current_weather(city_b)
    return a + "\n\n" + b


# Tool registry
TOOL_REGISTRY = {
    "get_current_weather": get_current_weather,
    "get_forecast": get_forecast,
    "compare_weather": compare_weather,
}

# Gemini function declarations (tool schema)
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_current_weather",
        description=(
            "Get live current weather for a city: temperature, "
            "humidity, wind, precipitation, cloud cover."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Chennai"},
            },
            "required": ["city"],
        },
    ),
    types.FunctionDeclaration(
        name="get_forecast",
        description="Get daily weather forecast for up to 7 days for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {
                    "type": "integer",
                    "description": "Number of days 1 to 7, default 3",
                },
            },
            "required": ["city"],
        },
    ),
    types.FunctionDeclaration(
        name="compare_weather",
        description="Compare current weather between two cities side by side.",
        parameters={
            "type": "object",
            "properties": {
                "city_a": {"type": "string", "description": "First city"},
                "city_b": {"type": "string", "description": "Second city"},
            },
            "required": ["city_a", "city_b"],
        },
    ),
]


# ------------------------------------------------------------
# TTS UTILITIES (copied/adapted from Task4RAG-CAG)
# ------------------------------------------------------------

async def _edge_async(text: str, voice="en-US-AriaNeural", rate=None):
    if not edge_tts:
        return None
    kwargs = {"text": text, "voice": voice}
    if rate:
        r = rate.strip()
        if r == "0%" or r == "0":
            r = "+0%"
        elif not r.startswith(("+", "-")):
            r = f"+{r}"
        kwargs["rate"] = r
    comm = edge_tts.Communicate(**kwargs)
    out = io.BytesIO()
    async for chunk in comm.stream():
        if chunk[2]:
            out.write(chunk[2])
    return out.getvalue()


def tts_edge(text: str, voice="en-US-AriaNeural", rate=None):
    try:
        return asyncio.run(_edge_async(text, voice, rate)), None
    except Exception as e:
        return None, str(e)


def tts_gtts(text: str, lang="en"):
    if not gTTS:
        return None, "gTTS not available."
    try:
        buf = io.BytesIO()
        gTTS(text, lang=lang).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e:
        return None, str(e)


def synthesize(text: str, engine: str, lang_code="en"):
    """Returns (audio_bytes, mime, error)"""
    edge_voice_map = {
        "en": "en-US-AriaNeural",
        "hi": "hi-IN-SwaraNeural",
        "ta": "ta-IN-PallaviNeural",
        "bn": "bn-IN-BashkarNeural",
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "ar": "ar-SA-HamedNeural",
        "zh-Hans": "zh-CN-XiaoxiaoNeural",
        "zh-cn": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "pt": "pt-PT-FernandaNeural",
        "it": "it-IT-ElsaNeural",
        "nl": "nl-NL-ColetteNeural",
        "tr": "tr-TR-AhmetNeural",
        "ru": "ru-RU-DariyaNeural",
    }

    if engine == "Edge-TTS":
        voice = edge_voice_map.get(lang_code, "en-US-AriaNeural")
        audio, err = tts_edge(text, voice=voice, rate="+0%")
        if audio:
            return audio, "audio/mp3", None
        # fallback to gTTS
        audio2, err2 = tts_gtts(text, lang=lang_code if lang_code else "en")
        return audio2, "audio/mp3", err or err2

    if engine == "gTTS":
        audio, err = tts_gtts(text, lang=lang_code if lang_code else "en")
        return audio, "audio/mp3", err

    return None, None, "Unknown engine"


# ------------------------------------------------------------
# GEMINI AGENT
# ------------------------------------------------------------

class GeminiWeatherAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def run(self, user_prompt: str) -> Dict[str, Any]:
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

            # If no tools requested, return final answer
            if not function_calls:
                final_text = response.text or "No response generated."
                return {"answer": final_text, "trace": trace}

            # Add model message so far
            contents.append(candidate.content)

            # Execute tools
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

            # Send tool results back to Gemini
            contents.append(
                types.Content(role="tool", parts=function_response_parts)
            )

        return {"answer": "Reached tool-call limit.", "trace": trace}


# ------------------------------------------------------------
# LANGGRAPH SETUP
# ------------------------------------------------------------

class AgentState(TypedDict):
    user_input: str
    answer: str
    trace: List[Dict[str, Any]]


def build_langgraph(agent: GeminiWeatherAgent):
    graph = StateGraph(AgentState)

    async def weather_node(state: AgentState) -> AgentState:
        result = await agent.run(state["user_input"])
        return {
            "user_input": state["user_input"],
            "answer": result["answer"],
            "trace": result["trace"],
        }

    graph.add_node("weather_agent", weather_node)
    graph.set_entry_point("weather_agent")
    graph.add_edge("weather_agent", END)

    return graph.compile()


# ------------------------------------------------------------
# STREAMLIT HELPERS
# ------------------------------------------------------------

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
    st.sidebar.markdown("- **get_current_weather** — Live weather for any city")
    st.sidebar.markdown("- **get_forecast** — Up to 7-day daily forecast")
    st.sidebar.markdown("- **compare_weather** — Side-by-side city comparison")
    st.sidebar.markdown("---")
    st.sidebar.caption("Weather data: Open-Meteo (free, no key needed)")
    st.sidebar.caption("LangGraph orchestrates the agent loop")

    # Voice options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Response Options")
    st.session_state.voice_mode = st.sidebar.checkbox("Enable voice reply", value=False)
    st.session_state.tts_engine = st.sidebar.selectbox(
        "TTS engine",
        ["Edge-TTS", "gTTS"],
        index=0,
    )
    # simple language code for TTS
    st.session_state.tts_lang = st.sidebar.selectbox(
        "Voice language",
        ["en", "ta", "hi"],
        index=0,
        help="TTS language code (used for voice)",
    )


# ------------------------------------------------------------
# MAIN STREAMLIT APP
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Weather MCP with Gemini",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ Weather MCP Dashboard — Gemini + LangGraph")
    st.caption(
        "Gemini 2.5 Flash uses weather tools (Open-Meteo) via function calling. "
        "LangGraph coordinates the reasoning loop and tools. "
        "Streamlit shows the final answer, tool execution trace, and optional voice reply."
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
            value=(
                "Give me current weather in Chennai "
                "and a short practical summary."
            ),
        )
        run = st.button("Run weather task", use_container_width=True, type="primary")

        st.markdown("**Try these examples:**")
        examples = [
            "Current weather in Chennai with humidity and wind.",
            "3-day forecast for Madurai with clothing advice.",
            "Compare Chennai and Coimbatore weather today.",
            "Weather in Bengaluru with a travel recommendation.",
            "5-day forecast for Delhi — should I carry an umbrella?",
        ]
        for item in examples:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Performance Trace")
        trace_box = st.container(border=True)
        answer_box = st.container(border=True)

    if run and prompt.strip():
        with st.spinner("LangGraph + Gemini agent is working..."):
            state: AgentState = {
                "user_input": prompt.strip(),
                "answer": "",
                "trace": [],
            }
            result_state = loop.run_until_complete(graph.ainvoke(state))

        with answer_box:
            st.markdown("### Final Answer")
            st.write(result_state["answer"])

            # Voice reply
            if (
                getattr(st.session_state, "voice_mode", False)
                and result_state["answer"].strip()
            ):
                with st.spinner("Synthesizing voice reply..."):
                    audio, mime, err = synthesize(
                        result_state["answer"],
                        st.session_state.tts_engine,
                        st.session_state.tts_lang,
                    )
                if audio:
                    st.markdown("#### Audio Reply")
                    st.audio(io.BytesIO(audio), format=mime)
                elif err:
                    st.warning(f"Voice reply failed: {err}")

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
                        key=f"trace_{i}",
                    )

    st.divider()
    st.subheader("What this project demonstrates")
    st.markdown(
        """
- Gemini 2.5 Flash function calling with weather tools  
- LangGraph agent orchestration in a single node  
- Open-Meteo APIs for live weather and forecast (no key)  
- Streamlit Cloud–friendly single-file deployment  
- Full visibility into tool calls and results  
- Optional voice reply using Edge-TTS or gTTS
        """
    )


if __name__ == "__main__":
    main()
