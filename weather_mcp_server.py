from typing import Any, Dict

import httpx
from mcp.server.fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("weather-server")

USER_AGENT = "weather-mcp-streamlit/1.0"


async def geocode_city(city: str) -> Dict[str, Any]:
    """
    Resolve a city name to coordinates using Open-Meteo's geocoding API.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }
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


@mcp.tool()
async def get_current_weather(city: str) -> str:
    """
    Get current weather for a city (temperature, humidity, wind, etc.).
    """
    location = await geocode_city(city)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "is_day,precipitation,rain,showers,weather_code,cloud_cover,"
            "wind_speed_10m"
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


@mcp.tool()
async def get_forecast(city: str, days: int = 3) -> str:
    """
    Get daily forecast (min/max temp, precipitation, wind) for up to 7 days.
    """
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
    lines = [
        f"Forecast for {location['name']}, {location['country']} ({days} days):"
    ]

    for i in range(len(daily.get("time", []))):
        lines.append(
            f"- {daily['time'][i]}: "
            f"min {daily['temperature_2m_min'][i]}°C, "
            f"max {daily['temperature_2m_max'][i]}°C, "
            f"precipitation {daily['precipitation_sum'][i]} mm, "
            f"wind up to {daily['wind_speed_10m_max'][i]} km/h, "
            f"weather code {daily['weather_code'][i]}"
        )

    return "\n".join(lines)


@mcp.tool()
async def compare_weather(city_a: str, city_b: str) -> str:
    """
    Compare current weather between two cities.
    """
    first = await get_current_weather(city_a)
    second = await get_current_weather(city_b)
    return first + "\n" + second


if __name__ == "__main__":
    mcp.run()
