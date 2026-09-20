"""Open-Meteo weather input for the transmission studies."""

from datetime import date, timedelta
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 48.8566
LONGITUDE = 2.3522
TIMEZONE = "Europe/Paris"


def _requested_date(selected_date=None):
    if selected_date:
        return pd.Timestamp(selected_date).date()

    # The forecast endpoint exposes recent observations through past_days.
    # Use yesterday so the default is a complete local calendar day.
    return date.today() - timedelta(days=1)


def load_open_meteo_weather(selected_date=None, past_days=7):
    """Return one complete Paris day from Open-Meteo's hourly forecast data."""
    target_date = _requested_date(selected_date)
    today = date.today()
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,wind_speed_100m,cloud_cover",
        "timezone": TIMEZONE,
    }

    if selected_date and target_date < today - timedelta(days=7):
        endpoint = OPEN_METEO_ARCHIVE_URL
        params["start_date"] = target_date.isoformat()
        params["end_date"] = target_date.isoformat()
    elif selected_date:
        endpoint = OPEN_METEO_URL
        params["past_days"] = 7
        params["forecast_days"] = 1
    else:
        endpoint = OPEN_METEO_URL
        params["past_days"] = past_days

    request_url = f"{endpoint}?{urlencode(params)}"
    try:
        with urlopen(request_url, timeout=30) as response:
            payload = json.load(response)
    except Exception as error:
        raise RuntimeError(f"Unable to retrieve Open-Meteo weather data: {error}") from error

    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo returned no hourly weather data: {payload}")

    hourly = payload["hourly"]
    weather = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_c": pd.to_numeric(hourly["temperature_2m"]),
        "cloud_cover": pd.to_numeric(hourly["cloud_cover"]),
        "wind_speed_100m": pd.to_numeric(hourly["wind_speed_100m"]),
    })
    selected_weather = weather[
        weather["datetime"].dt.date == target_date
    ].dropna().reset_index(drop=True)

    if len(selected_weather) != 24:
        raise ValueError(
            f"Open-Meteo returned {len(selected_weather)} hourly records for "
            f"{target_date}; expected 24."
        )
    return selected_weather


def prompt_for_weather_date():
    """Read an optional date from the terminal; blank means recent seven days."""
    entered_date = input(
        "Enter weather date YYYY-MM-DD (press Enter for the latest day in the past 7 days): "
    ).strip()
    if not entered_date:
        return None
    try:
        return pd.Timestamp(entered_date).date().isoformat()
    except ValueError as error:
        raise ValueError("Weather date must use YYYY-MM-DD format.") from error