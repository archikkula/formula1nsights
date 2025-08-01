# build race_weather.py
import json
import pandas as pd
import requests
import time
from datetime import datetime, timezone

with open("data/circuits.json") as f:
    circuits = json.load(f)

ERGAST = "https://api.jolpi.ca/ergast/f1"
START_YEAR = 2024
END_YEAR = datetime.now(timezone.utc).year
today = datetime.now(timezone.utc).date()

weather_records = []

for season in range(START_YEAR, END_YEAR + 1):
    try:
        sched = requests.get(f"{ERGAST}/{season}/races.json", timeout=10)
        sched.raise_for_status()
        races = sched.json()["MRData"]["RaceTable"]["Races"]
    except Exception:
        continue

    for race in races:
        rnd      = int(race["round"])
        cid      = race["Circuit"]["circuitId"]
        date_str = race.get("date")
        if not date_str or cid not in circuits:
            continue

        rd = datetime.fromisoformat(date_str).date()
        lat = circuits[cid]["lat"]
        lon = circuits[cid]["lng"]
        days_ahead = (rd - today).days

        if rd <= today:
            url = "https://archive-api.open-meteo.com/v1/archive"
        elif 0 < days_ahead <= 7:
            url = "https://api.open-meteo.com/v1/forecast"
        else:
            continue

        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": date_str,
            "end_date":   date_str,
            "daily":      "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone":   "UTC"
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            tmax_list = daily.get("temperature_2m_max") or []
            tmin_list = daily.get("temperature_2m_min") or []
            rain_list = daily.get("precipitation_sum") or []

            if not (tmax_list and tmin_list and rain_list):
                continue

            tmax = tmax_list[0]
            tmin = tmin_list[0]
            rain = rain_list[0]
            avg_temp = (tmax + tmin) / 2

            weather_records.append({
                "season":    season,
                "round":     rnd,
                "race_date": date_str,
                "avg_temp":  avg_temp,
                "precip_mm": rain
            })
        except Exception:
            continue

        time.sleep(1)

pd.DataFrame(weather_records).to_csv("data/race_weather.csv", index=False)
