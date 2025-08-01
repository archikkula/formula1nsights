# build race_history.csv with season,round,driver,circuitId,race_date,Q1,Q2,Q3,fastest_lap,finish_pos,circuit
import json
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import os

ERGAST = "https://api.jolpi.ca/ergast/f1"
DATA_PATH = "data/race_history.csv"

with open("data/circuits.json") as f:
    circuits_meta = json.load(f)

def fetch_all_races():
    new_records = []
    for year in range(2025, datetime.now().year + 2):
        sched_url = f"{ERGAST}/{year}/races.json"
        try:
            resp = requests.get(sched_url, timeout=10)
            resp.raise_for_status()
            races = resp.json()["MRData"]["RaceTable"]["Races"]
        except Exception:
            continue
        for race in races:
            rnd  = int(race["round"])
            cid  = race["Circuit"]["circuitId"]
            date = race.get("date")
            results_url = f"{ERGAST}/{year}/{rnd}/results.json"
            try:
                r = requests.get(results_url, timeout=10)
                r.raise_for_status()
                data = r.json()
                race_results = data["MRData"]["RaceTable"]["Races"][0]["Results"]
            except Exception:
                continue
            qual_results = []
            qual_url = f"{ERGAST}/{year}/{rnd}/qualifying.json"
            try:
                q = requests.get(qual_url, timeout=10)
                q.raise_for_status()
                qdata = q.json()
                qual_results = qdata["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]
            except Exception:
                pass
            for row in race_results:
                drv = row["Driver"]["driverId"]
                fl  = row.get("FastestLap", {})
                fastest = np.nan
                try:
                    t = fl["Time"]["time"]
                    m, s = t.split(":")
                    fastest = int(m) * 60 + float(s)
                except Exception:
                    pass
                qrow = next((q for q in qual_results if q["Driver"]["driverId"] == drv), {})
                Q1 = qrow.get("Q1", np.nan)
                Q2 = qrow.get("Q2", np.nan)
                Q3 = qrow.get("Q3", np.nan)
                new_records.append({
                    "season":      year,
                    "round":       rnd,
                    "circuitId":   cid,
                    "race_date":   date,
                    "driver":      drv,
                    "finish_pos":  int(row.get("position", 0)),
                    "fastest_lap": fastest,
                    "Q1":          Q1,
                    "Q2":          Q2,
                    "Q3":          Q3
                })

    # Load existing data if present
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH, parse_dates=["race_date"])
        combined = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
    else:
        combined = pd.DataFrame(new_records)

    # Deduplicate: keep the first occurrence when sorted by season, round, driver, fastest_lap descending
    combined = combined.sort_values(
        ["season", "round", "driver", "fastest_lap"],
        ascending=[True, True, True, False]
    )
    combined = combined.drop_duplicates(subset=["season", "round", "driver"], keep="first")

    # Filter out any “stub” rows without real result data
    # i.e., drop rows where fastest_lap is NaN or finish_pos is zero
    combined = combined.dropna(subset=["fastest_lap"])
    combined = combined[combined["finish_pos"] > 0]

    combined.to_csv(DATA_PATH, index=False)

if __name__ == "__main__":
    fetch_all_races()
