import requests
import csv
from datetime import datetime

YEAR     = datetime.now().year
ERG_BASE = "https://api.jolpi.ca/ergast/f1"
SCHED_OUT = f"data/schedule_{YEAR}.csv"


# 1) Scrape the race schedule for the current season via your Ergast proxy
resp = requests.get(f"{ERG_BASE}/{YEAR}.json", timeout=10)
resp.raise_for_status()
races = resp.json().get("MRData", {}) \
                   .get("RaceTable", {}) \
                   .get("Races", [])

with open(SCHED_OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["season","round","circuitId","race_date"])
    writer.writeheader()
    for race in races:
        writer.writerow({
            "season":    YEAR,
            "round":     int(race["round"]),
            "circuitId": race["Circuit"]["circuitId"],
            "race_date": race["date"]
        })
print(f"✅ Wrote {len(races)} rows to {SCHED_OUT}")


