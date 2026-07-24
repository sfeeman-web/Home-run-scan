
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

MLB_STATS = "https://statsapi.mlb.com/api/v1"
LIVE_FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

SAVANT_CSV = (
    "https://baseballsavant.mlb.com/statcast_search/csv?"
    "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
    "hfGT=R%7CPO%7CS%7C=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&"
    "opponent=&pitcher_throws=&batter_stands=&hfSA=&"
    "game_date_gt={start_dt}&game_date_lt={end_dt}&team=&position=&hfRO=&"
    "home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&"
    "group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&"
    "sort_order=desc&min_abs=0&type=details"
)

BIP_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "force_out",
    "grounded_into_double_play", "field_error", "double_play",
    "fielders_choice", "fielders_choice_out", "sac_fly", "sac_bunt",
    "triple_play"
}

PITCH_GROUPS = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curve",
    "KC": "Knuckle curve", "CH": "Changeup", "FS": "Splitter",
    "SV": "Slurve", "KN": "Knuckleball", "EP": "Eephus"
}

PARK_FACTORS = {
    "COL": 1.24, "CIN": 1.15, "PHI": 1.12, "NYY": 1.11, "BOS": 1.10,
    "ATL": 1.07, "MIL": 1.06, "HOU": 1.05, "ARI": 1.05, "CHC": 1.04,
    "LAA": 1.02, "KC": 1.00, "TEX": 1.00, "WSH": 0.99, "BAL": 0.99,
    "LAD": 0.98, "SD": 0.97, "STL": 0.97, "DET": 0.96, "PIT": 0.96,
    "SF": 0.94, "SEA": 0.90, "OAK": 0.96, "ATH": 0.96, "CLE": 0.95,
    "MIN": 0.98, "CWS": 1.00, "NYM": 0.97, "MIA": 0.94, "TB": 0.96,
    "TOR": 1.02
}


# Stadium coordinates and approximate home-plate-to-center-field bearings.
# Bearings are used only to translate forecast wind into an outfield component.
STADIUM_WEATHER = {
    "ARI": (33.4453, -112.0667, 0.0, "retractable"),
    "ATH": (38.0456, -122.5111, 40.0, "outdoor"),
    "OAK": (38.0456, -122.5111, 40.0, "outdoor"),
    "ATL": (33.8908, -84.4677, 20.0, "outdoor"),
    "BAL": (39.2838, -76.6217, 45.0, "outdoor"),
    "BOS": (42.3467, -71.0972, 55.0, "outdoor"),
    "CHC": (41.9484, -87.6553, 30.0, "outdoor"),
    "CWS": (41.8300, -87.6338, 15.0, "outdoor"),
    "CIN": (39.0979, -84.5082, 55.0, "outdoor"),
    "CLE": (41.4962, -81.6852, 15.0, "outdoor"),
    "COL": (39.7559, -104.9942, 20.0, "outdoor"),
    "DET": (42.3390, -83.0485, 10.0, "outdoor"),
    "HOU": (29.7573, -95.3555, 25.0, "retractable"),
    "KC": (39.0517, -94.4803, 45.0, "outdoor"),
    "LAA": (33.8003, -117.8827, 20.0, "outdoor"),
    "LAD": (34.0739, -118.2400, 25.0, "outdoor"),
    "MIA": (25.7781, -80.2197, 15.0, "retractable"),
    "MIL": (43.0280, -87.9712, 20.0, "retractable"),
    "MIN": (44.9817, -93.2776, 45.0, "outdoor"),
    "NYM": (40.7571, -73.8458, 45.0, "outdoor"),
    "NYY": (40.8296, -73.9262, 65.0, "outdoor"),
    "PHI": (39.9061, -75.1665, 15.0, "outdoor"),
    "PIT": (40.4469, -80.0057, 25.0, "outdoor"),
    "SD": (32.7076, -117.1570, 20.0, "outdoor"),
    "SEA": (47.5914, -122.3325, 35.0, "retractable"),
    "SF": (37.7786, -122.3893, 60.0, "outdoor"),
    "STL": (38.6226, -90.1928, 15.0, "outdoor"),
    "TB": (27.7683, -82.6534, 0.0, "dome"),
    "TEX": (32.7473, -97.0848, 20.0, "retractable"),
    "TOR": (43.6414, -79.3894, 25.0, "retractable"),
    "WSH": (38.8730, -77.0074, 25.0, "outdoor"),
}

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def fetch_game_weather(game: GameContext) -> dict[str, Any]:
    """Return forecast nearest first pitch. Neutral fallback on any failure."""
    base = {
        "game_time_utc": game.game_date,
        "Weather_Factor": 1.0,
        "weather_source": "Neutral fallback",
        "roof_status": "",
        "temp_f": np.nan,
        "humidity_pct": np.nan,
        "wind_speed_mph": np.nan,
        "wind_direction_deg": np.nan,
        "wind_out_mph": np.nan,
        "precip_probability_pct": np.nan,
        "weather_warning": "",
    }
    stadium = STADIUM_WEATHER.get(game.home)
    if not stadium:
        base["weather_warning"] = "Stadium weather mapping unavailable"
        return base

    lat, lon, cf_bearing, roof_type = stadium
    if roof_type == "dome":
        base.update({
            "weather_source": "Indoor park",
            "roof_status": "Closed/dome",
            "weather_warning": "Indoor conditions",
        })
        return base
    if roof_type == "retractable":
        base["roof_status"] = "Unknown retractable"

    try:
        payload = get_json(OPEN_METEO, {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m", "relative_humidity_2m",
                "precipitation_probability", "wind_speed_10m",
                "wind_direction_10m"
            ]),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
            "forecast_days": 3,
        })
        hourly = payload.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
        if len(times) == 0:
            raise RuntimeError("No hourly forecast returned")
        first_pitch = pd.to_datetime(game.game_date, utc=True)
        idx = int(np.argmin(np.abs((times - first_pitch).total_seconds())))
        temp = float(hourly["temperature_2m"][idx])
        humidity = float(hourly["relative_humidity_2m"][idx])
        precip = float(hourly["precipitation_probability"][idx])
        wind_speed = float(hourly["wind_speed_10m"][idx])
        wind_from = float(hourly["wind_direction_10m"][idx])
        wind_toward = (wind_from + 180.0) % 360.0
        angle = math.radians(_angle_difference(wind_toward, cf_bearing))
        wind_out = wind_speed * math.cos(angle)

        # Retractable-roof weather stays neutral unless a manual input confirms open.
        if roof_type == "retractable":
            factor = 1.0
            warning = "ROOF STATUS UNCONFIRMED"
        else:
            temp_factor = float(np.clip(1.0 + (temp - 70.0) * 0.002, 0.95, 1.05))
            wind_factor = float(np.clip(1.0 + wind_out * 0.006, 0.93, 1.08))
            humidity_factor = float(np.clip(1.0 + (humidity - 50.0) * 0.0003, 0.985, 1.015))
            factor = float(np.clip(temp_factor * wind_factor * humidity_factor, 0.90, 1.12))
            if precip >= 70:
                warning = "HIGH DELAY/POSTPONEMENT RISK"
            elif precip >= 45:
                warning = "Meaningful rain/delay risk"
            elif precip >= 25:
                warning = "Monitor rain"
            else:
                warning = ""

        base.update({
            "Weather_Factor": factor,
            "weather_source": "Open-Meteo hourly forecast",
            "temp_f": temp,
            "humidity_pct": humidity,
            "wind_speed_mph": wind_speed,
            "wind_direction_deg": wind_from,
            "wind_out_mph": round(wind_out, 1),
            "precip_probability_pct": precip,
            "weather_warning": warning,
        })
        return base
    except Exception as exc:
        base["weather_warning"] = f"Weather unavailable: {type(exc).__name__}"
        return base


def game_weather_map(games: list[GameContext], enabled: bool = True) -> dict[int, dict[str, Any]]:
    if not enabled:
        return {g.game_pk: {
            "game_time_utc": g.game_date,
            "Weather_Factor": 1.0,
            "weather_source": "Automatic weather disabled",
            "roof_status": "",
            "temp_f": np.nan,
            "humidity_pct": np.nan,
            "wind_speed_mph": np.nan,
            "wind_direction_deg": np.nan,
            "wind_out_mph": np.nan,
            "precip_probability_pct": np.nan,
            "weather_warning": "",
        } for g in games}
    return {g.game_pk: fetch_game_weather(g) for g in games}


def apply_manual_environment(weather: dict[str, Any], rec: pd.Series) -> dict[str, Any]:
    """Manual CSV values override automatic fields when supplied."""
    out = weather.copy()
    mapping = {
        "weather_factor": "Weather_Factor",
        "roof_status": "roof_status",
        "temp_f": "temp_f",
        "humidity_pct": "humidity_pct",
        "wind_speed_mph": "wind_speed_mph",
        "wind_direction_deg": "wind_direction_deg",
        "wind_out_mph": "wind_out_mph",
        "precip_probability_pct": "precip_probability_pct",
        "weather_warning": "weather_warning",
    }
    changed = False
    for source, target in mapping.items():
        if source in rec.index and pd.notna(rec[source]) and str(rec[source]).strip() != "":
            value = rec[source]
            if target not in {"roof_status", "weather_warning"}:
                value = pd.to_numeric(value, errors="coerce")
                if pd.isna(value):
                    continue
            out[target] = value
            changed = True
    roof = str(out.get("roof_status", "")).strip().lower()
    if roof in {"closed", "closed/dome", "dome"}:
        out["Weather_Factor"] = 1.0
        out["wind_out_mph"] = 0.0
        out["weather_warning"] = "Roof closed"
    if changed:
        out["weather_source"] = "Automatic forecast + manual override"
    return out

DEFAULT_WEIGHTS = {
    "contact_quality": 0.30,
    "pitcher_vulnerability": 0.25,
    "pitch_mix_matchup": 0.15,
    "park_environment": 0.15,
    "regression_due": 0.10,
    "market_value": 0.05,
}


@dataclass
class GameContext:
    game_pk: int
    game_date: str
    away: str
    home: str
    venue: str
    venue_id: int | None
    status: str
    away_pitcher_id: int | None
    away_pitcher_name: str | None
    home_pitcher_id: int | None
    home_pitcher_name: str | None


def get_json(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed: {url}") from last_exc


def schedule_for_date(game_date: str) -> list[GameContext]:
    payload = get_json(
        f"{MLB_STATS}/schedule",
        {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team,venue",
        },
    )
    games: list[GameContext] = []
    for d in payload.get("dates", []):
        for g in d.get("games", []):
            teams = g["teams"]
            away_prob = teams["away"].get("probablePitcher") or {}
            home_prob = teams["home"].get("probablePitcher") or {}
            games.append(
                GameContext(
                    game_pk=g["gamePk"],
                    game_date=g["gameDate"],
                    away=teams["away"]["team"]["abbreviation"],
                    home=teams["home"]["team"]["abbreviation"],
                    venue=g.get("venue", {}).get("name", ""),
                    venue_id=g.get("venue", {}).get("id"),
                    status=g.get("status", {}).get("detailedState", ""),
                    away_pitcher_id=away_prob.get("id"),
                    away_pitcher_name=away_prob.get("fullName"),
                    home_pitcher_id=home_prob.get("id"),
                    home_pitcher_name=home_prob.get("fullName"),
                )
            )
    return games


def confirmed_lineup(game_pk: int) -> tuple[list[dict], list[dict]]:
    """Returns away, home lineups. Empty lists mean not yet posted."""
    payload = get_json(LIVE_FEED.format(game_pk=game_pk))
    box = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
    result = []
    for side in ("away", "home"):
        team_box = box.get(side, {})
        order = team_box.get("battingOrder") or []
        players = team_box.get("players") or {}
        lineup = []
        for slot, pid in enumerate(order, start=1):
            key = f"ID{pid}"
            p = players.get(key, {})
            person = p.get("person", {})
            position = p.get("position", {}).get("abbreviation")
            lineup.append({
                "player_id": int(pid),
                "player": person.get("fullName", str(pid)),
                "lineup_spot": slot,
                "position": position,
            })
        result.append(lineup)
    return result[0], result[1]


def fallback_roster(team_id: int, game_date: str) -> list[dict]:
    payload = get_json(
        f"{MLB_STATS}/teams/{team_id}/roster",
        {"rosterType": "active", "date": game_date},
    )
    return [
        {
            "player_id": int(x["person"]["id"]),
            "player": x["person"]["fullName"],
            "lineup_spot": np.nan,
            "position": x.get("position", {}).get("abbreviation"),
        }
        for x in payload.get("roster", [])
        if x.get("position", {}).get("type") != "Pitcher"
    ]


def team_id_map() -> dict[str, int]:
    payload = get_json(f"{MLB_STATS}/teams", {"sportId": 1})
    return {
        x["abbreviation"]: int(x["id"])
        for x in payload.get("teams", [])
    }


def _coerce_statcast_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "game_pk", "batter", "pitcher", "release_speed", "plate_x", "plate_z",
        "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y",
        "bat_score", "post_bat_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _fetch_savant_day(day: str, probable_pitcher_ids: set[int]) -> pd.DataFrame:
    """
    Fetch one day directly from Baseball Savant and immediately discard
    unnecessary pitch rows. We retain:
      - terminal plate-appearance rows,
      - all measured batted balls,
      - all pitches thrown by today's probable starters.
    """
    url = SAVANT_CSV.format(start_dt=day, end_dt=day)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
            "AppleWebKit/605.1.15 Safari/604.1"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    text = response.text
    if not text.strip():
        return pd.DataFrame()
    if text.lstrip().lower().startswith("<!doctype html") or "<html" in text[:300].lower():
        raise RuntimeError(
            f"Baseball Savant returned an HTML page instead of CSV for {day}."
        )

    daily = pd.read_csv(io.StringIO(text), low_memory=False)
    daily.columns = [str(c).strip() for c in daily.columns]

    needed = [
        "game_date", "game_pk", "batter", "pitcher", "stand", "events",
        "pitch_type", "release_speed", "plate_x", "plate_z", "description",
        "launch_speed", "launch_angle", "hit_distance_sc",
        "hc_x", "hc_y", "bat_score", "post_bat_score"
    ]
    for col in needed:
        if col not in daily.columns:
            daily[col] = np.nan
    daily = daily[needed].copy()
    daily = _coerce_statcast_types(daily)

    terminal = daily["events"].notna()
    batted_ball = daily["launch_speed"].notna() & daily["launch_angle"].notna()
    starter_pitch = daily["pitcher"].isin(probable_pitcher_ids)
    daily = daily.loc[terminal | batted_ball | starter_pitch].copy()
    return daily


def pull_statcast(
    start_dt: str,
    end_dt: str,
    probable_pitcher_ids: set[int] | None = None,
) -> pd.DataFrame:
    """
    Direct Baseball Savant CSV ingestion.

    The query is split into one-day requests because Savant caps large result
    sets. Each day is filtered before concatenation, keeping memory stable on
    Streamlit Community Cloud.
    """
    pitcher_ids = probable_pitcher_ids or set()
    days = pd.date_range(start=start_dt, end=end_dt, freq="D")
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for idx, timestamp in enumerate(days, start=1):
        day = timestamp.strftime("%Y-%m-%d")
        print(f"Statcast day {idx}/{len(days)}: {day}", flush=True)
        try:
            daily = _fetch_savant_day(day, pitcher_ids)
        except Exception as exc:
            failures.append(f"{day}: {type(exc).__name__}: {exc}")
            continue
        if not daily.empty:
            frames.append(daily)
        gc.collect()

    if not frames:
        detail = "\n".join(failures[-5:])
        raise RuntimeError(
            "Baseball Savant returned no usable Statcast rows."
            + (f"\nRecent request errors:\n{detail}" if detail else "")
        )

    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    if len(failures) > max(3, len(days) // 4):
        detail = "\n".join(failures[-8:])
        raise RuntimeError(
            f"Too many Baseball Savant date requests failed "
            f"({len(failures)} of {len(days)}).\n{detail}"
        )

    return combined


def batting_side(row: pd.Series) -> str:
    stand = str(row.get("stand", "")).upper()
    return stand if stand in {"L", "R"} else "U"


def is_bbe(df: pd.DataFrame) -> pd.Series:
    return (
        df["launch_speed"].notna()
        & df["launch_angle"].notna()
        & df["events"].fillna("").isin(BIP_EVENTS)
    )


def is_barrel_row(ev: float, la: float) -> bool:
    """
    Public barrel approximation using the MLB barrel window:
    minimum 98 mph; launch-angle window expands with EV.
    This is intentionally labeled an approximation.
    """
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return False
    low = max(8, 26 - (ev - 98))
    high = min(50, 30 + 2 * (ev - 98))
    return low <= la <= high


def expected_hr_proxy(ev: float, la: float, distance: float | None) -> float:
    """
    Transparent custom xHR proxy, not MLB's official xHR.
    Uses EV, launch angle, and projected distance when available.
    """
    if pd.isna(ev) or pd.isna(la):
        return 0.0
    ev_component = 1 / (1 + math.exp(-(ev - 101.5) / 3.0))
    angle_component = math.exp(-((la - 28.0) ** 2) / (2 * 10.0 ** 2))
    if distance is None or pd.isna(distance):
        dist_component = 0.45
    else:
        dist_component = 1 / (1 + math.exp(-(distance - 385.0) / 16.0))
    return float(np.clip(0.45 * ev_component + 0.30 * angle_component + 0.25 * dist_component, 0, 1))


def last_n_games_for_batter(df: pd.DataFrame, batter_id: int, n: int = 10) -> pd.DataFrame:
    p = df[df["batter"] == batter_id].copy()
    dates = sorted(p["game_date"].dropna().dt.normalize().unique(), reverse=True)[:n]
    return p[p["game_date"].dt.normalize().isin(dates)].copy()


def count_runs_rbi_hits(p: pd.DataFrame) -> dict[str, float]:
    pa = p[p["events"].notna()].copy()
    hits = pa["events"].isin(["single", "double", "triple", "home_run"]).sum()
    hrs = (pa["events"] == "home_run").sum()
    doubles = (pa["events"] == "double").sum()
    triples = (pa["events"] == "triple").sum()
    walks = pa["events"].isin(["walk", "intent_walk", "hit_by_pitch"]).sum()
    ab = (~pa["events"].isin([
        "walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "catcher_interf"
    ])).sum()
    avg = hits / ab if ab else np.nan
    tb = hits + doubles + 2 * triples + 3 * hrs
    rbi = pd.to_numeric(pa.get("post_bat_score", 0), errors="coerce").fillna(0).sub(
        pd.to_numeric(pa.get("bat_score", 0), errors="coerce").fillna(0)
    ).clip(lower=0).sum()
    # Runs scored cannot be perfectly reconstructed from batter-only rows.
    # Count from runner movements when available; otherwise leave blank.
    runs = np.nan
    return {
        "G": int(pa["game_date"].dt.normalize().nunique()),
        "PA": int(len(pa)),
        "AB": int(ab),
        "AVG": avg,
        "H": int(hits),
        "HR": int(hrs),
        "R": runs,
        "RBI": float(rbi),
        "TB": int(tb),
        "BB_HBP": int(walks),
    }


def batted_ball_metrics(p: pd.DataFrame) -> dict[str, float]:
    b = p[is_bbe(p)].copy()
    if b.empty:
        return {k: np.nan for k in [
            "BBE", "Avg_EV", "EV90", "Max_EV", "HH_95", "HH_pct",
            "EV_100_plus", "EV_100_plus_outs", "Barrels_approx", "Barrel_pct_approx",
            "Avg_LA", "SweetSpot", "SweetSpot_pct", "PullAir", "PullAir_pct",
            "Fly_350_plus", "Fly_375_plus", "Out_380_400", "Near_HR",
            "xHR_proxy", "xHR_minus_HR"
        ]}
    b["launch_speed"] = pd.to_numeric(b["launch_speed"], errors="coerce")
    b["launch_angle"] = pd.to_numeric(b["launch_angle"], errors="coerce")
    b["hit_distance_sc"] = pd.to_numeric(b.get("hit_distance_sc"), errors="coerce")
    b["is_hit"] = b["events"].isin(["single", "double", "triple", "home_run"])
    b["is_out"] = ~b["is_hit"]
    b["barrel_approx"] = [
        is_barrel_row(ev, la) for ev, la in zip(b["launch_speed"], b["launch_angle"])
    ]
    b["xhr_proxy"] = [
        expected_hr_proxy(ev, la, dist)
        for ev, la, dist in zip(b["launch_speed"], b["launch_angle"], b["hit_distance_sc"])
    ]
    sweet = b["launch_angle"].between(8, 32, inclusive="both")
    air = b["launch_angle"] >= 10
    pull = (
        ((b["stand"] == "R") & (b["hc_x"] < 125))
        | ((b["stand"] == "L") & (b["hc_x"] > 125))
    )
    pull_air = air & pull
    fly_350 = (b["launch_angle"] >= 15) & (b["hit_distance_sc"] >= 350)
    fly_375 = (b["launch_angle"] >= 15) & (b["hit_distance_sc"] >= 375)
    out_380_400 = b["is_out"] & b["hit_distance_sc"].between(380, 400, inclusive="both")
    near_hr = (
        (b["is_out"] & (b["hit_distance_sc"] >= 375))
        | (b["is_out"] & (b["launch_speed"] >= 100) & b["launch_angle"].between(18, 36))
    )
    actual_hr = int((b["events"] == "home_run").sum())
    return {
        "BBE": int(len(b)),
        "Avg_EV": float(b["launch_speed"].mean()),
        "EV90": float(b["launch_speed"].quantile(0.90)),
        "Max_EV": float(b["launch_speed"].max()),
        "HH_95": int((b["launch_speed"] >= 95).sum()),
        "HH_pct": float((b["launch_speed"] >= 95).mean()),
        "EV_100_plus": int((b["launch_speed"] >= 100).sum()),
        "EV_100_plus_outs": int(((b["launch_speed"] >= 100) & b["is_out"]).sum()),
        "Barrels_approx": int(b["barrel_approx"].sum()),
        "Barrel_pct_approx": float(b["barrel_approx"].mean()),
        "Avg_LA": float(b["launch_angle"].mean()),
        "SweetSpot": int(sweet.sum()),
        "SweetSpot_pct": float(sweet.mean()),
        "PullAir": int(pull_air.sum()),
        "PullAir_pct": float(pull_air.mean()),
        "Fly_350_plus": int(fly_350.sum()),
        "Fly_375_plus": int(fly_375.sum()),
        "Out_380_400": int(out_380_400.sum()),
        "Near_HR": int(near_hr.sum()),
        "xHR_proxy": float(b["xhr_proxy"].sum()),
        "xHR_minus_HR": float(b["xhr_proxy"].sum() - actual_hr),
    }


def _empty_pitcher_metrics() -> dict[str, float | str]:
    return {
        "Pitcher_BBE": np.nan,
        "Pitcher_BBE_Overall": np.nan,
        "Pitcher_HR": np.nan,
        "Pitcher_HR_Overall": np.nan,
        "Pitcher_HR_pct": np.nan,
        "Pitcher_HR_pct_Overall": np.nan,
        "Pitcher_HH_pct": np.nan,
        "Pitcher_HH_pct_Overall": np.nan,
        "Pitcher_Barrel_pct_approx": np.nan,
        "Pitcher_Barrel_pct_Overall": np.nan,
        "Pitcher_Avg_EV": np.nan,
        "Pitcher_Avg_EV_Overall": np.nan,
        "Pitcher_FB_pct": np.nan,
        "Pitcher_PullAir_Damage_pct": np.nan,
        "Pitcher_Top_Pitches": "",
        "Pitcher_HR_Pitch_Types": "",
        "Pitcher_Primary_Velo": np.nan,
    }


def _pitcher_bbe_summary(rows: pd.DataFrame) -> dict[str, float]:
    b = rows[is_bbe(rows)].copy()
    if b.empty:
        return {
            "BBE": 0, "HR": 0, "HR_pct": np.nan, "HH_pct": np.nan,
            "Barrel_pct": np.nan, "Avg_EV": np.nan, "FB_pct": np.nan,
            "PullAir_Damage_pct": np.nan,
        }
    ev = pd.to_numeric(b["launch_speed"], errors="coerce")
    la = pd.to_numeric(b["launch_angle"], errors="coerce")
    b["barrel_approx"] = [is_barrel_row(x, y) for x, y in zip(ev, la)]
    air = la >= 10
    pull = (
        ((b["stand"] == "R") & (pd.to_numeric(b["hc_x"], errors="coerce") < 125))
        | ((b["stand"] == "L") & (pd.to_numeric(b["hc_x"], errors="coerce") > 125))
    )
    damaging_pull_air = air & pull & ((ev >= 95) | (b["events"] == "home_run"))
    return {
        "BBE": int(len(b)),
        "HR": int((b["events"] == "home_run").sum()),
        "HR_pct": float((b["events"] == "home_run").mean()),
        "HH_pct": float((ev >= 95).mean()),
        "Barrel_pct": float(b["barrel_approx"].mean()),
        "Avg_EV": float(ev.mean()),
        "FB_pct": float((la >= 20).mean()),
        "PullAir_Damage_pct": float(damaging_pull_air.mean()),
    }


def pitcher_vulnerability(df: pd.DataFrame, pitcher_id: int | None, batter_stand: str) -> dict[str, float | str]:
    """Recent pitcher damage, both overall and versus the hitter's batting side."""
    if not pitcher_id:
        return _empty_pitcher_metrics()
    p_all = df[df["pitcher"] == pitcher_id].copy()
    if p_all.empty:
        return _empty_pitcher_metrics()
    p_side = p_all[p_all["stand"] == batter_stand].copy() if batter_stand in {"L", "R"} else p_all
    overall = _pitcher_bbe_summary(p_all)
    side = _pitcher_bbe_summary(p_side)

    pitch_usage = p_all["pitch_type"].dropna().value_counts(normalize=True).head(4)
    top_pitches = ", ".join(
        f"{PITCH_GROUPS.get(pt, pt)} {usage:.0%}" for pt, usage in pitch_usage.items()
    )
    hr_rows = p_side[p_side["events"] == "home_run"]
    hr_types = hr_rows["pitch_type"].dropna().value_counts()
    hr_pitch_types = ", ".join(
        f"{PITCH_GROUPS.get(pt, pt)} {int(count)}" for pt, count in hr_types.head(4).items()
    )
    primary_types = set(pitch_usage.head(2).index)
    velo_rows = p_all[p_all["pitch_type"].isin(primary_types)] if primary_types else p_all
    primary_velo = pd.to_numeric(velo_rows.get("release_speed"), errors="coerce").mean()

    return {
        "Pitcher_BBE": side["BBE"],
        "Pitcher_BBE_Overall": overall["BBE"],
        "Pitcher_HR": side["HR"],
        "Pitcher_HR_Overall": overall["HR"],
        "Pitcher_HR_pct": side["HR_pct"],
        "Pitcher_HR_pct_Overall": overall["HR_pct"],
        "Pitcher_HH_pct": side["HH_pct"],
        "Pitcher_HH_pct_Overall": overall["HH_pct"],
        "Pitcher_Barrel_pct_approx": side["Barrel_pct"],
        "Pitcher_Barrel_pct_Overall": overall["Barrel_pct"],
        "Pitcher_Avg_EV": side["Avg_EV"],
        "Pitcher_Avg_EV_Overall": overall["Avg_EV"],
        "Pitcher_FB_pct": side["FB_pct"],
        "Pitcher_PullAir_Damage_pct": side["PullAir_Damage_pct"],
        "Pitcher_Top_Pitches": top_pitches,
        "Pitcher_HR_Pitch_Types": hr_pitch_types,
        "Pitcher_Primary_Velo": float(primary_velo) if pd.notna(primary_velo) else np.nan,
    }


def _pitch_type_batter_score(rows: pd.DataFrame) -> tuple[float, int, float, float]:
    b = rows[is_bbe(rows)].copy()
    if b.empty:
        return 50.0, 0, np.nan, np.nan
    ev = pd.to_numeric(b["launch_speed"], errors="coerce")
    la = pd.to_numeric(b["launch_angle"], errors="coerce")
    barrels = np.array([is_barrel_row(x, y) for x, y in zip(ev, la)], dtype=float)
    hr_rate = float((b["events"] == "home_run").mean())
    hh = float((ev >= 95).mean())
    barrel = float(np.nanmean(barrels)) if len(barrels) else 0.0
    avg_ev = float(ev.mean())
    score = np.clip((avg_ev - 82) * 3.2 + hh * 28 + barrel * 45 + hr_rate * 70, 0, 100)
    return float(score), int(len(b)), avg_ev, barrel


def pitch_mix_matchup_details(batter_rows: pd.DataFrame, pitcher_rows: pd.DataFrame) -> dict[str, float | str]:
    """Match batter damage to the pitcher's actual recent pitch usage and velocity."""
    if pitcher_rows.empty:
        return {
            "Pitch_Mix_Score": 50.0, "Pitch_Type_Matchup": "No pitcher sample",
            "Velocity_Matchup_Score": 50.0, "Velocity_Matchup": "No velocity sample",
        }
    pitcher_usage = pitcher_rows["pitch_type"].dropna().value_counts(normalize=True).head(5)
    if pitcher_usage.empty:
        return {
            "Pitch_Mix_Score": 50.0, "Pitch_Type_Matchup": "No pitch mix",
            "Velocity_Matchup_Score": 50.0, "Velocity_Matchup": "No velocity sample",
        }

    weighted_scores, weights, detail_parts = [], [], []
    for pitch_type, usage in pitcher_usage.items():
        score, sample, avg_ev, barrel = _pitch_type_batter_score(
            batter_rows[batter_rows["pitch_type"] == pitch_type]
        )
        reliability = min(sample / 12.0, 1.0)
        adjusted = 50 + (score - 50) * reliability
        weighted_scores.append(adjusted)
        weights.append(float(usage))
        label = PITCH_GROUPS.get(pitch_type, pitch_type)
        detail_parts.append(f"{label} {usage:.0%}: {adjusted:.0f} ({sample} BBE)")
    pitch_score = float(np.average(weighted_scores, weights=weights))

    # Velocity compatibility: compare batter damage in the pitcher's primary velocity band.
    pitcher_velo = pd.to_numeric(pitcher_rows.get("release_speed"), errors="coerce").dropna()
    primary_velo = float(pitcher_velo.mean()) if not pitcher_velo.empty else np.nan
    if pd.isna(primary_velo):
        velo_score, velo_text = 50.0, "No velocity sample"
    else:
        batter_velo = pd.to_numeric(batter_rows.get("release_speed"), errors="coerce")
        band_rows = batter_rows[batter_velo.between(primary_velo - 1.5, primary_velo + 1.5)]
        raw_velo_score, sample, avg_ev, barrel = _pitch_type_batter_score(band_rows)
        reliability = min(sample / 15.0, 1.0)
        velo_score = float(50 + (raw_velo_score - 50) * reliability)
        velo_text = f"{primary_velo:.1f} mph band: {velo_score:.0f} ({sample} BBE)"

    # Preserve V3's 15% category, but improve its input by blending pitch type and velocity fit.
    combined = float(np.clip(pitch_score * 0.80 + velo_score * 0.20, 0, 100))
    return {
        "Pitch_Mix_Score": combined,
        "Pitch_Type_Matchup": " | ".join(detail_parts),
        "Velocity_Matchup_Score": velo_score,
        "Velocity_Matchup": velo_text,
    }


def pitch_mix_matchup_score(batter_rows: pd.DataFrame, pitcher_rows: pd.DataFrame) -> float:
    return float(pitch_mix_matchup_details(batter_rows, pitcher_rows)["Pitch_Mix_Score"])

def normalize_series(s: pd.Series, low: float = 0, high: float = 100) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() <= 1 or s.max() == s.min():
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    return low + (s - s.min()) * (high - low) / (s.max() - s.min())


def add_model_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    contact_raw = (
        out["Avg_EV"].fillna(out["Avg_EV"].median()) * 0.18
        + out["EV90"].fillna(out["EV90"].median()) * 0.16
        + out["Max_EV"].fillna(out["Max_EV"].median()) * 0.10
        + out["HH_pct"].fillna(0) * 100 * 0.16
        + out["Barrel_pct_approx"].fillna(0) * 100 * 0.16
        + out["PullAir_pct"].fillna(0) * 100 * 0.10
        + out["Fly_375_plus"].fillna(0) * 3.5
        + out["Near_HR"].fillna(0) * 2.5
    )
    pitcher_raw = (
        out["Pitcher_HR_pct"].fillna(out["Pitcher_HR_pct_Overall"]).fillna(0) * 100 * 0.30
        + out["Pitcher_HH_pct"].fillna(out["Pitcher_HH_pct_Overall"]).fillna(0) * 100 * 0.22
        + out["Pitcher_Barrel_pct_approx"].fillna(out["Pitcher_Barrel_pct_Overall"]).fillna(0) * 100 * 0.22
        + out["Pitcher_FB_pct"].fillna(0) * 100 * 0.10
        + out["Pitcher_PullAir_Damage_pct"].fillna(0) * 100 * 0.08
        + out["Pitcher_Avg_EV"].fillna(out["Pitcher_Avg_EV_Overall"]).fillna(85) * 0.08
    )
    due_raw = (
        out["xHR_minus_HR"].fillna(0) * 20
        + out["EV_100_plus_outs"].fillna(0) * 3
        + out["Out_380_400"].fillna(0) * 5
    )
    out["Contact_Score"] = normalize_series(contact_raw)
    out["Pitcher_Vuln_Score"] = normalize_series(pitcher_raw)
    out["Pitch_Mix_Score"] = out["Pitch_Mix_Score"].fillna(50).clip(0, 100)
    out["Park_Env_Score"] = normalize_series(out["Park_Factor"] * out["Weather_Factor"])
    out["Due_Score"] = normalize_series(due_raw)
    out["Market_Value_Score"] = out["Market_Value_Score"].fillna(50).clip(0, 100)

    out["Model_Score"] = (
        out["Contact_Score"] * weights["contact_quality"]
        + out["Pitcher_Vuln_Score"] * weights["pitcher_vulnerability"]
        + out["Pitch_Mix_Score"] * weights["pitch_mix_matchup"]
        + out["Park_Env_Score"] * weights["park_environment"]
        + out["Due_Score"] * weights["regression_due"]
        + out["Market_Value_Score"] * weights["market_value"]
    )
    out["Qualifying_Power_Signals"] = (
        (out["Barrels_approx"].fillna(0) >= 2).astype(int)
        + (out["Max_EV"].fillna(0) >= 105).astype(int)
        + (out["EV_100_plus"].fillna(0) >= 2).astype(int)
        + (out["Fly_375_plus"].fillna(0) >= 1).astype(int)
        + (out["xHR_minus_HR"].fillna(0) > 0.35).astype(int)
        + (out["PullAir_pct"].fillna(0) >= 0.20).astype(int)
    )
    out["Core_HR_Eligible"] = (
        (out["Qualifying_Power_Signals"] >= 2)
        & (out["lineup_spot"].fillna(9) <= 6)
        & (out["status"].str.lower().isin(["scheduled", "pre-game", "warmup"]))
    )
    return out.sort_values(["Model_Score", "Qualifying_Power_Signals"], ascending=False)


def apply_matchup_first_overlay(
    board: pd.DataFrame,
    individual_weight: float = 0.75,
    matchup_weight: float = 0.25,
) -> pd.DataFrame:
    """Blend the original V3 score with offense-specific matchup attackability."""
    out = board.copy()

    out["Individual_Model_Score"] = pd.to_numeric(
        out["Model_Score"], errors="coerce"
    ).fillna(50.0)

    pitcher_score = pd.to_numeric(
        out["Pitcher_Vuln_Score"], errors="coerce"
    ).fillna(50.0)
    environment_score = pd.to_numeric(
        out["Park_Env_Score"], errors="coerce"
    ).fillna(50.0)

    raw_attackability = pitcher_score * 0.70 + environment_score * 0.30

    matchup_keys = [
        c for c in ["game_pk", "team", "opponent", "opposing_pitcher_id"]
        if c in out.columns
    ]

    if matchup_keys:
        out["Game_Attackability_Score"] = raw_attackability.groupby(
            [out[c] for c in matchup_keys], dropna=False
        ).transform("mean")
    else:
        out["Game_Attackability_Score"] = raw_attackability

    out["Game_Attackability_Score"] = pd.to_numeric(
        out["Game_Attackability_Score"], errors="coerce"
    ).fillna(50.0).clip(0, 100)

    out["Model_Score"] = (
        out["Individual_Model_Score"] * individual_weight
        + out["Game_Attackability_Score"] * matchup_weight
    ).clip(0, 100)

    if matchup_keys:
        out["Matchup_Hitter_Rank"] = (
            out.groupby(matchup_keys, dropna=False)["Model_Score"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
    else:
        out["Matchup_Hitter_Rank"] = pd.Series(
            range(1, len(out) + 1), index=out.index, dtype="Int64"
        )

    lineup_spot = pd.to_numeric(out.get("lineup_spot"), errors="coerce").fillna(9)
    out["Matchup_Cluster_Pick"] = (
        (out["Game_Attackability_Score"] >= 60)
        & (out["Matchup_Hitter_Rank"] <= 2)
        & (lineup_spot <= 6)
    )

    out["Attackability_Grade"] = pd.cut(
        out["Game_Attackability_Score"],
        bins=[-1, 44.999, 54.999, 64.999, 74.999, 100],
        labels=["Avoid", "Neutral", "Attackable", "Strong", "Pinata"],
    ).astype(str)

    return out.sort_values(
        ["Model_Score", "Game_Attackability_Score", "Qualifying_Power_Signals"],
        ascending=[False, False, False],
    )


def load_optional_inputs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run(
    game_date: str,
    output_dir: Path,
    lookback_days: int = 32,
    include_unconfirmed: bool = False,
    auto_weather: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    games = schedule_for_date(game_date)
    ids = team_id_map()

    start = (pd.Timestamp(game_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = game_date
    probable_pitcher_ids = {
        int(pid)
        for game in games
        for pid in (game.away_pitcher_id, game.home_pitcher_id)
        if pid is not None
    }
    print(f"Pulling Statcast {start} through {end}...", flush=True)
    sc = pull_statcast(start, end, probable_pitcher_ids)
    if sc.empty:
        raise RuntimeError("No Statcast data returned.")

    # Use the same recent window for pitcher vulnerability and pitch-mix scoring.
    # This avoids a second season-wide download that can exceed free-host memory.
    season_sc = sc

    env = load_optional_inputs(Path("environment_inputs.csv"))
    weather_by_game = game_weather_map(games, enabled=auto_weather)
    odds = load_optional_inputs(Path("odds_inputs.csv"))
    weights = DEFAULT_WEIGHTS.copy()
    weight_path = Path("weights.json")
    if weight_path.exists():
        weights.update(json.loads(weight_path.read_text()))

    rows = []
    for game in games:
        try:
            away_lineup, home_lineup = confirmed_lineup(game.game_pk)
        except Exception:
            away_lineup, home_lineup = [], []

        if include_unconfirmed:
            if not away_lineup:
                away_lineup = fallback_roster(ids[game.away], game_date)
            if not home_lineup:
                home_lineup = fallback_roster(ids[game.home], game_date)

        for side, lineup, team, opp, pitcher_id, pitcher_name, park_team in [
            ("away", away_lineup, game.away, game.home, game.home_pitcher_id, game.home_pitcher_name, game.home),
            ("home", home_lineup, game.home, game.away, game.away_pitcher_id, game.away_pitcher_name, game.home),
        ]:
            for hitter in lineup:
                pid = hitter["player_id"]
                p10 = last_n_games_for_batter(sc, pid, 10)
                if p10.empty:
                    continue
                stand_mode = p10["stand"].dropna().mode()
                stand = stand_mode.iloc[0] if not stand_mode.empty else "U"
                production = count_runs_rbi_hits(p10)
                contact = batted_ball_metrics(p10)
                pv = pitcher_vulnerability(season_sc, pitcher_id, stand)
                batter_season = season_sc[season_sc["batter"] == pid]
                pitcher_season = season_sc[season_sc["pitcher"] == pitcher_id] if pitcher_id else pd.DataFrame()
                pitch_details = (
                    pitch_mix_matchup_details(batter_season, pitcher_season)
                    if pitcher_id else {
                        "Pitch_Mix_Score": 50.0,
                        "Pitch_Type_Matchup": "No confirmed pitcher",
                        "Velocity_Matchup_Score": 50.0,
                        "Velocity_Matchup": "No confirmed pitcher",
                    }
                )

                park_factor = PARK_FACTORS.get(park_team, 1.0)
                weather = weather_by_game.get(game.game_pk, {}).copy()
                if not env.empty and "game_pk" in env.columns:
                    m = env[env["game_pk"] == game.game_pk]
                    if not m.empty:
                        weather = apply_manual_environment(weather, m.iloc[0])
                weather_factor = float(weather.get("Weather_Factor", 1.0))

                hr_odds = np.nan
                market_value = 50.0
                if not odds.empty and "player_id" in odds.columns:
                    m = odds[odds["player_id"] == pid]
                    if not m.empty:
                        hr_odds = pd.to_numeric(m.iloc[0].get("hr_odds_american"), errors="coerce")
                        if pd.notna(hr_odds):
                            implied = 100 / (hr_odds + 100) if hr_odds > 0 else (-hr_odds) / ((-hr_odds) + 100)
                            # Initial market score: longer price is better, but capped.
                            market_value = float(np.clip((0.18 - implied) * 500 + 50, 0, 100))

                row = {
                    "game_pk": game.game_pk,
                    "game_date": game_date,
                    "status": game.status,
                    "team": team,
                    "opponent": opp,
                    "home_away": side,
                    "venue": game.venue,
                    "player_id": pid,
                    "player": hitter["player"],
                    "lineup_spot": hitter["lineup_spot"],
                    "position": hitter["position"],
                    "bat_side": stand,
                    "opposing_pitcher_id": pitcher_id,
                    "opposing_pitcher": pitcher_name,
                    "Park_Factor": park_factor,
                    "Weather_Factor": weather_factor,
                    "game_time_utc": weather.get("game_time_utc", game.game_date),
                    "weather_source": weather.get("weather_source", ""),
                    "roof_status": weather.get("roof_status", ""),
                    "temp_f": weather.get("temp_f", np.nan),
                    "humidity_pct": weather.get("humidity_pct", np.nan),
                    "wind_speed_mph": weather.get("wind_speed_mph", np.nan),
                    "wind_direction_deg": weather.get("wind_direction_deg", np.nan),
                    "wind_out_mph": weather.get("wind_out_mph", np.nan),
                    "precip_probability_pct": weather.get("precip_probability_pct", np.nan),
                    "weather_warning": weather.get("weather_warning", ""),
                    "HR_Odds_American": hr_odds,
                    "Market_Value_Score": market_value,
                    **pitch_details,
                    **production,
                    **contact,
                    **pv,
                }
                rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        raise RuntimeError(
            "No hitter rows were created. Lineups may not be posted yet. "
            "Run with --include-unconfirmed for active-roster screening."
        )
    board = add_model_scores(board, weights)
    board = apply_matchup_first_overlay(board)
    gc.collect()

    pct_cols = [
        "HH_pct", "Barrel_pct_approx", "SweetSpot_pct", "PullAir_pct",
        "Pitcher_HR_pct", "Pitcher_HR_pct_Overall",
        "Pitcher_HH_pct", "Pitcher_HH_pct_Overall",
        "Pitcher_Barrel_pct_approx", "Pitcher_Barrel_pct_Overall",
        "Pitcher_FB_pct", "Pitcher_PullAir_Damage_pct"
    ]
    for c in pct_cols:
        if c in board:
            board[c] = board[c] * 100

    ordered = [
        "Model_Score", "Individual_Model_Score", "Game_Attackability_Score",
        "Attackability_Grade", "Matchup_Hitter_Rank", "Matchup_Cluster_Pick",
        "Core_HR_Eligible", "Qualifying_Power_Signals",
        "player", "team", "opponent", "lineup_spot", "position", "bat_side",
        "opposing_pitcher", "G", "PA", "AB", "AVG", "H", "HR", "R", "RBI", "TB",
        "BBE", "Avg_EV", "EV90", "Max_EV", "HH_95", "HH_pct",
        "EV_100_plus", "EV_100_plus_outs", "Barrels_approx", "Barrel_pct_approx",
        "Avg_LA", "SweetSpot", "SweetSpot_pct", "PullAir", "PullAir_pct",
        "Fly_350_plus", "Fly_375_plus", "Out_380_400", "Near_HR",
        "xHR_proxy", "xHR_minus_HR",
        "Pitcher_BBE", "Pitcher_BBE_Overall", "Pitcher_HR", "Pitcher_HR_Overall",
        "Pitcher_HR_pct", "Pitcher_HR_pct_Overall", "Pitcher_HH_pct",
        "Pitcher_HH_pct_Overall", "Pitcher_Barrel_pct_approx",
        "Pitcher_Barrel_pct_Overall", "Pitcher_Avg_EV", "Pitcher_Avg_EV_Overall",
        "Pitcher_FB_pct", "Pitcher_PullAir_Damage_pct", "Pitcher_Top_Pitches",
        "Pitcher_HR_Pitch_Types", "Pitcher_Primary_Velo", "Pitch_Mix_Score",
        "Pitch_Type_Matchup", "Velocity_Matchup_Score", "Velocity_Matchup",
        "Park_Factor", "Weather_Factor", "game_time_utc",
        "temp_f", "humidity_pct", "wind_speed_mph", "wind_direction_deg",
        "wind_out_mph", "precip_probability_pct", "roof_status",
        "weather_warning", "weather_source",
        "HR_Odds_American", "Market_Value_Score",
        "Contact_Score", "Pitcher_Vuln_Score", "Park_Env_Score", "Due_Score",
        "status", "game_pk", "player_id", "opposing_pitcher_id", "venue",
    ]
    ordered = [c for c in ordered if c in board.columns]
    board = board[ordered]

    csv_path = output_dir / f"outlaw_scanner_{game_date}.csv"
    xlsx_path = output_dir / f"outlaw_scanner_{game_date}.xlsx"
    board.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        board.to_excel(writer, sheet_name="Ranked Board", index=False)
        clusters = board[board["Matchup_Cluster_Pick"] == True].head(40)
        clusters.to_excel(writer, sheet_name="Matchup Clusters", index=False)
        core = board[board["Core_HR_Eligible"] == True].head(30)
        core.to_excel(writer, sheet_name="Core HR", index=False)
        top40 = board.head(40)
        top40.to_excel(writer, sheet_name="Top 40", index=False)
        workbook = writer.book
        for sheet_name, frame in [
            ("Ranked Board", board),
            ("Matchup Clusters", clusters),
            ("Core HR", core),
            ("Top 40", top40),
        ]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 4)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            ws.set_row(0, 24)
            ws.set_column(0, len(frame.columns)-1, 12)
            for idx, col in enumerate(frame.columns):
                width = min(max(len(col) + 2, 11), 24)
                if col in {"player", "opposing_pitcher", "venue"}:
                    width = 22
                ws.set_column(idx, idx, width)
            if len(frame):
                score_col = frame.columns.get_loc("Model_Score")
                ws.conditional_format(1, score_col, len(frame), score_col, {
                    "type": "3_color_scale",
                    "min_color": "#F8696B",
                    "mid_color": "#FFEB84",
                    "max_color": "#63BE7B",
                })

    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")



def refresh_weather_only(game_date: str, output_dir: Path, auto_weather: bool = True) -> None:
    csv_path = output_dir / f"outlaw_scanner_{game_date}.csv"
    xlsx_path = output_dir / f"outlaw_scanner_{game_date}.xlsx"
    if not csv_path.exists():
        raise RuntimeError("Run a full scan before refreshing weather.")
    board = pd.read_csv(csv_path)
    games = schedule_for_date(game_date)
    weather_by_game = game_weather_map(games, enabled=auto_weather)
    env = load_optional_inputs(Path("environment_inputs.csv"))

    for game in games:
        weather = weather_by_game.get(game.game_pk, {}).copy()
        if not env.empty and "game_pk" in env.columns:
            m = env[env["game_pk"] == game.game_pk]
            if not m.empty:
                weather = apply_manual_environment(weather, m.iloc[0])
        mask = pd.to_numeric(board["game_pk"], errors="coerce") == game.game_pk
        for col, value in weather.items():
            board.loc[mask, col] = value

    old_env = pd.to_numeric(board.get("Park_Env_Score"), errors="coerce").fillna(50)
    board["Park_Env_Score"] = normalize_series(
        pd.to_numeric(board["Park_Factor"], errors="coerce").fillna(1.0)
        * pd.to_numeric(board["Weather_Factor"], errors="coerce").fillna(1.0)
    )
    weights = DEFAULT_WEIGHTS.copy()
    weight_path = Path("weights.json")
    if weight_path.exists():
        weights.update(json.loads(weight_path.read_text()))
    individual = pd.to_numeric(
        board.get("Individual_Model_Score", board.get("Model_Score")),
        errors="coerce",
    ).fillna(50)
    board["Model_Score"] = (
        individual
        - old_env * weights["park_environment"]
        + board["Park_Env_Score"] * weights["park_environment"]
    )
    board = apply_matchup_first_overlay(board)
    board.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        board.to_excel(writer, sheet_name="Ranked Board", index=False)
        board[board["Matchup_Cluster_Pick"] == True].head(40).to_excel(
            writer, sheet_name="Matchup Clusters", index=False
        )
        board[board["Core_HR_Eligible"] == True].head(30).to_excel(
            writer, sheet_name="Core HR", index=False
        )
        board.head(40).to_excel(writer, sheet_name="Top 40", index=False)
    print(f"Weather refreshed: {csv_path}")
    print(f"Weather refreshed: {xlsx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Last-10-game MLB HR scanner")
    parser.add_argument("--date", default=str(date.today()), help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--lookback-days", type=int, default=32)
    parser.add_argument("--weather-only", action="store_true")
    parser.add_argument("--no-auto-weather", action="store_true")
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Screen active rosters when confirmed lineups are unavailable.",
    )
    args = parser.parse_args()
    if args.weather_only:
        refresh_weather_only(
            game_date=args.date,
            output_dir=Path(args.output_dir),
            auto_weather=not args.no_auto_weather,
        )
    else:
        run(
            game_date=args.date,
            output_dir=Path(args.output_dir),
            lookback_days=args.lookback_days,
            include_unconfirmed=args.include_unconfirmed,
            auto_weather=not args.no_auto_weather,
        )


if __name__ == "__main__":
    main()
