
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from pybaseball import statcast

MLB_STATS = "https://statsapi.mlb.com/api/v1"
LIVE_FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

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


def pull_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False, parallel=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [str(c) for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


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


def pitcher_vulnerability(df: pd.DataFrame, pitcher_id: int | None, batter_stand: str) -> dict[str, float]:
    if not pitcher_id:
        return {
            "Pitcher_BBE": np.nan, "Pitcher_HR": np.nan, "Pitcher_HR_pct": np.nan,
            "Pitcher_HH_pct": np.nan, "Pitcher_Barrel_pct_approx": np.nan,
            "Pitcher_Avg_EV": np.nan,
        }
    p = df[df["pitcher"] == pitcher_id].copy()
    if batter_stand in {"L", "R"}:
        p = p[p["stand"] == batter_stand]
    b = p[is_bbe(p)].copy()
    if b.empty:
        return {
            "Pitcher_BBE": 0, "Pitcher_HR": 0, "Pitcher_HR_pct": np.nan,
            "Pitcher_HH_pct": np.nan, "Pitcher_Barrel_pct_approx": np.nan,
            "Pitcher_Avg_EV": np.nan,
        }
    b["barrel_approx"] = [
        is_barrel_row(ev, la)
        for ev, la in zip(
            pd.to_numeric(b["launch_speed"], errors="coerce"),
            pd.to_numeric(b["launch_angle"], errors="coerce"),
        )
    ]
    return {
        "Pitcher_BBE": int(len(b)),
        "Pitcher_HR": int((b["events"] == "home_run").sum()),
        "Pitcher_HR_pct": float((b["events"] == "home_run").mean()),
        "Pitcher_HH_pct": float((pd.to_numeric(b["launch_speed"], errors="coerce") >= 95).mean()),
        "Pitcher_Barrel_pct_approx": float(b["barrel_approx"].mean()),
        "Pitcher_Avg_EV": float(pd.to_numeric(b["launch_speed"], errors="coerce").mean()),
    }


def pitch_mix_matchup_score(batter_rows: pd.DataFrame, pitcher_rows: pd.DataFrame) -> float:
    """
    Score 0-100 using batter damage on the pitcher's most-used pitch types.
    This is a transparent matchup score, not a proprietary projection.
    """
    pitcher_usage = pitcher_rows["pitch_type"].value_counts(normalize=True).head(4)
    if pitcher_usage.empty:
        return 50.0
    scores = []
    weights = []
    for pitch_type, usage in pitcher_usage.items():
        b = batter_rows[(batter_rows["pitch_type"] == pitch_type) & is_bbe(batter_rows)].copy()
        if b.empty:
            score = 50.0
        else:
            ev = pd.to_numeric(b["launch_speed"], errors="coerce").mean()
            hh = (pd.to_numeric(b["launch_speed"], errors="coerce") >= 95).mean()
            score = np.clip((ev - 82) * 4.0 + hh * 35, 0, 100)
        scores.append(score)
        weights.append(float(usage))
    return float(np.average(scores, weights=weights))


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
        out["Pitcher_HR_pct"].fillna(0) * 100 * 0.35
        + out["Pitcher_HH_pct"].fillna(0) * 100 * 0.30
        + out["Pitcher_Barrel_pct_approx"].fillna(0) * 100 * 0.25
        + out["Pitcher_Avg_EV"].fillna(85) * 0.10
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


def load_optional_inputs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run(game_date: str, output_dir: Path, lookback_days: int = 45, include_unconfirmed: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    games = schedule_for_date(game_date)
    ids = team_id_map()

    start = (pd.Timestamp(game_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = game_date
    print(f"Pulling Statcast {start} through {end}...")
    sc = pull_statcast(start, end)
    if sc.empty:
        raise RuntimeError("No Statcast data returned.")

    season_start = f"{pd.Timestamp(game_date).year}-03-01"
    # Reuse the same range if it begins before season start; otherwise pull season data.
    if pd.Timestamp(start) <= pd.Timestamp(season_start):
        season_sc = sc
    else:
        print(f"Pulling pitcher season Statcast {season_start} through {end}...")
        season_sc = pull_statcast(season_start, end)

    env = load_optional_inputs(Path("environment_inputs.csv"))
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
                pitch_score = pitch_mix_matchup_score(batter_season, pitcher_season) if pitcher_id else 50.0

                park_factor = PARK_FACTORS.get(park_team, 1.0)
                weather_factor = 1.0
                roof_status = ""
                temp_f = np.nan
                wind_out_mph = np.nan
                if not env.empty and "game_pk" in env.columns:
                    m = env[env["game_pk"] == game.game_pk]
                    if not m.empty:
                        rec = m.iloc[0]
                        weather_factor = float(rec.get("weather_factor", 1.0))
                        roof_status = rec.get("roof_status", "")
                        temp_f = rec.get("temp_f", np.nan)
                        wind_out_mph = rec.get("wind_out_mph", np.nan)

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
                    "roof_status": roof_status,
                    "temp_f": temp_f,
                    "wind_out_mph": wind_out_mph,
                    "HR_Odds_American": hr_odds,
                    "Market_Value_Score": market_value,
                    "Pitch_Mix_Score": pitch_score,
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

    pct_cols = [
        "HH_pct", "Barrel_pct_approx", "SweetSpot_pct", "PullAir_pct",
        "Pitcher_HR_pct", "Pitcher_HH_pct", "Pitcher_Barrel_pct_approx"
    ]
    for c in pct_cols:
        if c in board:
            board[c] = board[c] * 100

    ordered = [
        "Model_Score", "Core_HR_Eligible", "Qualifying_Power_Signals",
        "player", "team", "opponent", "lineup_spot", "position", "bat_side",
        "opposing_pitcher", "G", "PA", "AB", "AVG", "H", "HR", "R", "RBI", "TB",
        "BBE", "Avg_EV", "EV90", "Max_EV", "HH_95", "HH_pct",
        "EV_100_plus", "EV_100_plus_outs", "Barrels_approx", "Barrel_pct_approx",
        "Avg_LA", "SweetSpot", "SweetSpot_pct", "PullAir", "PullAir_pct",
        "Fly_350_plus", "Fly_375_plus", "Out_380_400", "Near_HR",
        "xHR_proxy", "xHR_minus_HR",
        "Pitcher_BBE", "Pitcher_HR", "Pitcher_HR_pct", "Pitcher_HH_pct",
        "Pitcher_Barrel_pct_approx", "Pitcher_Avg_EV",
        "Pitch_Mix_Score", "Park_Factor", "Weather_Factor",
        "temp_f", "wind_out_mph", "roof_status",
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
        core = board[board["Core_HR_Eligible"] == True].head(30)
        core.to_excel(writer, sheet_name="Core HR", index=False)
        top20 = board.head(20)
        top20.to_excel(writer, sheet_name="Top 20", index=False)
        workbook = writer.book
        for sheet_name, frame in [("Ranked Board", board), ("Core HR", core), ("Top 20", top20)]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Last-10-game MLB HR scanner")
    parser.add_argument("--date", default=str(date.today()), help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Screen active rosters when confirmed lineups are unavailable.",
    )
    args = parser.parse_args()
    run(
        game_date=args.date,
        output_dir=Path(args.output_dir),
        lookback_days=args.lookback_days,
        include_unconfirmed=args.include_unconfirmed,
    )


if __name__ == "__main__":
    main()
