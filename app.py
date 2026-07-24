
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Outlaw MLB Scanner",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    .stButton > button {width: 100%; min-height: 3rem; font-size: 1.05rem;}
    @media (max-width: 700px) {
        .block-container {padding-left: .65rem; padding-right: .65rem;}
        h1 {font-size: 1.7rem !important;}
        h2 {font-size: 1.3rem !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚾ Outlaw MLB Scanner")
st.caption("Direct-Savant scanner — v3.5 Hot Form + Best Matchup.")

with st.expander("Scanner model", expanded=False):
    st.markdown(
        """
        **Weights:** 30% contact quality, 25% pitcher vulnerability,
        15% pitch mix, 15% environment, 10% due indicators and 5% value.

        The mobile build uses a **32-day source window** to reconstruct each
        hitter's 10 most recent games. Version 3.2 keeps the original V3 weights
        while adding pitcher damage by batter side, pitch compatibility,
        automatic first-pitch weather, and a matchup-first overlay. The original
        V3 individual score remains 75% of the final grade, while offense-specific
        starter and environment attackability contributes 25%.
        """
    )

c1, c2 = st.columns(2)
with c1:
    scan_date = st.date_input("Slate date", value=date.today())
with c2:
    include_unconfirmed = st.toggle(
        "Morning roster scan",
        value=True,
        help="Use active rosters when confirmed batting orders are unavailable.",
    )

auto_weather = st.toggle(
    "Automatic first-pitch weather",
    value=True,
    help="Uses Open-Meteo hourly forecasts. Retractable roofs stay neutral until confirmed.",
)

st.subheader("Optional inputs")
st.caption("Weather is automatic. Upload a CSV only to override roof status or a forecast value.")

weather_upload = st.file_uploader(
    "Upload environment_inputs.csv",
    type=["csv"],
)
odds_upload = st.file_uploader(
    "Upload odds_inputs.csv",
    type=["csv"],
)

if weather_upload is not None:
    (APP_DIR / "environment_inputs.csv").write_bytes(weather_upload.getvalue())
    st.success("Weather inputs loaded.")

if odds_upload is not None:
    (APP_DIR / "odds_inputs.csv").write_bytes(odds_upload.getvalue())
    st.success("Odds inputs loaded.")

if "last_error" not in st.session_state:
    st.session_state.last_error = ""

refresh_col, scan_col = st.columns(2)
refresh_weather_clicked = refresh_col.button(
    "Refresh Weather Only",
    use_container_width=True,
    disabled=not (OUTPUT_DIR / f"outlaw_scanner_{scan_date.isoformat()}.csv").exists(),
)
run_full_clicked = scan_col.button(
    "Run Full Scan", type="primary", use_container_width=True
)

if refresh_weather_clicked:
    st.session_state.last_error = ""
    weather_cmd = [
        sys.executable, str(APP_DIR / "scanner.py"),
        "--date", scan_date.isoformat(),
        "--output-dir", str(OUTPUT_DIR),
        "--weather-only",
    ]
    if not auto_weather:
        weather_cmd.append("--no-auto-weather")
    with st.status("Refreshing first-pitch weather...", expanded=True) as status:
        process = subprocess.run(
            weather_cmd, cwd=APP_DIR, capture_output=True, text=True, timeout=90
        )
        if process.returncode != 0:
            status.update(label="Weather refresh failed", state="error")
            st.session_state.last_error = (process.stderr or process.stdout)[-6000:]
        else:
            status.update(label="Weather refreshed", state="complete")
            st.rerun()

if run_full_clicked:
    st.session_state.last_error = ""

    cmd = [
        sys.executable,
        str(APP_DIR / "scanner.py"),
        "--date",
        scan_date.isoformat(),
        "--output-dir",
        str(OUTPUT_DIR),
        "--lookback-days",
        "32",
    ]
    if include_unconfirmed:
        cmd.append("--include-unconfirmed")
    if not auto_weather:
        cmd.append("--no-auto-weather")

    child_env = os.environ.copy()
    child_env.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
        "PYTHONUNBUFFERED": "1",
    })

    with st.status("Running scanner...", expanded=True) as status:
        st.write("Downloading daily Baseball Savant CSV data...")
        st.write("Calculating last-10 contact and matchup metrics...")
        try:
            process = subprocess.run(
                cmd,
                cwd=APP_DIR,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=420,
            )
        except subprocess.TimeoutExpired:
            status.update(label="Scan timed out", state="error")
            st.session_state.last_error = (
                "The public Statcast download exceeded seven minutes. "
                "Wait a few minutes and run it again."
            )
        except Exception as exc:
            status.update(label="Scan failed", state="error")
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"
        else:
            if process.returncode != 0:
                status.update(label="Scan failed", state="error")
                details = (process.stderr or process.stdout or "Unknown scanner error").strip()
                st.session_state.last_error = details[-6000:]
            else:
                status.update(label="Scan complete", state="complete")
                if process.stdout:
                    st.code(process.stdout[-2500:])

if st.session_state.last_error:
    st.error("The scan did not complete.")
    st.code(st.session_state.last_error)

csv_path = OUTPUT_DIR / f"outlaw_scanner_{scan_date.isoformat()}.csv"
xlsx_path = OUTPUT_DIR / f"outlaw_scanner_{scan_date.isoformat()}.xlsx"

if csv_path.exists():
    try:
        board = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Results file could not be opened: {exc}")
    else:
        st.subheader("Top targets")
        m1, m2, m3 = st.columns(3)
        m1.metric("Players scanned", len(board))
        eligible = (
            int(board["Core_HR_Eligible"].fillna(False).sum())
            if "Core_HR_Eligible" in board else 0
        )
        m2.metric("Core HR eligible", eligible)
        top_score = board["Model_Score"].max() if "Model_Score" in board else float("nan")
        m3.metric("Top score", f"{top_score:.1f}" if pd.notna(top_score) else "—")

        preferred = [
            "Player_Display","Hot_Symbol","Platoon_Marker",
            "Matchup_Label","Best_Matchup","Best_Matchup_Score",
            "Model_Score","Individual_Model_Score",
            "Game_Attackability_Score","Attackability_Grade",
            "Matchup_Hitter_Rank","Matchup_Cluster_Pick",
            "Core_HR_Eligible","Qualifying_Power_Signals",
            "player","team","opponent","lineup_spot","opposing_pitcher",
            "AVG","H","HR","RBI","TB","Avg_EV","EV90","Max_EV",
            "HH_95","HH_pct","EV_100_plus","EV_100_plus_outs",
            "Barrels_approx","Barrel_pct_approx","Avg_LA",
            "SweetSpot_pct","PullAir_pct","Fly_350_plus","Fly_375_plus",
            "Out_380_400","Near_HR","xHR_proxy","xHR_minus_HR",
            "Pitcher_BBE","Pitcher_HR_pct","Pitcher_HR_pct_Overall",
            "Pitcher_HH_pct","Pitcher_Barrel_pct_approx","Pitcher_FB_pct",
            "Pitcher_PullAir_Damage_pct","Pitcher_Top_Pitches",
            "Pitcher_HR_Pitch_Types","Pitcher_Primary_Velo","Pitch_Mix_Score",
            "Pitch_Type_Matchup","Velocity_Matchup_Score","Velocity_Matchup",
            "Park_Factor","Weather_Factor","game_time_utc","temp_f",
            "humidity_pct","wind_speed_mph","wind_direction_deg",
            "wind_out_mph","precip_probability_pct","roof_status",
            "weather_warning","weather_source","HR_Odds_American"
        ]
        display_cols = [col for col in preferred if col in board.columns]


        st.caption(
            "🔥 = hot recent power/contact | * = platoon advantage | "
            "** = switch hitter | BEST MATCHUP = strongest hitter-pitcher fit in that offense"
        )

        tab1, tab2, tab3, tab4 = st.tabs(["Top 40", "Matchup Clusters", "Core HR", "Full board"])
        with tab1:
            st.dataframe(
                board[display_cols].head(40),
                use_container_width=True,
                hide_index=True,
                height=650,
            )
        with tab2:
            if "Matchup_Cluster_Pick" not in board:
                st.info("Matchup cluster field is unavailable.")
            else:
                clusters = board[board["Matchup_Cluster_Pick"] == True]
                if clusters.empty:
                    st.info("No offenses currently meet the matchup-cluster threshold.")
                else:
                    st.dataframe(
                        clusters[display_cols].head(40),
                        use_container_width=True,
                        hide_index=True,
                        height=650,
                    )
        with tab3:
            if "Core_HR_Eligible" not in board:
                st.info("Core-HR field is unavailable.")
            else:
                core = board[board["Core_HR_Eligible"] == True]
                if core.empty:
                    st.info("No hitters currently meet the full Core HR gate.")
                else:
                    st.dataframe(
                        core[display_cols].head(30),
                        use_container_width=True,
                        hide_index=True,
                        height=650,
                    )
        with tab4:
            st.dataframe(
                board[display_cols],
                use_container_width=True,
                hide_index=True,
                height=700,
            )

        st.subheader("Downloads")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                use_container_width=True,
            )
        if xlsx_path.exists():
            with d2:
                st.download_button(
                    "Download Excel",
                    data=xlsx_path.read_bytes(),
                    file_name=xlsx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
else:
    st.info("Choose the slate date and tap **Run Full Scan**.")
