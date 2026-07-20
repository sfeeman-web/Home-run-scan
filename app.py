
from __future__ import annotations

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
st.caption("Last-10-game Statcast scanner for HR, hit, run, RBI and total-base targeting.")

with st.expander("Scanner model", expanded=False):
    st.markdown(
        """
        **Weights**
        - 30% batter contact quality
        - 25% opposing-pitcher vulnerability
        - 15% pitch-mix compatibility
        - 15% park/environment
        - 10% regression/due indicators
        - 5% market value

        **Core HR gate**
        A hitter needs at least two qualifying power signals and must be in the top six of the lineup.
        """
    )

col1, col2 = st.columns(2)
with col1:
    scan_date = st.date_input("Slate date", value=date.today())
with col2:
    include_unconfirmed = st.toggle(
        "Morning roster scan",
        value=True,
        help="Use active-roster hitters before confirmed lineups are posted.",
    )

st.subheader("Optional inputs")
st.caption("Leave blank for neutral weather and market adjustments.")

weather_upload = st.file_uploader(
    "Upload environment_inputs.csv",
    type=["csv"],
    help="game_pk,temp_f,wind_out_mph,roof_status,weather_factor",
)
odds_upload = st.file_uploader(
    "Upload odds_inputs.csv",
    type=["csv"],
    help="player_id,player,hr_odds_american,sportsbook,timestamp_et",
)

if weather_upload is not None:
    (APP_DIR / "environment_inputs.csv").write_bytes(weather_upload.getvalue())
    st.success("Weather inputs loaded.")

if odds_upload is not None:
    (APP_DIR / "odds_inputs.csv").write_bytes(odds_upload.getvalue())
    st.success("Odds inputs loaded.")

if st.button("Run Full Scan", type="primary", use_container_width=True):
    cmd = [
        sys.executable,
        str(APP_DIR / "scanner.py"),
        "--date",
        scan_date.isoformat(),
        "--output-dir",
        str(OUTPUT_DIR),
    ]
    if include_unconfirmed:
        cmd.append("--include-unconfirmed")

    with st.status("Running scanner...", expanded=True) as status:
        st.write("Loading the schedule, pitchers and lineups...")
        st.write("Calculating last-10 Statcast metrics...")
        process = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True)

        if process.returncode != 0:
            status.update(label="Scan failed", state="error")
            st.error(process.stderr or process.stdout)
            st.stop()

        status.update(label="Scan complete", state="complete")
        if process.stdout:
            st.code(process.stdout[-2500:])

csv_path = OUTPUT_DIR / f"outlaw_scanner_{scan_date.isoformat()}.csv"
xlsx_path = OUTPUT_DIR / f"outlaw_scanner_{scan_date.isoformat()}.xlsx"

if csv_path.exists():
    board = pd.read_csv(csv_path)

    st.subheader("Top targets")
    c1, c2, c3 = st.columns(3)
    c1.metric("Players scanned", len(board))
    c2.metric("Core HR eligible", int(board["Core_HR_Eligible"].fillna(False).sum()))
    c3.metric("Top score", f"{board['Model_Score'].max():.1f}")

    preferred = [
        "Model_Score","Core_HR_Eligible","Qualifying_Power_Signals",
        "player","team","opponent","lineup_spot","opposing_pitcher",
        "AVG","H","HR","RBI","TB","Avg_EV","EV90","Max_EV",
        "HH_95","HH_pct","EV_100_plus","EV_100_plus_outs",
        "Barrels_approx","Barrel_pct_approx","Avg_LA",
        "SweetSpot_pct","PullAir_pct","Fly_350_plus","Fly_375_plus",
        "Out_380_400","Near_HR","xHR_proxy","xHR_minus_HR",
        "Pitcher_HR_pct","Pitcher_HH_pct","Pitcher_Barrel_pct_approx",
        "Pitch_Mix_Score","Park_Factor","Weather_Factor","HR_Odds_American"
    ]
    display_cols = [c for c in preferred if c in board.columns]

    tab1, tab2, tab3 = st.tabs(["Top 20", "Core HR", "Full board"])
    with tab1:
        st.dataframe(board[display_cols].head(20), use_container_width=True, hide_index=True, height=650)
    with tab2:
        core = board[board["Core_HR_Eligible"] == True]
        if core.empty:
            st.info("No hitters currently meet the full Core HR gate.")
        else:
            st.dataframe(core[display_cols].head(30), use_container_width=True, hide_index=True, height=650)
    with tab3:
        st.dataframe(board[display_cols], use_container_width=True, hide_index=True, height=700)

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
