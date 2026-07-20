
# Outlaw MLB Last-10 Scanner

This project builds the exact rolling hitter table requested:

- Games, plate appearances, at-bats, average, hits, HR, RBI, total bases
- Average EV, EV90, maximum EV
- 95+ mph hard-hit count and rate
- 100+ mph contact and 100+ mph outs
- Approximate barrels and barrel rate
- Average launch angle and sweet-spot rate
- Pull-air count and rate
- 350+ and 375+ foot fly balls
- 380–400 foot outs
- Near-HR count
- Transparent custom xHR proxy and xHR minus actual HR
- Opposing-pitcher HR, hard-hit and barrel vulnerability by batter side
- Pitch-mix matchup score
- Park/weather and market-value inputs
- Final custom model score

## Important definitions

- **Last 10** means each hitter's ten most recent distinct MLB game dates.
- **Hard hit** means exit velocity of at least 95 mph.
- **Sweet spot** means launch angle from 8 through 32 degrees.
- **Barrels_approx** is a public-rule approximation calculated from EV and launch angle. It is labeled approximate.
- **xHR_proxy** is this scanner's transparent custom estimate. It is **not** MLB's proprietary/official xHR.
- Runs scored are left blank because they cannot be reliably reconstructed from batter-only Statcast event rows without play-runner data. Hits, HR, RBI and TB are calculated from plate-appearance outcomes.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run after lineups are confirmed

```bash
python scanner.py --date 2026-07-20
```

Outputs:

- `output/outlaw_scanner_2026-07-20.csv`
- `output/outlaw_scanner_2026-07-20.xlsx`

## Morning roster screen

Before lineups are posted:

```bash
python scanner.py --date 2026-07-20 --include-unconfirmed
```

This screens active non-pitchers. Treat lineup spot and eligibility as preliminary.

## Weather and roof inputs

Edit `environment_inputs.csv`:

```csv
game_pk,temp_f,wind_out_mph,roof_status,weather_factor
823523,82,7,open,1.05
```

Suggested weather factors:

- 0.90–0.96: cold/heavy air or strong wind in
- 0.98–1.02: neutral
- 1.03–1.08: warm with modest wind out
- 1.09–1.15: extreme hitter environment
- Roof closed: normally 0.99–1.01 unless park-specific evidence says otherwise

## Odds inputs

Edit `odds_inputs.csv` with current HR prices. The model converts price to a preliminary market-value score. This is not a substitute for shopping multiple books.

## Model weights

Edit `weights.json`. Default saved model:

- 30% batter contact quality
- 25% opposing-pitcher vulnerability
- 15% pitch-mix compatibility
- 15% park/environment
- 10% historical contact quality/regression
- 5% sportsbook value

## Core HR gate

A hitter needs at least two of these six last-10 signals:

1. Two or more approximate barrels
2. One 105+ mph maximum EV
3. Two or more 100+ mph batted balls
4. One or more 375+ foot flies
5. xHR proxy exceeds actual HR by more than 0.35
6. Pull-air rate at least 20%

He must also be scheduled, active, and projected/confirmed in the top six.

## Data sources

The scanner uses:

- MLB Stats API for schedule, probable pitchers and game lineups
- Baseball Savant/Statcast data through `pybaseball`

Baseball Savant CSV field documentation:
https://baseballsavant.mlb.com/csv-docs

PyBaseball:
https://github.com/jldbc/pybaseball

## Limitations

- Public Statcast endpoints can change.
- Minor-league or very recent call-up data may be sparse.
- Confirmed lineups are usually unavailable early in the day.
- Weather, scratches, bullpen availability and prices still require final verification.
- The scanner ranks candidates; it does not guarantee profitable bets.
