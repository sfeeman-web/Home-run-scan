# Version 4 Calibration

Changes made after reviewing the July 20 export:

1. xHR is now multiplicative and conservative. Exit velocity, launch angle,
   and distance must all align.
2. Pitcher HR, hard-hit, barrel, and EV metrics are shrunk toward league-average
   priors using a 40-BBE prior.
3. Pitch-mix scores are shrunk toward neutral when the batter sample is small.
4. Contact, pitcher, due, and park scores now use fixed baseball ranges instead
   of slate-relative min/max scaling.
5. Due-score xHR gap is capped so it cannot dominate the model.
6. Core HR eligibility now requires:
   - at least 4 of 7 power signals,
   - at least 15 BBE,
   - a top-six lineup position,
   - active pregame status.
7. The app now displays G, PA, AB, BBE, pitcher BBE, raw and adjusted pitcher
   rates, sample reliability, and a sample flag.

This patch calibrates the HR model first. Separate Hit, Run, RBI, Total Base,
and HRR scores should be added only after the HR score is validated against
several completed slates.
