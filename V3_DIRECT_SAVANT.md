# Version 3 — Direct Baseball Savant

This version removes pybaseball.

Changes:
- Direct requests to Baseball Savant's Statcast CSV output
- One-day sequential requests to respect result limits
- Immediate filtering to terminal PA rows, BBE rows and today's probable pitchers
- No pybaseball post-processing layer
- Smaller dependency installation
