# V3.2 Enhanced Matchup Engine

This patch keeps the original Version 3 model weights and Top 40 output.

## Added
- Pitcher HR, hard-hit, barrel, fly-ball, and damaging pull-air rates versus the hitter's batting side.
- Overall pitcher rates alongside handedness-specific rates.
- Pitcher's most-used pitch types.
- HRs allowed by pitch type versus the hitter's side.
- Batter performance against the pitcher's actual recent pitch mix.
- Batter performance in the pitcher's primary velocity band.

## Not invented
- Weather, roof, odds, and bullpen adjustments remain neutral unless supplied through optional inputs or future verified data sources.

## Model weights unchanged
- 30% batter contact quality
- 25% pitcher vulnerability
- 15% pitch mix and velocity compatibility
- 15% park/environment
- 10% due indicators
- 5% market value
