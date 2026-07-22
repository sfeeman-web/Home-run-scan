# V3.3 Automatic Weather

- Preserves Version 3.2 scoring and Top 40 output.
- Pulls Open-Meteo hourly forecasts nearest scheduled first pitch.
- Adds temperature, humidity, precipitation probability, wind speed/direction, and stadium-relative wind-out component.
- Applies a capped 0.90–1.12 weather factor for outdoor parks.
- Retractable-roof parks remain neutral until roof status is manually confirmed.
- Manual environment_inputs.csv values override automatic values by game_pk.
- Adds Refresh Weather Only, which updates environment scores without downloading Statcast again.

Optional override columns: game_pk, weather_factor, roof_status, temp_f, humidity_pct, wind_speed_mph, wind_direction_deg, wind_out_mph, precip_probability_pct, weather_warning.
