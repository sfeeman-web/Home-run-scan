
import pandas as pd
from scanner import batted_ball_metrics, add_model_scores, DEFAULT_WEIGHTS

def test_metrics():
    rows = [
        {"events":"home_run","launch_speed":108.0,"launch_angle":27.0,"hit_distance_sc":420,
         "stand":"R","hc_x":95,"game_date":"2026-07-01"},
        {"events":"field_out","launch_speed":103.0,"launch_angle":29.0,"hit_distance_sc":392,
         "stand":"R","hc_x":100,"game_date":"2026-07-02"},
        {"events":"single","launch_speed":97.0,"launch_angle":12.0,"hit_distance_sc":220,
         "stand":"R","hc_x":110,"game_date":"2026-07-03"},
    ]
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    m = batted_ball_metrics(df)
    assert m["BBE"] == 3
    assert m["EV_100_plus"] == 2
    assert m["EV_100_plus_outs"] == 1
    assert m["Fly_375_plus"] == 2
    assert m["Near_HR"] >= 1
    print("Synthetic metric test passed.", m)

if __name__ == "__main__":
    test_metrics()
