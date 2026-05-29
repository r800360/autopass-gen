"""Front gap must not use passing-lane rear detections."""
from __future__ import annotations

from autopass.perception_state import classify_car_detection, finalize_front_lead_detection, gaps_from_classified_cars


def test_rear_labeled_detection_never_used_for_front_gap():
    raw = {
        "bbox": [500, 100, 700, 200],
        "median_depth": 16.0,
        "position": "rear",
        "cy_mean": 150.0,
    }
    c = classify_car_detection(raw, image_width=1280.0, image_height=256.0)
    assert c["used_for_front_gap"] is False
    assert "rear" in c["classification_reason"]


def test_finalize_picks_furthest_non_rear_as_lead():
    classified = [
        classify_car_detection(
            {"bbox": [0, 0, 1, 1], "median_depth": 16.0, "position": "rear", "cy_mean": 150.0},
            image_width=1280.0,
            image_height=256.0,
        ),
        classify_car_detection(
            {"bbox": [0, 0, 1, 1], "median_depth": 31.0, "position": "front", "cy_mean": 150.0},
            image_width=1280.0,
            image_height=256.0,
        ),
    ]
    out = finalize_front_lead_detection(classified)
    gaps = gaps_from_classified_cars(out)
    assert gaps["front_gap_m"] == 31.0
    lead = [c for c in out if c.get("used_for_front_gap")][0]
    assert lead["position"] == "front"
