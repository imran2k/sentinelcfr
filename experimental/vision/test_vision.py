from experimental.vision.vision import violations

def test_missing_helmet():
    dets = [{"label": "person", "box": [0, 0, 100, 200], "confidence": 0.9}]
    assert violations(dets)[0]["missing"] == "helmet"

def test_helmet_present():
    dets = [{"label": "person", "box": [0, 0, 100, 200], "confidence": 0.9},
            {"label": "helmet", "box": [30, 0, 70, 30], "confidence": 0.9}]
    assert violations(dets) == []