from experimental.vision.vision import violations

PERSON = {"label": "person", "box": [0, 0, 100, 200], "confidence": 0.9}
HELMET = {"label": "helmet", "box": [30, 0, 70, 30], "confidence": 0.9}

def test_missing_helmet():
    assert violations([PERSON])[0]["missing"] == "helmet"

def test_helmet_present():
    assert violations([PERSON, HELMET]) == []

def test_helmet_on_someone_else():
    far = {"label": "helmet", "box": [500, 0, 540, 30], "confidence": 0.9}
    assert len(violations([PERSON, far])) == 1