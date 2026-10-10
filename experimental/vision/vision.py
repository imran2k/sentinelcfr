import os
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_VISION_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _VISION_DIR.parent.parent

def _load_model():
    # Search order: experimental/vision/ first, then models/
    search_candidates = [
        _VISION_DIR / "ppe.pt",
        _VISION_DIR / "best.pt",
        _VISION_DIR / "yolov8n.pt",
        _ROOT_DIR / "models" / "ppe.pt",
        Path("models/ppe.pt"),
        _ROOT_DIR / "models" / "best.pt",
        Path("models/best.pt"),
        _ROOT_DIR / "yolov8n.pt",
        Path("yolov8n.pt"),
    ]

    # Optional explicit environment override
    env_override = os.getenv("YOLO_MODEL_PATH", "").strip()
    if env_override and os.path.isfile(env_override):
        search_candidates.insert(0, Path(env_override))

    resolved_path = None
    for cand in search_candidates:
        cand_str = str(cand.resolve()) if hasattr(cand, "resolve") else str(cand)
        # Verify strictly with os.path.isfile to NEVER pass a directory or "."
        if os.path.isfile(cand_str):
            resolved_path = cand_str
            break

    if not resolved_path:
        # Fallback to yolov8n.pt only if no file could be verified on disk
        resolved_path = "yolov8n.pt"

    logger.info("Loading YOLO model from verified path: %s", resolved_path)
    try:
        loaded = YOLO(resolved_path, task="detect")
        loaded.model_name = os.path.basename(resolved_path)
        logger.info("YOLO model successfully loaded from %s", resolved_path)
        return loaded
    except Exception as e:
        logger.error("Exception loading YOLO model from %s: %s", resolved_path, e)
        if resolved_path != "yolov8n.pt":
            logger.warning("Attempting emergency fallback to yolov8n.pt")
            fallback = YOLO("yolov8n.pt", task="detect")
            fallback.model_name = "yolov8n.pt"
            return fallback
        raise

model = _load_model()
REQUIRED = ["helmet"]  # must match YOLO class names from Step 3; add e.g. "safety-vest"

def detect(img):
    r = model(img)[0]
    return [{"label": r.names[int(b.cls)],
             "confidence": round(float(b.conf), 3),
             "box": [round(v, 1) for v in b.xyxy[0].tolist()]} for b in r.boxes]

def _inside(box, outer):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

def is_negative_label(label: str) -> bool:
    lbl = str(label).lower()
    return lbl.startswith("no_") or lbl.startswith("no ") or lbl.startswith("no-")

def _clean_negative_label(label: str) -> str:
    clean = str(label).lower()
    for prefix in ("no_", "no-", "no "):
        if clean.startswith(prefix):
            return clean[len(prefix):]
    return clean

def violations(dets):
    out = []
    people = [d for d in dets if d.get("label") == "person"]

    neg_idx = 0
    flagged_person_items = set()

    # 1. Direct violations from negative classes (e.g., no_goggles, no_helmet, no_shoes, etc.)
    for d in dets:
        lbl = d.get("label", "")
        if is_negative_label(lbl):
            clean_item = _clean_negative_label(lbl)
            matched_person = None
            for p_idx, p in enumerate(people):
                if _inside(d["box"], p["box"]):
                    matched_person = p_idx
                    break

            assigned_idx = matched_person if matched_person is not None else neg_idx
            viol = {
                "person": assigned_idx,
                "missing": clean_item,
                "box": d["box"],
                "confidence": d.get("confidence", 1.0)
            }
            out.append(viol)
            flagged_person_items.add((assigned_idx, clean_item))
            if matched_person is None:
                neg_idx += 1

    # 2. Retain person-containment logic for positive required classes
    for i, p in enumerate(people):
        for item in REQUIRED:
            if (i, item) in flagged_person_items:
                continue
            if not any(d.get("label") == item and _inside(d["box"], p["box"]) for d in dets):
                out.append({
                    "person": i,
                    "missing": item,
                    "box": p["box"],
                    "confidence": p.get("confidence", 1.0)
                })
                flagged_person_items.add((i, item))

    return out

def detect_violations(image_path):
    return violations(detect(image_path))