HUMAN_LABELS = {
    "near_left_corner": "Bottom-left outer corner",
    "near_right_corner": "Bottom-right outer corner",
    "far_right_corner": "Top-right outer corner",
    "far_left_corner": "Top-left outer corner",
    "near_nvz_left": "Bottom-left Kitchen corner",
    "near_nvz_right": "Bottom-right Kitchen corner",
    "far_nvz_left": "Top-left Kitchen corner",
    "far_nvz_right": "Top-right Kitchen corner",
}

def human_label(name: str) -> str:
    return HUMAN_LABELS.get(name, name.replace("_", " ").title())
