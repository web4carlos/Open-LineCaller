from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=14)
    a = p.parse_args()

    img = cv2.imread(a.input)
    if img is None:
        raise RuntimeError(f"Could not read template: {a.input}")

    h, w = img.shape[:2]
    s = max(6, min(int(a.size), h, w))

    cx = w // 2
    cy = h // 2
    r = s // 2

    crop = img[max(0, cy-r):min(h, cy-r+s),
               max(0, cx-r):min(w, cx-r+s)]

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(out), crop):
        raise RuntimeError("Could not save refined template")

    print(f"input={a.input}")
    print(f"output={out}")
    print(f"refined_size={crop.shape[1]}x{crop.shape[0]}")


if __name__ == "__main__":
    main()
