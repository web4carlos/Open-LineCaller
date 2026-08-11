# CP-0023 Sprint 1 — Pickleball Annotation Studio

Goal: zero-typing annotation.

Open video → click pickleball → SPACE → next frame.

No ball: N

Navigation:
- Left/Right: 1 frame
- Shift+Left/Right: 10 frames
- Ctrl+Left/Right: 100 frames
- Delete: remove current label

Dataset output:
- `C:\validation\pickleball-dataset\raw\images`
- `C:\validation\pickleball-dataset\raw\labels`

YOLO class:
- `0 = pickleball`

Negative frames create an empty `.txt`.

The studio autosaves `<video>.annotation.json` in the raw dataset folder and restores progress automatically.

Recommended pilot: label 200–300 diverse frames first, train a short YOLO26 pilot, inspect the overlay, then expand.
