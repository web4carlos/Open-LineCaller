# CP-0036 â€” Live App Bridge

Status: IMPLEMENTED

## Goal

Move Open LineCaller from offline/engineering modules to a real live-camera
application path without changing the OFFICIAL outside-only decision architecture.

## Runtime

Browser camera
â†’ JPEG frame
â†’ FastAPI
â†’ `OfficialExternalLiveRuntime`
â†’ Camera Pose Guard
â†’ Camera Ownership
â†’ Receiving Side
â†’ Official External Grid v2
â†’ Pillow / locked ball color
â†’ Z0 candidate
â†’ UP after the fact
â†’ confirmed OUT
â†’ soft floor glow
â†’ browser result

## Safety

The API does not decide OUT independently.

It exposes the decisions produced by the existing official runtime.

- `BLOCKED` pose means no external-grid scan and no OUT.
- `MICRO_ADJUST` suppresses the transition frame.
- No ownership for the current receiving side means zero scanned cells.
- A Z0 candidate alone never becomes OUT.
- Only an owned `OUT_*` cell can produce an `OfficialOutCall`.

## Endpoints

- `GET /`
  - local engineering camera console.
- `GET /health`
- `GET /api/live/status`
- `POST /api/live/configure-paths`
  - configure from local server paths.
- `POST /api/live/configure-upload`
  - upload calibration/background/ball template from browser.
- `POST /api/live/receiving-side`
- `POST /api/live/frame`
  - raw JPEG/PNG body.
  - returns pose status, active external cells, Z0 count, OUT events and
    optionally a processed JPEG with glow.

## Browser console

The engineering console uses `navigator.mediaDevices.getUserMedia()`.

It captures frames into a canvas and sends them to FastAPI at a conservative
engineering rate (~8 fps). This is intentionally a live validation gate before
the final React/TypeScript/Tailwind product UI is layered over the same API.

## Run

From PowerShell:

```powershell
cd C:\olinecaller
python tools\run_live_api.py
```

Then open:

`http://127.0.0.1:8000`

## Why this comes before React

The repository had no existing FastAPI/React/WebRTC application layer.

CP-0036 first proves the complete live court loop with the official CV runtime.
The React UI can then consume a stable, tested API instead of coupling UI work
to unfinished CV wiring.