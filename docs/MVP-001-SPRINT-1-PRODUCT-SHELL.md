# MVP-001 — Sprint 1 Product Shell

Goal: turn Open-LineCaller from a collection of engineering tools into one coherent desktop application.

User paths:

- Home -> Analyze Match -> Open Video -> Start
- Home -> Live Match -> Start Live
- Home -> Developer Tools

Analyze Match and Live Match both call the existing `create_live_pipeline_adapter()` core.

Included:
- Home
- Analyze Match workspace
- Live Match workspace
- unified dark visual system
- Settings
- Developer Tools
- basic officiating HUD
- session statistics
- graceful source start/pause/stop
- skip initial black video frames

Deferred:
- final audio calls
- final replay experience
- calibration wizard integration
- production-grade settings
- final visual polish
- accuracy optimization
