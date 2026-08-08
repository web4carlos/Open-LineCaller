# CP-0016 Technical Design — Live Match Application v1

## Goal

Expose the CP-0015 Live Officiating Core through a real operator/player UI.

## Components

LiveMatchWindow
- camera selection
- lifecycle
- preview
- status cards
- call banner
- replay trigger
- developer diagnostics

LiveCameraWorker
- captures frames in a QThread
- emits source-frame images
- reports measured capture FPS

LiveAudioNotifier
- abstraction for IN / OUT notification
- uses system beep fallback in v1
- future mobile implementations may use native audio assets

LiveMatchController
- state machine for READY / LIVE / STOPPED
- converts LiveEvidence into UI state
- REVIEW opens replay state

## Safety

REVIEW remains replay-only.
The UI never converts REVIEW into IN/OUT.

## Future

CP-0017:
- mobile runtime / Android-first shell
- native camera
- native audio
- mobile calibration flow
