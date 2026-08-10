# CP-0020 Technical Design — Real Artificial Vision Core Integration

## Problem

The live factory currently returns:

    RealTimeOfficiatingAdapter(
        NullLivePerceptionAdapter(),
        minimum_call_confidence=0.98,
    )

Therefore UIs and validation tools can be structurally correct while receiving
no real perception events.

## Existing core reused

The project already has:

- MotionBallDetector
- BallEngine
- BallTracker
- BounceEngine
- DecisionEngine
- CourtFusion through DecisionEngine

CP-0020 extracts the frame-by-frame logic already present in
IntegratedVideoPipeline into a LivePerceptionAdapter implementation.

## New adapter

RealArtificialVisionPerceptionAdapter

Input:
- frame_number
- frame
- timestamp

Output:
- LivePipelineResult

Processing:
1. BallEngine.process(frame_number, frame)
2. Convert tracked/predicted state into TrajectoryPoint
3. BounceEngine.process(point)
4. If bounce:
   - with calibration: DecisionEngine.decide()
   - without calibration: REVIEW candidate
5. Return structured LivePipelineResult

## Shared core requirement

Recorded files and live cameras must use the same perception object.
Only the frame source differs.

## Detector policy

Default detector:
- MotionBallDetector (existing baseline)

Injectable detector:
- any BallDetector-compatible detector

This preserves a future path to YOLO/custom pickleball weights without changing
the product UI or validation interfaces.

## Validation safety

No calibration means no automatic IN/OUT decision.

A detected bounce becomes REVIEW so annotation tools can surface the event while
the human operator supplies ground truth.

## Reset

The perception adapter supports reset() for a new match/session. It recreates
the default tracker and resets BounceEngine state where possible.
