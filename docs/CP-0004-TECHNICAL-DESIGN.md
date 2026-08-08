# CP-0004 Technical Design

Pipeline:

Ball track -> trajectory -> local velocity analysis -> bounce candidate ->
evidence validation -> cooldown -> BounceEvent.

Image convention: +Y is downward. A normal bounce therefore tends to have
positive pre-bounce vertical velocity and negative post-bounce vertical velocity.

A direction reversal alone is insufficient. The detector also requires
measured support, tracking confidence, minimum speed, and temporal confirmation.
