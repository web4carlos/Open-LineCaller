from __future__ import annotations

import sys


class LiveAudioNotifier:
    def announce(self, call: str):
        call = str(call).upper()

        if call not in ("IN", "OUT"):
            return

        try:
            if sys.platform.startswith("win"):
                import winsound
                frequency = 900 if call == "IN" else 500
                winsound.Beep(frequency, 180)
        except Exception:
            pass
