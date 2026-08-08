from linecaller.apps.live_audio import LiveAudioNotifier


def test_audio_ignores_review():
    notifier = LiveAudioNotifier()

    # Must not raise.
    notifier.announce("REVIEW")
