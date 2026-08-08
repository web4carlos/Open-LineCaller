from linecaller.benchmark.matching import match_events


def test_matching_within_tolerance():
    expected = [{"frame":100,"decision":"OUT"}]
    detected = [{"bounce_frame":101,"decision":"OUT"}]

    matches, fp, fn = match_events(
        expected,
        detected,
        frame_tolerance=2,
    )

    assert len(matches) == 1
    assert matches[0].frame_error == 1
    assert matches[0].decision_match is True
    assert fp == 0
    assert fn == 0


def test_matching_counts_fp_and_fn():
    expected = [{"frame":100}]
    detected = [{"bounce_frame":120,"decision":"IN"}]

    matches, fp, fn = match_events(
        expected,
        detected,
        frame_tolerance=2,
    )

    assert len(matches) == 0
    assert fp == 1
    assert fn == 1
