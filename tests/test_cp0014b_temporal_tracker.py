from linecaller.proposals.temporal_tracker import TemporalProposalTracker


def test_temporal_tracker_predicts_constant_velocity():
    t = TemporalProposalTracker(confidence=.8)

    t.observe(10, 100, 100, 10, 10)
    t.observe(11, 105, 103, 10, 10)

    p = t.predict(12)

    assert p is not None
    assert p.x == 110
    assert p.y == 106


def test_temporal_tracker_requires_history():
    assert TemporalProposalTracker().predict(1) is None
