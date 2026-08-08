from linecaller.dataset.assisted_controller import AssistedMetrics


def test_acceptance_rate():
    m = AssistedMetrics(
        accepted=8,
        adjusted=1,
        rejected=1,
    )

    assert m.acceptance_rate == .9
