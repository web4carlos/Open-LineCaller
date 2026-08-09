from linecaller.product.match_summary import MatchSummaryBuilder


def test_tracking_losses_count_lost_only():
    b = MatchSummaryBuilder()

    b.record_runtime(
        tracking_status="LOCKED"
    )
    b.record_runtime(
        tracking_status="LOST"
    )
    b.record_runtime(
        tracking_status="SEARCHING"
    )

    assert b.tracking_losses == 1
