from linecaller.dcf.candidate_selector import (
    BallCandidateObservation,
    DCFBallCandidateSelector,
)

def c(frame,x,y,conf,w=12,h=12):
    return BallCandidateObservation(frame,x,y,conf,w,h)

def main():
    selector=DCFBallCandidateSelector(max_distance_px=90)

    expected=(220.0,500.0)
    candidates=[
        c(21611,865.0,532.0,.80,40,50),   # wrong object / player region
        c(21611,235.0,505.0,.42,10,10),   # plausible ball
        c(21611,940.0,455.0,.90,18,18),   # high confidence but impossible
    ]

    result=selector.select(
        candidates,
        expected_x=expected[0],
        expected_y=expected[1],
        expected_size=10.0,
    )

    print("DCF CANDIDATE SELECTOR DEMO")
    print(f"candidate_count={result.candidate_count}")
    print(f"reason={result.reason}")

    if result.selected:
        s=result.selected
        print(
            f"selected=({s.x:.1f},{s.y:.1f}) "
            f"confidence={s.confidence:.3f} "
            f"size=({s.width:.1f},{s.height:.1f})"
        )

    for item in result.scored:
        candidate=item.candidate
        print(
            f"candidate=({candidate.x:.1f},{candidate.y:.1f}) "
            f"score={item.score:.4f} "
            f"distance={item.distance_px:.2f}px "
            f"size_ratio={item.size_ratio:.3f} "
            f"confidence={candidate.confidence:.3f}"
        )

if __name__=="__main__":
    main()

