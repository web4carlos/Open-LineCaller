from linecaller.validation.candidate_scanner import CandidateScanner

def test_min_gap():
    scanner=CandidateScanner(pipeline=object(),min_frame_gap=5)
    assert scanner.min_frame_gap==5
