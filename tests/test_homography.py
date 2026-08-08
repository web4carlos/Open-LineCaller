import numpy as np

from linecaller.calibration.homography import HomographyEstimator


def test_homography_maps_points():
    image = [(100, 100), (500, 100), (500, 900), (100, 900)]
    court = [(0, 0), (6, 0), (6, 12), (0, 12)]

    result = HomographyEstimator.estimate(image, court)
    transformed = HomographyEstimator.transform_points(image, result.matrix)

    assert np.allclose(transformed, np.asarray(court), atol=1e-6)
