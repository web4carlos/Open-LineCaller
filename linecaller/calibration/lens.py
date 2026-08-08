from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class LensModel:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(
            image,
            self.camera_matrix,
            self.distortion_coefficients,
        )

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LensModel":
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                data["distortion_coefficients"], dtype=np.float64
            ),
        )
