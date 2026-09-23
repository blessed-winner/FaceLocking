"""Facial-expression and eye-state classification from FaceMesh landmarks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, Sequence, TypeVar

import numpy as np


class ExpressionSignal(str, Enum):
    NEUTRAL = "NEUTRAL"
    SMILE = "SMILE"


class EyeSignal(str, Enum):
    EYES_OPEN = "EYES_OPEN"
    EYES_CLOSED = "EYES_CLOSED"


@dataclass(frozen=True)
class FacialSignals:
    expression: ExpressionSignal
    eyes: EyeSignal


EYE_CLOSED_THRESHOLD = 0.19
SMILE_WIDTH_THRESHOLD = 0.62
SMILE_REFERENCE_EYE_DISTANCE_PX = 90.0
SMILE_FAR_FACE_TOLERANCE = 0.08
STABLE_FRAMES = 3

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_CORNERS = (61, 291)


def _distance(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    a = _distance(points, indices[1], indices[5])
    b = _distance(points, indices[2], indices[4])
    c = _distance(points, indices[0], indices[3])
    return (a + b) / (2.0 * c) if c > 1e-6 else 0.0


def smile_width_threshold(eye_distance: float) -> float:
    if eye_distance >= SMILE_REFERENCE_EYE_DISTANCE_PX:
        return SMILE_WIDTH_THRESHOLD
    fraction = 1.0 - max(0.0, eye_distance) / SMILE_REFERENCE_EYE_DISTANCE_PX
    return SMILE_WIDTH_THRESHOLD - SMILE_FAR_FACE_TOLERANCE * fraction


def detect_face_signals(face_landmarks: np.ndarray) -> FacialSignals:
    points = np.asarray(face_landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) <= max(RIGHT_EYE):
        raise ValueError("Expected MediaPipe FaceMesh landmarks with x/y coordinates.")
    average_ear = (eye_aspect_ratio(points, LEFT_EYE) + eye_aspect_ratio(points, RIGHT_EYE)) / 2.0
    eye_distance = _distance(points, 33, 263)
    mouth_width = _distance(points, *MOUTH_CORNERS)
    expression = (
        ExpressionSignal.SMILE
        if mouth_width / max(eye_distance, 1e-6) >= smile_width_threshold(eye_distance)
        else ExpressionSignal.NEUTRAL
    )
    eyes = EyeSignal.EYES_CLOSED if average_ear <= EYE_CLOSED_THRESHOLD else EyeSignal.EYES_OPEN
    return FacialSignals(expression=expression, eyes=eyes)


SignalType = TypeVar("SignalType", bound=Enum)


class SignalStabilizer(Generic[SignalType]):
    def __init__(self, initial_signal: SignalType, stable_frames: int = STABLE_FRAMES):
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        self.current = initial_signal
        self.stable_frames = stable_frames
        self._candidate: Optional[SignalType] = None
        self._candidate_frames = 0

    def update(self, observed: SignalType) -> SignalType:
        if observed == self.current:
            self._candidate, self._candidate_frames = None, 0
        elif observed == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = observed, 1
        if self._candidate_frames >= self.stable_frames:
            self.current = observed
            self._candidate, self._candidate_frames = None, 0
        return self.current
