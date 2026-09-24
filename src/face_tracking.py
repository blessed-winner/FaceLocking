"""Lock facial-signal tracking to one exact enrolled identity.

Run: python -m src.face_tracking --name="Winner"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2

from .face_signals import EyeSignal, ExpressionSignal, SignalStabilizer, detect_face_signals
from .recognition import ArcFaceEmbedderONNX, FaceDBMatcher, FaceDet, FaceMeshDetector, MatchResult, align_face_5pt, load_db_npz

DB_PATH = Path("data/db/face_db.npz")


def _recognize(frame, faces: List[FaceDet], embedder: ArcFaceEmbedderONNX, matcher: FaceDBMatcher) -> List[MatchResult]:
    return [matcher.match(embedder.embed(align_face_5pt(frame, face.kps))) for face in faces]


def _label(frame, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display facial signals for one enrolled person only.")
    parser.add_argument("--name", required=True, help="Exact name enrolled in data/db/face_db.npz.")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=.34, help="ArcFace distance cutoff; higher is more forgiving.")
    parser.add_argument("--stable-frames", type=int, default=3)
    args = parser.parse_args()

    target_name = args.name.strip()
    database = load_db_npz(DB_PATH)
    if not target_name or target_name not in database:
        parser.error(f'Target "{target_name}" was not found in the face database.')

    detector = FaceMeshDetector()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(database, args.threshold)
    expression_stabilizer = SignalStabilizer(ExpressionSignal.NEUTRAL, args.stable_frames)
    eyes_stabilizer = SignalStabilizer(EyeSignal.EYES_OPEN, args.stable_frames)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError("Camera not available. Try --camera 1.")

    locked_once = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            faces = detector.detect(frame)
            matches = _recognize(frame, faces, embedder, matcher)
            target_index: Optional[int] = next(
                (index for index, match in enumerate(matches) if match.accepted and match.name == target_name), None,
            )
            expression, eyes = "NOT_DETECTED", "NOT_DETECTED"
            if target_index is not None:
                locked_once = True
                observed = detect_face_signals(faces[target_index].landmarks)
                expression = expression_stabilizer.update(observed.expression).value
                eyes = eyes_stabilizer.update(observed.eyes).value

            view = frame.copy()
            for index, (face, match) in enumerate(zip(faces, matches)):
                is_target = index == target_index
                color = (0, 255, 255) if is_target else ((0, 255, 0) if match.accepted else (0, 0, 255))
                label = match.name if match.accepted else "Unknown"
                if is_target:
                    label += " [LOCKED]"
                cv2.rectangle(view, (face.x1, face.y1), (face.x2, face.y2), color, 3 if is_target else 2)
                _label(view, f"{label} d={match.distance:.3f}", (face.x1, max(24, face.y1 - 10)), color)

            state = "LOCKED" if locked_once else "WAITING"
            _label(view, f"TARGET = {target_name} ({state})", (10, 30), (0, 255, 255))
            _label(view, f"EXPRESSION = {expression}", (10, 60), (0, 255, 255))
            _label(view, f"EYES = {eyes}", (10, 90), (0, 255, 255))
            _label(view, "Q = quit", (10, view.shape[0] - 15), (220, 220, 220))
            cv2.imshow("FaceLocking", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
