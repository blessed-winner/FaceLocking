"""Enroll or replace an identity in the local FaceLocking database."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .recognition import ArcFaceEmbedderONNX, FaceMeshDetector, align_face_5pt, load_db_npz

DB_PATH = Path("data/db/face_db.npz")
META_PATH = Path("data/db/face_db.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll an identity for FaceLocking.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    name = args.name.strip()
    if not name:
        parser.error("--name must not be blank")

    detector, embedder = FaceMeshDetector(), ArcFaceEmbedderONNX()
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError("Camera not available.")
    samples: List[np.ndarray] = []
    auto, last_auto = False, 0.0
    print("SPACE=capture, a=auto capture, s=save, q=quit")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            faces = detector.detect(frame, max_faces=1)
            aligned = align_face_5pt(frame, faces[0].kps) if faces else None
            view = frame.copy()
            if faces:
                face = faces[0]
                cv2.rectangle(view, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            cv2.putText(view, f"ENROLL {name}: {len(samples)}/{args.samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            cv2.putText(view, "SPACE=capture a=auto s=save q=quit", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.imshow("FaceLocking Enrollment", view)
            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            capture_sample = key == ord(" ") or (auto and aligned is not None and now - last_auto >= .25)
            if capture_sample and aligned is not None:
                samples.append(embedder.embed(aligned))
                last_auto = now
            if key == ord("a"):
                auto = not auto
            if key == ord("s"):
                if len(samples) < args.samples:
                    print(f"Need {args.samples} samples; have {len(samples)}.")
                    continue
                template = np.mean(np.stack(samples), axis=0)
                template /= np.linalg.norm(template) + 1e-12
                db = load_db_npz(DB_PATH)
                db[name] = template.astype(np.float32)
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.savez(DB_PATH, **db)
                META_PATH.write_text(json.dumps({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "names": sorted(db)}, indent=2), encoding="utf-8")
                print(f"Saved {name} ({len(samples)} samples).")
                break
            if key == ord("q"):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
