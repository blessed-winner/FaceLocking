# FaceLocking

Face recognition that locks onto one enrolled identity and displays that
person's expression (`SMILE` or `NEUTRAL`) and eye state (`EYES_OPEN` or
`EYES_CLOSED`). Other faces never provide signals for the target.

## Setup

Use Python 3.11, then install dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place a compatible ArcFace ONNX model at either:

```text
models/embedder_arcface.onnx
models/w600k_r50.onnx
```

Model weights and face databases are deliberately ignored by Git.

## Enroll identities

```powershell
python -m src.enroll --name="Winner"
```

Press `SPACE` to take samples, `a` for automatic capture, `s` to save after
the requested sample count, or `q` to quit. Enrollment is stored locally at
`data/db/face_db.npz`.

To keep your existing enrollments when moving from the original project,
copy these local files yourself (do not commit them):

```text
old-project/data/db/face_db.npz -> FaceLocking/data/db/face_db.npz
old-project/data/db/face_db.json -> FaceLocking/data/db/face_db.json
```

## Track one person

```powershell
python -m src.face_tracking --name="Winner"
```

`--name` must be an exact enrolled name. The tracker shows `NOT_DETECTED`
when that person leaves the frame and does not switch to another face. Press
`Q` to quit.

## Tuning signals

`src/face_signals.py` contains the eye and smile thresholds. Smile tolerance
is adjusted slightly for smaller faces to account for noisier distant
landmarks. `--stable-frames` controls how many matching frames are required
before an expression or eye state display changes.
