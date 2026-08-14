"""Record raw webcam frames, so a failure seen live can be replayed and diagnosed.

    GEM-X/.venv/bin/python record_input.py outputs/input_capture.mp4 20
"""

import sys
import time

import cv2


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/input_capture.mp4"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "cannot open camera 0 — is a demo still holding it?"
    ok, f = cap.read()
    h, w = f.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    print(f"recording {secs:.0f}s at {w}x{h}", flush=True)
    t0, n = time.time(), 0
    while time.time() - t0 < secs:
        ok, f = cap.read()
        if not ok:
            break
        writer.write(f)
        n += 1
    writer.release()
    cap.release()
    print(f"wrote {n} frames to {out_path}", flush=True)


if __name__ == "__main__":
    main()
