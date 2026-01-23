import cv2
import time
from collections import deque

# =========================
# CONFIG
# =========================
RTSP_URL = "rtsp://127.0.0.1:8554/live"
GOP_SIZE = 30          # 1 I-frame every 30 frames (1 sec @ 30fps)
FPS_SMOOTHING = 30

# =========================
# OPEN STREAM
# =========================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# IMPORTANT FOR LOW LATENCY
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream")
    exit(1)

# =========================
# STAGE 3 STATE
# =========================
frame_index = 0
prev_ref_frame = None

frame_times = deque(maxlen=FPS_SMOOTHING)
last_time = time.perf_counter()

# =========================
# FRAME TYPE DECISION
# =========================
def frame_type(index, gop):
    return 'I' if index % gop == 0 else 'P'

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue   # DO NOT break (RTSP hiccups)

    now = time.perf_counter()
    frame_times.append(now - last_time)
    last_time = now

    fps = len(frame_times) / sum(frame_times)

    # -------- Stage 3 --------
    ftype = frame_type(frame_index, GOP_SIZE)

    if ftype == 'I':
        prev_ref_frame = frame   # reference update
    else:
        # P-frame: reference exists, no processing yet
        pass

    frame_index += 1

    # -------- Display --------
    cv2.putText(
        frame,
        f"Frame: {frame_index} | Type: {ftype} | FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Stage 3 - Temporal Structure", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
