import cv2
import time
from collections import deque
import numpy as np

RTSP_URL = "rtsp://127.0.0.1:8554/live"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("RTSP Opening Problem")
    exit(1)

frame_times = deque(maxlen=30)
last_ts = time.perf_counter()

# =========================
# STAGE 3: FRAME TYPE LOGIC
# =========================
GOP_SIZE = 8
frame_index = 0
prev_recon_Y = None

def decide_frame_type(idx, gop):
    return "I" if idx % gop == 0 else "P"

# =========================
# STAGE 4: DCT
# =========================
BLOCK_SIZE = 8
VIS_EVERY_N_FRAMES = 10

def level_shift(block):
    return block.astype(np.float32) - 128.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.perf_counter()
    frame_times.append(now - last_ts)
    last_ts = now

    # -------- Stage 1 + 2 --------
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    h, w = Y.shape

    # -------- 4:2:0 subsampling --------
    Cb_420 = cv2.resize(Cb, (w // 2, h // 2))
    Cr_420 = cv2.resize(Cr, (w // 2, h // 2))

    # -------- Stage 3 --------
    frame_type = decide_frame_type(frame_index, GOP_SIZE)
    frame_index += 1

    if frame_type == "I":
        prev_recon_Y = Y.copy()

    # -------- Stage 4: DCT (sample block) --------
    sample_block = Y[0:8, 0:8]
    shifted = level_shift(sample_block)
    dct_block = cv2.dct(shifted)

    # ALWAYS show video (smooth playback)
    cv2.imshow("Original", frame)
    cv2.imshow("Y - Luma", Y)

    # Throttle only the DCT visualization
    if frame_index % VIS_EVERY_N_FRAMES == 0:
        dct_vis = np.log(np.abs(dct_block) + 1)
        dct_vis = cv2.normalize(dct_vis, None, 0, 255, cv2.NORM_MINMAX)
        dct_vis = dct_vis.astype(np.uint8)
        cv2.imshow("DCT Magnitude (8x8 Y Block)", dct_vis)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
