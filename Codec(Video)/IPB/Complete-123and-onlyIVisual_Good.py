import cv2
import numpy as np
import time
from collections import deque

# =========================
# INPUT
# =========================
RTSP_URL = "rtsp://127.0.0.1:8554/live"
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("RTSP Opening Problem")
    exit(1)

# =========================
# TIMING / FPS
# =========================
frame_times = deque(maxlen=30)
last_ts = time.perf_counter()

# =========================
# STAGE 3 SETTINGS (I/P FRAME)
# =========================
GOP_SIZE = 8
frame_index = 0
prev_recon_Y = None

def decide_frame_type(idx, gop):
    """Decide I or P frame"""
    return "I" if idx % gop == 0 else "P"

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    # FPS tracking
    now = time.perf_counter()
    frame_times.append(now - last_ts)
    last_ts = now
    fps = 1.0 / (sum(frame_times)/len(frame_times)) if frame_times else 0.0

    # -----------------
    # STAGE 1: COLORSPACE (RGB -> YCbCr)
    # -----------------
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    # -----------------
    # STAGE 2: CHROMA SUBSAMPLING 4:2:0
    # -----------------
    h, w = Y.shape
    Cb_420 = cv2.resize(Cb, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    Cr_420 = cv2.resize(Cr, (w//2, h//2), interpolation=cv2.INTER_LINEAR)

    # For visualization, upsample back to original size
    Cb_up = cv2.resize(Cb_420, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_420, (w, h), interpolation=cv2.INTER_LINEAR)
    ycbcr_420_vis = cv2.merge((Y, Cr_up, Cb_up))
    recon_420 = cv2.cvtColor(ycbcr_420_vis, cv2.COLOR_YCrCb2BGR)

    # -----------------
    # STAGE 3: TEMPORAL STRUCTURE (I/P FRAME)
    # -----------------
    frame_type = decide_frame_type(frame_index, GOP_SIZE)
    frame_index += 1

    if frame_type == "I" or prev_recon_Y is None:
        # I-frame: reference is itself
        prev_recon_Y = Y.copy()
        recon_Y = Y.copy()
    else:
        # P-frame: currently no motion, just copy previous
        recon_Y = prev_recon_Y.copy()

    # -----------------
    # VISUALIZATION
    # -----------------
    cv2.putText(frame, f"Original | FrameType: {frame_type} | FPS: {fps:.1f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(Y, f"Y - Luma", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(Cb_up, f"Cb - Chroma (4:2:0)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(Cr_up, f"Cr - Chroma (4:2:0)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Y - Luma (Stage 1)", Y)
    cv2.imshow("Cb - Chroma Upsampled (Stage 2)", Cb_up)
    cv2.imshow("Cr - Chroma Upsampled (Stage 2)", Cr_up)
    cv2.imshow("Chroma Reconstructed (Stage 2)", recon_420)
    cv2.imshow("Reconstructed Y (Stage 3)", recon_Y)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
