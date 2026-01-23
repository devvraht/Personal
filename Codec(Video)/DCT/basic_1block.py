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

# =========================
# STAGE 4 SETTINGS (DCT)
# =========================
BLOCK_SIZE = 8
VIS_EVERY_N_FRAMES = 5  # visualize DCT every N frames

def decide_frame_type(idx, gop):
    """Decide I or P frame"""
    return "I" if idx % gop == 0 else "P"

def level_shift(block):
    """Shift 8-bit block to signed range for DCT"""
    return block.astype(np.float32) - 128.0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    # -----------------
    # FPS tracking
    # -----------------
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

    # Optional visualization for Stage 2
    Cb_up = cv2.resize(Cb_420, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_420, (w, h), interpolation=cv2.INTER_LINEAR)
    ycbcr_420_vis = cv2.merge((Y, Cr_up, Cb_up))
    recon_420 = cv2.cvtColor(ycbcr_420_vis, cv2.COLOR_YCrCb2BGR)

    # -----------------
    # STAGE 3: FRAME TYPE LOGIC
    # -----------------
    frame_type = decide_frame_type(frame_index, GOP_SIZE)
    frame_index += 1

    if frame_type == "I" or prev_recon_Y is None:
        prev_recon_Y = Y.copy()
        recon_Y = Y.copy()
    else:
        recon_Y = prev_recon_Y.copy()

    # -----------------
    # STAGE 4: DCT (on sample blocks)
    # -----------------
    if frame_index % VIS_EVERY_N_FRAMES == 0:
        # Take top-left 8x8 block as a demo
        sample_block = Y[0:BLOCK_SIZE, 0:BLOCK_SIZE]
        shifted = level_shift(sample_block)
        dct_block = cv2.dct(shifted)

        # Visualization: log scale + normalize
        dct_vis = np.log(np.abs(dct_block)+1)
        dct_vis = cv2.normalize(dct_vis, None, 0, 255, cv2.NORM_MINMAX)
        dct_vis = dct_vis.astype(np.uint8)

        cv2.imshow("DCT Magnitude (8x8 Y Block)", dct_vis)

    # -----------------
    # VISUALIZATION
    # -----------------
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Original | FrameType: {frame_type} | FPS: {fps:.1f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    Y_vis = Y.copy()
    cv2.putText(Y_vis, f"Y - Luma", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    if frame_type == "I":
        cv2.imshow("I-Frame (Reconstructed Y)", recon_Y)
        if cv2.getWindowProperty("P-Frame (Reconstructed Y)", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("P-Frame (Reconstructed Y)")
    else:
        cv2.imshow("P-Frame (Reconstructed Y)", recon_Y)
        if cv2.getWindowProperty("I-Frame (Reconstructed Y)", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("I-Frame (Reconstructed Y)")

    # Always show original and Y
    cv2.imshow("Original", display_frame)
    cv2.imshow("Y - Luma (Stage 1)", Y_vis)
    cv2.imshow("Chroma Reconstructed (Stage 2)", recon_420)

    # -----------------
    # EXIT CONDITION
    # -----------------
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
