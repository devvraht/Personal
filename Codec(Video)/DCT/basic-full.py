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
# STAGE 3 SETTINGS
# =========================
GOP_SIZE = 8
frame_index = 0
prev_recon_Y = None

# =========================
# STAGE 4 SETTINGS (DCT)
# =========================
BLOCK_SIZE = 8

def decide_frame_type(idx, gop):
    return "I" if idx % gop == 0 else "P"

def level_shift(block):
    return block.astype(np.float32) - 128.0

def inv_level_shift(block):
    return np.clip(block + 128.0, 0, 255).astype(np.uint8)

def block_dct(frame_Y):
    h, w = frame_Y.shape
    dct_frame = np.zeros_like(frame_Y, dtype=np.float32)
    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            block = frame_Y[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            if block.shape[0] != BLOCK_SIZE or block.shape[1] != BLOCK_SIZE:
                continue
            shifted = level_shift(block)
            dct_block = cv2.dct(shifted)
            dct_frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = dct_block
    return dct_frame

def block_idct(dct_frame):
    h, w = dct_frame.shape
    rec_frame = np.zeros_like(dct_frame, dtype=np.float32)
    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            block = dct_frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            if block.shape[0] != BLOCK_SIZE or block.shape[1] != BLOCK_SIZE:
                continue
            idct_block = cv2.idct(block)
            rec_frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = idct_block
    return inv_level_shift(rec_frame)

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
    # STAGE 1: COLORSPACE
    # -----------------
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    # -----------------
    # STAGE 2: CHROMA SUBSAMPLING 4:2:0
    # -----------------
    h, w = Y.shape
    Cb_420 = cv2.resize(Cb, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    Cr_420 = cv2.resize(Cr, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    # Optional upsample for display
    Cb_up = cv2.resize(Cb_420, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_420, (w, h), interpolation=cv2.INTER_LINEAR)
    ycbcr_420_vis = cv2.merge((Y, Cr_up, Cb_up))
    recon_420 = cv2.cvtColor(ycbcr_420_vis, cv2.COLOR_YCrCb2BGR)

    # -----------------
    # STAGE 3: FRAME TYPE
    # -----------------
    frame_type = decide_frame_type(frame_index, GOP_SIZE)
    frame_index += 1

    # -----------------
    # STAGE 4: DCT + Residual for P-frame
    # -----------------
    if frame_type == "I" or prev_recon_Y is None:
        # I-frame: full-frame DCT
        dct_frame = block_dct(Y)
        recon_Y = block_idct(dct_frame)
        prev_recon_Y = recon_Y.copy()
    else:
        # P-frame: compute residual
        residual = Y.astype(np.int16) - prev_recon_Y.astype(np.int16)
        residual_dct = block_dct(residual)
        # Reconstruct
        rec_residual = block_idct(residual_dct)
        recon_Y = np.clip(prev_recon_Y.astype(np.int16) + rec_residual.astype(np.int16), 0, 255).astype(np.uint8)
        prev_recon_Y = recon_Y.copy()

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
