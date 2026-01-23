import cv2
import numpy as np
import time

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
# STAGE 3: GOP SETTINGS
# =========================
GOP_SIZE = 8
frame_index = 0
prev_recon_Y = None

def decide_frame_type(idx, gop):
    return "I" if idx % gop == 0 else "P"

# =========================
# STAGE 4: DCT HELPERS
# =========================
BLOCK = 8

def block_process(channel):
    """Apply 8x8 DCT + IDCT on a channel"""
    h, w = channel.shape
    recon = np.zeros_like(channel, dtype=np.float32)

    for y in range(0, h, BLOCK):
        for x in range(0, w, BLOCK):
            block = channel[y:y+BLOCK, x:x+BLOCK]

            if block.shape != (8, 8):
                continue

            # Level shift
            block = block.astype(np.float32) - 128.0

            # Forward DCT
            dct = cv2.dct(block)

            # Inverse DCT
            idct = cv2.idct(dct)

            # Inverse level shift
            recon[y:y+BLOCK, x:x+BLOCK] = idct + 128.0

    return np.clip(recon, 0, 255).astype(np.uint8)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------
    # STAGE 1: RGB → YCbCr
    # -----------------
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    # -----------------
    # STAGE 2: 4:2:0 SUBSAMPLING
    # -----------------
    h, w = Y.shape
    Cb_420 = cv2.resize(Cb, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    Cr_420 = cv2.resize(Cr, (w//2, h//2), interpolation=cv2.INTER_LINEAR)

    # -----------------
    # STAGE 3: I / P FRAME LOGIC
    # -----------------
    frame_type = decide_frame_type(frame_index, GOP_SIZE)
    frame_index += 1

    if frame_type == "I" or prev_recon_Y is None:
        prediction_Y = np.zeros_like(Y)
    else:
        prediction_Y = prev_recon_Y

    residual_Y = Y.astype(np.int16) - prediction_Y.astype(np.int16)

    # -----------------
    # STAGE 4: TRANSFORM (DCT)
    # -----------------
    recon_residual_Y = block_process(residual_Y.astype(np.uint8))
    recon_Y = prediction_Y + recon_residual_Y
    recon_Y = np.clip(recon_Y, 0, 255).astype(np.uint8)

    # Save reference
    prev_recon_Y = recon_Y.copy()

    # -----------------
    # VISUALIZATION
    # -----------------
    vis = frame.copy()
    cv2.putText(vis, f"Frame {frame_index} | Type: {frame_type}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.imshow("Original", vis)
    cv2.imshow("Y - Luma (Stage 1)", Y)
    cv2.imshow("Residual Y (Stage 3)", np.clip(residual_Y + 128, 0, 255).astype(np.uint8))
    cv2.imshow("Reconstructed Y (Stage 4)", recon_Y)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
