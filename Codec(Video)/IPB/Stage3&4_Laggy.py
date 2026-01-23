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

GOP_SIZE = 8           # I-frame every 8 frames
frame_index = 0
prev_recon_Y = None    # reference frame placeholder (for next stage)

def decide_frame_type(frame_index, gop_size):
    return "I" if frame_index % gop_size == 0 else "P"

# =========================
# STAGE 4: BLOCK + DCT
# =========================

BLOCK_SIZE = 8

def split_into_blocks(channel, block_size=8):
    h, w = channel.shape
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = channel[y:y+block_size, x:x+block_size]
            if block.shape == (block_size, block_size):
                blocks.append(block)
    return blocks

def level_shift(block):
    return block.astype(np.float32) - 128.0

def dct2(block):
    return cv2.dct(block)


while True:
    ret, frame = cap.read()
    if not ret:
        print(" Frame not received")
        break

    now = time.perf_counter()
    frame_times.append(now - last_ts)
    last_ts = now

    avg_dt = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

    #This is the parth where we Convert Image from BGR to YCbCr
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    cv2.putText(Y, "Y (Luma)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    cv2.putText(Cb, "Cb (Blue Chroma)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    cv2.putText(Cr, "Cr (Red Chroma)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    # Here we are visualizaing the Y Cb and Cr which we extracte from RGB
    cv2.imshow("Y - Luma", Y)
    cv2.imshow("Cb - Chroma", Cb)
    cv2.imshow("Cr - Chroma", Cr) 

    h, w = Y.shape


    Cb_420 = cv2.resize(Cb, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    Cr_420 = cv2.resize(Cr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    Cb_up = cv2.resize(Cb_420, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_420, (w, h), interpolation=cv2.INTER_LINEAR)

    ycbcr_420 = cv2.merge((Y, Cr_up, Cb_up))
    recon = cv2.cvtColor(ycbcr_420, cv2.COLOR_YCrCb2BGR)

    cv2.imshow("Original", frame)
    cv2.imshow("After 4:2:0", recon)

    frame_type = decide_frame_type(frame_index, GOP_SIZE)
    frame_index += 1
    cv2.putText(
        frame,
        f"FRAME TYPE: {frame_type}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    if frame_type == "I":
        prev_recon_Y = Y.copy()
    elif frame_type == "P" and prev_recon_Y is not None:
        pass  # motion estimation will use prev_recon_Y in next stage

    Cb_420 = cv2.resize(Cb, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    Cr_420 = cv2.resize(Cr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    # =========================
    # STAGE 4: BLOCKING + DCT
    # =========================

    # Split into 8x8 blocks
    Y_blocks  = split_into_blocks(Y, BLOCK_SIZE)
    Cb_blocks = split_into_blocks(Cb_420, BLOCK_SIZE)
    Cr_blocks = split_into_blocks(Cr_420, BLOCK_SIZE)

    # Apply level shift + DCT (Y only for now)
    Y_dct_blocks = []

    for block in Y_blocks:
        shifted = level_shift(block)
        dct_block = dct2(shifted)
        Y_dct_blocks.append(dct_block)

    # Visualize first DCT block once
    if frame_index == 1:
        dct_vis = np.log(np.abs(Y_dct_blocks[0]) + 1)
        dct_vis = cv2.normalize(dct_vis, None, 0, 255, cv2.NORM_MINMAX)
        dct_vis = dct_vis.astype(np.uint8)

        cv2.imshow("DCT Magnitude (Y Block)", dct_vis)

        
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
 