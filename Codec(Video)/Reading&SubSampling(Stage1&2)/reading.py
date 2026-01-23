import cv2
import time
from collections import deque

RTSP_URL = "rtsp://127.0.0.1:8554/live"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("RTSP Opening Problem")
    exit(1)

frame_times = deque(maxlen=30)
last_ts = time.perf_counter()

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

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
