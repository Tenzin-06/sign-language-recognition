import cv2
cap = cv2.VideoCapture(0)          # on Windows, if this fails try: cv2.VideoCapture(0, cv2.CAP_DSHOW)
print('opened:', cap.isOpened())
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow('webcam (q to quit)', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()