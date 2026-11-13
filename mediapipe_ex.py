import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be strictly increasing
        timestamp_ms = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        for face in result.face_landmarks:
            for lm in face:  # 478 points, including irises
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

        cv2.imshow("MediaPipe Face Landmarker", cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()