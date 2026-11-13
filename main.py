import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RIGHT_EYE = {"outer": 33,  "inner": 133, "top": 159, "bottom": 145, "iris": 468}
LEFT_EYE  = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473}
COLORS = {"outer": (0, 0, 255), "inner": (255, 0, 0), "top": (0, 255, 255),
          "bottom": (255, 0, 255), "iris": (0, 255, 0)}

def px(face, idx, w, h):
    lm = face[idx]
    return (lm.x * w, lm.y * h)

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def main():

    # Open the camera
    cap = cv.VideoCapture(0)

    # Check if opened
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create sliders for adjusting the Canny thresholds in real time.
    cv.namedWindow("Camera Feed")
    cv.createTrackbar("Lower threshold", "Camera Feed", 50, 255, lambda _: None)
    cv.createTrackbar("Upper threshold", "Camera Feed", 200, 255, lambda _: None)

    # Begin camera processing loop
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # Capture frame by frame
            ret, frame = cap.read()

            # Check if can read frame
            if not ret:
                print("Error: Could not read frame. Exiting.")
                break

            # Perform processing here
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamps must be strictly increasing
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            for face in result.face_landmarks:
                for eye in (RIGHT_EYE, LEFT_EYE):
                    pts = {name: px(face, idx, w, h) for name, idx in eye.items()}

                    for name, (x, y) in pts.items():
                        cv.circle(frame, (int(x), int(y)), 3, COLORS[name], -1)

            # Display the processed frame
            cv.imshow("Camera Feed", frame)

            # Exit on 'q' key
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # End
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
